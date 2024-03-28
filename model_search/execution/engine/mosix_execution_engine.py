import time

import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, BATCH_MEASURES, INPUT, LABEL, CUDA, \
    MODEL_TO_DEVICE
from global_utils.model_operations import split_model
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass, \
    CachedDatasetInformation
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine, load_data_to_device, \
    inference, load_model_to_device
from model_search.execution.planning.execution_plan import ExecutionStep, ScoreModelStep
from model_search.execution.planning.mosix_planner import MosixExtractFeaturesStep, CachingConfig
from model_search.executionsteplogger import ExecutionStepLogger

BASE_MODEL = 'base_model'


def cat_tensors(tensors: [torch.Tensor]):
    # TODO don't know yet if its smart to concat the tensors, because most likely we create a copy even is this really
    #  faster than just using a smaller batch size?
    return torch.cat(tensors, dim=0)


class MosixExecutionEngine(BaselineExecutionEngine):

    def __init__(self, tensor_caching_service: CachingService,
                 model_caching_service: CachingService):
        super().__init__(tensor_caching_service)
        self.model_caching_service = model_caching_service
        self._prev_layer_hashes = []
        self._prev_sub_model = None

    def execute_step(self, exec_step: ExecutionStep):
        # reset logger for every step
        self.logger = ExecutionStepLogger()
        if isinstance(exec_step, MosixExtractFeaturesStep):
            self.execute_mosix_extract_features_step(exec_step)
        elif isinstance(exec_step, ScoreModelStep):
            self.execute_score_model_step(exec_step)
        else:
            raise TypeError

    def execute_mosix_extract_features_step(self, exec_step: MosixExtractFeaturesStep):
        partial_model = self._init_model(exec_step)
        data_loader = self._get_data_loader(exec_step)
        cache_conf = exec_step.cache_config
        cache_labels = exec_step.cache_labels
        self._extract_features_part_model(partial_model, data_loader, cache_conf, cache_labels)

    def _extract_features_part_model(self, partial_model, data_loader, cache_conf: CachingConfig, cache_labels):

        # extract features
        start = time.perf_counter()
        for i, (batch) in enumerate(data_loader):

            if cache_labels:
                inputs, labels = batch
            else:
                inputs = batch

            batch_measures = {}
            batch_measures[LOAD_DATA] = time.perf_counter() - start

            measurement, inputs = self.bench.micro_benchmark_cpu(load_data_to_device, inputs)
            batch_measures[DATA_TO_DEVICE] = measurement

            partial_model.eval()
            measurement, features = self.bench.micro_benchmark_gpu(inference, inputs, partial_model)
            batch_measures[INFERENCE] = measurement

            features_cache_id = f'{cache_conf.id_prefix}-{INPUT}-{i}'
            self.caching_service.cache_on_location(features_cache_id, features, cache_conf.location)

            if cache_labels:
                labels_cache_id = f'{cache_conf.id_prefix}-{LABEL}-{i}'
                self.caching_service.cache_on_location(labels_cache_id, labels, cache_conf.location)

            self.logger.append_value(BATCH_MEASURES, batch_measures)

            start = time.perf_counter()
        # TODO fix the logging
        # exec_step.execution_logs = self.logger

    def _get_data_loader(self, exec_step: MosixExtractFeaturesStep):
        if (isinstance(exec_step.data_info, DatasetInformation)
                and exec_step.data_info.data_set_class == DatasetClass.CUSTOM_IMAGE_FOLDER):
            data = CustomImageFolder(exec_step.data_info.dataset_path, exec_step.data_info.transform)
            # init data loader
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=exec_step.data_info.batch_size, shuffle=False,
                num_workers=exec_step.data_info.num_workers
            )

            if exec_step.data_range is not None:
                data.set_subrange(exec_step.data_range[0], exec_step.data_range[1])

        elif isinstance(exec_step.data_info, CachedDatasetInformation):
            data = CacheServiceDataset(
                self.caching_service,
                [f'{exec_step.data_info.data_prefix}-{INPUT}'],
                None
            )
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=exec_step.data_info.batch_size, shuffle=False,
                num_workers=exec_step.data_info.num_workers, collate_fn=cat_tensors
            )
        else:
            raise NotImplementedError
        return data_loader

    def _init_model(self, exec_step):
        relevant_layers = self._get_relevant_layers(exec_step)

        if self._prev_layer_hashes == [layer.state_dict_hash for layer in relevant_layers]:
            return self._prev_sub_model
        elif not set(self._prev_layer_hashes).isdisjoint([layer.state_dict_hash for layer in relevant_layers]):
            raise NotImplementedError

        # get a base model we can load parameters and that we split later to get the sub model
        if not self.model_caching_service.id_exists(BASE_MODEL):
            model: torch.nn.Module = self.initialize_model(exec_step.model_snapshot)
            self.model_caching_service.cache_on_cpu(BASE_MODEL, model)
        else:
            model: torch.nn.Module = self.model_caching_service.get_item(BASE_MODEL)

        # only load the state of the updated/relevant layers
        self._prev_layer_hashes = []
        for layer in relevant_layers:
            state_dict = torch.load(layer.state_dict_path)
            model.load_state_dict(state_dict, strict=False)
            self._prev_layer_hashes.append(layer.state_dict_hash)

        # split the base model to extract the model we are interested in
        sub_models = split_model(model, exec_step.layer_range)
        # at index 0 is always the base part of the model that we want to skip
        # at index above 1 is always the part we will process in a later step
        sub_model = sub_models[1]

        # load the model to the device
        measurement, _ = self.bench.micro_benchmark_cpu(load_model_to_device, sub_model, CUDA)
        self.logger.log_value(MODEL_TO_DEVICE, measurement)

        self._prev_sub_model = sub_model

        return sub_model

    def _get_relevant_layers(self, exec_step):
        layer_range = exec_step.layer_range
        if exec_step.open_layer_range:
            relevant_layers = exec_step.model_snapshot.layer_states[layer_range[0]:]
        else:
            relevant_layers = exec_step.model_snapshot.layer_states[layer_range[0]: layer_range[1]]
        return relevant_layers
