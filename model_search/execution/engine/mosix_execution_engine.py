import time

import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, BATCH_MEASURES, INPUT, LABEL, MODEL_TO_DEVICE, \
    CUDA, CALC_PROXY_SCORE, TRAIN, TEST
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass, \
    CachedDatasetInformation
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine, load_data_to_device, \
    inference, load_model_to_device
from model_search.execution.planning.execution_plan import ExecutionStep, ScoreModelStep
from model_search.execution.planning.mosix_planner import MosixExtractFeaturesStep, CachingConfig
from model_search.executionsteplogger import ExecutionStepLogger
from model_search.model_management.model_store import ModelStore
from model_search.proxies.nn_proxy import linear_proxy

BASE_MODEL = 'base_model'


def cat_tensors(tensors: [torch.Tensor]):
    # TODO don't know yet if its smart to concat the tensors, because most likely we create a copy even is this really
    #  faster than just using a smaller batch size?
    return torch.cat(tensors, dim=0)


class MosixExecutionEngine(BaselineExecutionEngine):

    def __init__(self, tensor_caching_service: CachingService,
                 model_caching_service: CachingService, model_store: ModelStore):
        super().__init__(tensor_caching_service)
        self.model_caching_service = model_caching_service
        self.model_store = model_store
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
        extract_labels = exec_step.extract_labels
        dataset_type = exec_step.data_info.dataset_type
        self._extract_features_part_model(partial_model, data_loader, cache_conf, extract_labels, dataset_type)

    def _extract_features_part_model(self, partial_model, data_loader, cache_conf: CachingConfig, extract_labels,
                                     dataset_type):

        # extract features
        start = time.perf_counter()
        for i, (batch) in enumerate(data_loader):

            if extract_labels:
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

            features_cache_id = f'{cache_conf.id_prefix}-{dataset_type}-{INPUT}-{i}'
            self.caching_service.cache_on_location(features_cache_id, features, cache_conf.location)

            if extract_labels:
                labels_cache_id = f'{dataset_type}-{LABEL}-{i}'
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
            data_info = exec_step.data_info
            data = CacheServiceDataset(
                self.caching_service,
                [f'{exec_step.input_node_id}-{data_info.dataset_type}-{INPUT}'],
                None
            )
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=1, shuffle=False,
                num_workers=exec_step.data_info.num_workers, collate_fn=cat_tensors
            )
        else:
            raise NotImplementedError
        return data_loader

    def _init_model(self, exec_step):
        layer_state_ids = [l.id for l in exec_step.layers]
        sub_model = self.model_store.get_composed_model(layer_state_ids)

        measurement, _ = self.bench.micro_benchmark_cpu(load_model_to_device, sub_model, CUDA)
        self.logger.log_value(MODEL_TO_DEVICE, measurement)

        return sub_model

    # TODO unify/refactor so that the following two methods are not duplicated code with baseline
    def execute_score_model_step(self, exex_step: ScoreModelStep):
        train_loader = self._get_proxy_data_loader(exex_step.train_feature_cache_prefixes, TRAIN)
        test_loader = self._get_proxy_data_loader(exex_step.test_feature_cache_prefixes, TEST)
        print(len(test_loader))
        print(len(train_loader))
        measurement, score = self.bench.benchmark_end_to_end(
            linear_proxy, train_loader, test_loader, exex_step.num_classes, device=torch.device(CUDA)
        )
        self.logger.log_value(CALC_PROXY_SCORE, measurement)
        exex_step.execution_result = {'score': score}
        exex_step.execution_logs = self.logger

    def _get_proxy_data_loader(self, cache_prefixes, dataset_type=None):
        data = CacheServiceDataset(
            self.caching_service,
            cache_prefixes,
            [f'{dataset_type}-{LABEL}']
        )
        data_loader = DataLoader(data, batch_size=1, num_workers=2)
        return data_loader
