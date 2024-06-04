import time

import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, BATCH_MEASURES, LABEL, MODEL_TO_DEVICE, \
    CUDA, CALC_PROXY_SCORE, TRAIN, TEST, GET_COMPOSED_MODEL
from global_utils.deterministic import check_deterministic_env_var_set
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass, \
    CachedDatasetInformation
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine, load_data_to_device, \
    inference, load_model_to_device
from model_search.execution.planning.execution_plan import ExecutionStep, ScoreModelStep, ModifyCacheStep
from model_search.execution.planning.mosix_planner import MosixExtractFeaturesStep
from model_search.executionsteplogger import ExecutionStepLogger
from model_search.model_management.model_store import ModelStore
from model_search.proxies.nn_proxy import linear_proxy


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
        elif isinstance(exec_step, ModifyCacheStep):
            self.execute_modify_cache_step(exec_step)
        else:
            raise TypeError

        return self.logger.log_dict

    def execute_mosix_extract_features_step(self, exec_step: MosixExtractFeaturesStep):
        partial_model = self._init_model(exec_step)
        data, data_loader = self._get_data_loader(exec_step)
        self._extract_features_part_model(partial_model, data, data_loader, exec_step)

        # clear GPU memory after step to remove partial model
        torch.cuda.empty_cache()

    def _extract_features_part_model(self, partial_model, data_set, data_loader, exec_step):

        if isinstance(data_set, CacheServiceDataset):
            self._extract_features_part_model_no_data_loader(data_set, exec_step, partial_model)
        else:
            self._extract_features_part_model_data_loader(data_loader, data_set, exec_step, partial_model)

    def _extract_features_part_model_data_loader(self, data_loader, data_set, exec_step, partial_model):

        partial_model.eval()

        start = time.perf_counter()

        for i, (batch) in enumerate(data_loader):

            if exec_step.extract_labels:
                inputs, labels = batch
            else:
                inputs = batch

            if isinstance(data_set, CacheServiceDataset):
                inputs = data_set.translate_to_actual_data(inputs)

            batch_measures = {}
            batch_measures[LOAD_DATA] = time.perf_counter() - start

            self._load_inf_cache(partial_model, inputs, exec_step, i, batch_measures)

            if exec_step.extract_labels:
                labels_cache_id = f'{exec_step.label_write_cache_config.id_prefix}-{i}'
                self.caching_service.cache_on_location(labels_cache_id, labels,
                                                       exec_step.label_write_cache_config.location)

            self.logger.append_value(BATCH_MEASURES, batch_measures)

            start = time.perf_counter()

    def _load_inf_cache(self, partial_model, inputs, exec_step, i, batch_measures):
        measurement, inputs = self.bench.micro_benchmark_cpu(load_data_to_device, inputs)
        batch_measures[DATA_TO_DEVICE] = measurement
        measurement, features = self.bench.micro_benchmark_gpu(inference, inputs, partial_model)
        batch_measures[INFERENCE] = measurement
        features_cache_id = f'{exec_step.inp_write_cache_config.id_prefix}-{i}'
        self.caching_service.cache_on_location(features_cache_id, features,
                                               exec_step.inp_write_cache_config.location)

    def _extract_features_part_model_no_data_loader(self, data_set, exec_step, partial_model):

        partial_model.eval()

        start = time.perf_counter()

        data_ids = data_set._data_ids
        for i, (data_id) in enumerate(data_ids):
            # TODO this significantly speeds up data loading when loading from GPU memory, when loading from slow
            # (persistent) storage we might want to use some background processes again like in a classical
            # PyTorch data loader
            inputs = self.caching_service.get_item(data_id)

            batch_measures = {}
            batch_measures[LOAD_DATA] = time.perf_counter() - start

            self._load_inf_cache(partial_model, inputs, exec_step, i, batch_measures)

            self.logger.append_value(BATCH_MEASURES, batch_measures)

            start = time.perf_counter()

    def _get_label_cache_prefix(self, exec_step):
        prefix = f'{exec_step.data_info.dataset_type}-{LABEL}'
        if exec_step.data_info.dataset_type == TRAIN:
            prefix += f'-{exec_step.data_range[0]}'
        return prefix

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
            data_prefix = exec_step.inp_read_cache_config.id_prefix

            data = CacheServiceDataset(
                self.caching_service,
                [data_prefix],
                None
            )
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=1, shuffle=False, num_workers=exec_step.data_info.num_workers
            )
        else:
            raise NotImplementedError
        return data, data_loader

    def _init_model(self, exec_step):
        layer_state_ids = [l.id for l in exec_step.layers]
        measurement, sub_model = self.bench.micro_benchmark_cpu(self.model_store.get_composed_model, layer_state_ids)
        self.logger.log_value(GET_COMPOSED_MODEL, measurement)

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
        if check_deterministic_env_var_set():
            num_workers = 0
        else:
            num_workers = 2
        data_loader = DataLoader(data, batch_size=1, num_workers=num_workers)
        return data_loader

    def execute_modify_cache_step(self, exec_step):
        for _id in exec_step.cache_evict_ids:
            self.caching_service.remove_all_ids_with_prefix(_id, remove_immediately=False)

        torch.cuda.empty_cache()
