import time

import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.models.init_models import initialize_model
from data.imdb.reduced_imdb import get_imbdb_bert_base_uncased_datasets
from global_utils.constants import LOAD_STATE_DICT, INIT_MODEL, STATE_TO_MODEL, MODEL_TO_DEVICE, CUDA, INPUT, LABEL, \
    LOAD_DATA, DATA_TO_DEVICE, INFERENCE, BATCH_MEASURES, CALC_PROXY_SCORE
from global_utils.deterministic import check_deterministic_env_var_set
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.abstract_execution_engine import ExecutionEngine
from model_search.execution.planning.execution_plan import ExecutionStep, BaselineExtractFeaturesStep, ScoreModelStep
from model_search.executionsteplogger import ExecutionStepLogger
from model_search.model_management.model_store import match_keys
from model_search.model_snapshots.base_snapshot import ModelSnapshot
from model_search.proxies.nn_proxy import linear_proxy


def load_model_state(state_dict_path):
    state_dict = torch.load(state_dict_path)
    return state_dict


def load_state_dict_in_model(model, state_dict):
    if next(iter(state_dict)).startswith("0.0"):
        state_dict = match_keys(model.state_dict(), state_dict)

    model.load_state_dict(state_dict)


def load_model_to_device(model, device):
    model.to(device)


def load_data_to_device(batch):
    if isinstance(batch, list):
        batch = [batch[0].to(CUDA), batch[1].to(CUDA)]
    else:
        batch = batch.to(CUDA)
    return batch


def inference(batch, model):
    with torch.no_grad():
        out = model(batch)
        return out


class BaselineExecutionEngine(ExecutionEngine):

    def __init__(self, cachingService: CachingService):

        super().__init__(cachingService)

    def load_snapshot(self, snapshot: ModelSnapshot):
        state_dict_path = snapshot.state_dict_path
        measurement, state_dict = self.bench.micro_benchmark_cpu(load_model_state, state_dict_path)
        self.logger.log_value(LOAD_STATE_DICT, measurement)

        return state_dict

    def initialize_model(self, snapshot: ModelSnapshot):
        kwargs = {'pretrained': False, 'new_num_classes': None, 'features_only': True, 'sequential_model': True,
                  'freeze_feature_extractor': False}
        measurement, model = self.bench.micro_benchmark_cpu(initialize_model, snapshot.architecture_id, **kwargs)
        self.logger.log_value(INIT_MODEL, measurement)

        return model

    def init_model(self, snapshot: ModelSnapshot):
        # load the snapshot from storage
        state_dict = self.load_snapshot(snapshot)
        # initialize model
        model = self.initialize_model(snapshot)
        # load state to model
        measurement, _ = self.bench.micro_benchmark_cpu(load_state_dict_in_model, model, state_dict)
        self.logger.log_value(STATE_TO_MODEL, measurement)
        # load model to device, measure cpu time
        measurement, _ = self.bench.micro_benchmark_cpu(load_model_to_device, model, CUDA)
        self.logger.log_value(MODEL_TO_DEVICE, measurement)

        return model

    def execute_step(self, exec_step: ExecutionStep):
        # reset logger for every step
        self.logger = ExecutionStepLogger()

        if isinstance(exec_step, BaselineExtractFeaturesStep):
            self.execute_baseline_extract_features_step(exec_step)
        elif isinstance(exec_step, ScoreModelStep):
            self.execute_score_model_step(exec_step)
        else:
            raise TypeError

        return self.logger.log_dict

    def execute_baseline_extract_features_step(self, exec_step: BaselineExtractFeaturesStep):

        # init data loader
        if exec_step.inp_data.data_set_class == DatasetClass.CUSTOM_IMAGE_FOLDER:
            data = CustomImageFolder(exec_step.inp_data.dataset_path, exec_step.inp_data.transform)
        elif exec_step.inp_data.data_set_class == DatasetClass.IMDB:
            data = get_imbdb_bert_base_uncased_datasets(exec_step.inp_data.dataset_path)

        else:
            raise NotImplementedError

        self._extract_features(data, exec_step)

    def _extract_features(self, data, exec_step):

        # initialize model
        model = self.init_model(exec_step.model_snapshot)

        # init data loader
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=exec_step.inp_data.batch_size, shuffle=False,
            num_workers=exec_step.inp_data.num_workers)

        # extract features
        start = time.perf_counter()
        for i, data in enumerate(data_loader):
            if len(data) == 2:
                inputs, labels = data
            else:
                inputs, labels = data[:-1], data[-1]
            batch_measures = {}
            batch_measures[LOAD_DATA] = time.perf_counter() - start

            measurement, inputs = self.bench.micro_benchmark_cpu(load_data_to_device, inputs)
            batch_measures[DATA_TO_DEVICE] = measurement

            model.eval()
            measurement, features = self.bench.micro_benchmark_gpu(inference, inputs, model)
            batch_measures[INFERENCE] = measurement

            features_cache_id = f'{exec_step.feature_cache_prefix}-{INPUT}-{i}'
            labels_cache_id = f'{exec_step.feature_cache_prefix}-{LABEL}-{i}'
            self.caching_service.cache_persistent(features_cache_id, features)
            self.caching_service.cache_persistent(labels_cache_id, labels)

            self.logger.append_value(BATCH_MEASURES, batch_measures)

            start = time.perf_counter()
        exec_step.execution_logs = self.logger

    def execute_score_model_step(self, exex_step: ScoreModelStep):
        train_loader = self._get_proxy_data_loader(exex_step.train_feature_cache_prefixes)
        test_loader = self._get_proxy_data_loader(exex_step.test_feature_cache_prefixes)
        print(len(test_loader))
        print(len(train_loader))
        measurement, score = self.bench.benchmark_end_to_end(
            linear_proxy, train_loader, test_loader, exex_step.num_classes, device=torch.device(CUDA)
        )
        self.logger.log_value(CALC_PROXY_SCORE, measurement)
        exex_step.execution_result = {'loss': score[0], 'top-1-acc': score[1]}
        exex_step.execution_logs = self.logger

    def _get_proxy_data_loader(self, cache_prefixes):
        data = CacheServiceDataset(
            self.caching_service,
            [f'{pre}-{INPUT}' for pre in cache_prefixes],
            [f'{pre}-{LABEL}' for pre in cache_prefixes]
        )
        if check_deterministic_env_var_set():
            num_workers = 0
        else:
            num_workers = 2
        data_loader = DataLoader(data, batch_size=1, num_workers=num_workers)
        return data_loader
