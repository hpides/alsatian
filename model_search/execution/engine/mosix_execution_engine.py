import time

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, BATCH_MEASURES
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine, load_data_to_device, \
    inference
from model_search.execution.planning.execution_plan import ExecutionStep, ScoreModelStep
from model_search.execution.planning.mosix_planner import MosixExtractFeaturesStep
from model_search.executionsteplogger import ExecutionStepLogger
from model_search.model_snapshot import RichModelSnapshot


class MosixExecutionEngine(BaselineExecutionEngine):

    def execute_step(self, exec_step: ExecutionStep):
        # reset logger for every step
        self.logger = ExecutionStepLogger()
        if isinstance(exec_step, MosixExtractFeaturesStep):
            self.execute_mosix_extract_features_step(exec_step)
        elif isinstance(exec_step, ScoreModelStep):
            self.execute_score_model_step(exec_step)
        else:
            raise TypeError

    def execute_mosix_extract_features_step(self, exec_step):
        # init data loader
        if exec_step.inp_data.data_set_class == DatasetClass.CUSTOM_IMAGE_FOLDER:
            data = CustomImageFolder(exec_step.inp_data.dataset_path, exec_step.inp_data.transform)
        else:
            raise NotImplementedError

        if exec_step.data_range is not None:
            data.set_subrange(exec_step.data_range[0], exec_step.data_range[1])

        self._extract_features(data, exec_step)

    def _extract_features(self, data, exec_step):

        # initialize model
        model = self.init_model(exec_step.model_snapshot)

        # init data loader
        # TODO dynamically adjust batch size and nuber of workers
        # TODO adjust to cached data loader if feasible
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=exec_step.inp_data.batch_size, shuffle=False,
            num_workers=exec_step.inp_data.num_workers)

        # extract features
        start = time.perf_counter()
        for i, (inputs, labels) in enumerate(data_loader):
            batch_measures = {}
            batch_measures[LOAD_DATA] = time.perf_counter() - start

            # TODO might do the data loading by the caching manager, lets see ...
            measurement, inputs = self.bench.micro_benchmark_cpu(load_data_to_device, inputs)
            batch_measures[DATA_TO_DEVICE] = measurement

            model.eval()
            # TODO this we need to adjust to split into multiple models to cache at all points we want to cahce
            measurement, features = self.bench.micro_benchmark_gpu(inference, inputs, model)
            batch_measures[INFERENCE] = measurement
            features_cache_id = f'{exec_step.feature_cache_prefix}-{INPUT}-{i}'
            labels_cache_id = f'{exec_step.feature_cache_prefix}-{LABEL}-{i}'
            # TODO also cache according to the planned cache, not just always persistent caching
            self.caching_service.cache_persistent(features_cache_id, features)
            self.caching_service.cache_persistent(labels_cache_id, labels)

            self.logger.append_value(BATCH_MEASURES, batch_measures)

            start = time.perf_counter()
        exec_step.execution_logs = self.logger

    def init_model(self, snapshot: RichModelSnapshot):
        # TODO
        pass
