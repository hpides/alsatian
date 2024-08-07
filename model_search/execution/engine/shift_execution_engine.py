from custom.data_loaders.custom_image_folder import CustomImageFolder
from data.imdb.reduced_imdb import get_imbdb_bert_base_uncased_datasets
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine
from model_search.execution.planning.execution_plan import ExecutionStep, BaselineExtractFeaturesStep, ScoreModelStep
from model_search.execution.planning.shift_planner import ShiftExtractFeaturesStep
from model_search.executionsteplogger import ExecutionStepLogger


class ShiftExecutionEngine(BaselineExecutionEngine):

    def execute_step(self, exec_step: ExecutionStep):
        # reset logger for every step
        self.logger = ExecutionStepLogger()

        if isinstance(exec_step, ShiftExtractFeaturesStep):
            self.execute_shift_extract_features_step(exec_step)
        elif isinstance(exec_step, BaselineExtractFeaturesStep):
            self.execute_baseline_extract_features_step(exec_step)
        elif isinstance(exec_step, ScoreModelStep):
            self.execute_score_model_step(exec_step)
        else:
            raise TypeError

        return self.logger.log_dict

    def execute_shift_extract_features_step(self, exec_step: ShiftExtractFeaturesStep):

        # init data loader
        if exec_step.inp_data.data_set_class == DatasetClass.CUSTOM_IMAGE_FOLDER:
            data = CustomImageFolder(exec_step.inp_data.dataset_path, exec_step.inp_data.transform)
            data.set_subrange(exec_step.data_range[0], exec_step.data_range[1])
        elif exec_step.inp_data.data_set_class == DatasetClass.IMDB:
            data = get_imbdb_bert_base_uncased_datasets(exec_step.inp_data.dataset_path)
            data.set_subrange(exec_step.data_range[0], exec_step.data_range[1])
        else:
            raise NotImplementedError

        self._extract_features(data, exec_step)
