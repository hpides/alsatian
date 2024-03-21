from abc import abstractmethod

from model_search.execution.engine.abstract_execution_engine import ExecutionEngine
from model_search.execution.planning.execution_plan import ExecutionStep, BaselineExtractFeaturesStep




class BaselineExecutionEngine(ExecutionEngine):

    def execute_step(self, exex_step: ExecutionStep):
        if isinstance(exex_step, BaselineExtractFeaturesStep):
            self.execute_baseline_extract_features_step(exex_step)
        else:
            raise NotImplementedError

    def execute_baseline_extract_features_step(self, exex_step):
        # TODO -> look into already existing code
        pass
