import torch.nn

from model_search.caching_service import TensorCachingService
from model_search.execution.engine.abstract_execution_engine import ExecutionEngine
from model_search.execution.planning.execution_plan import ExecutionStep


class MosixExecutionEngine(ExecutionEngine):

    def __init__(self, caching_service: TensorCachingService):
        self.caching_service: TensorCachingService = caching_service
        self.prev_model: torch.nn.Module = None

    def execute_step(self, exex_step: ExecutionStep):
        # TODO look into existing code to fill out
        # 1) init model, by loading state into previous model use prev_model
        # 2) split model
        # 3) execute single models + cache/reuse outputs
        # 4) clean up -> drop intermediates not needed anymore

        pass
