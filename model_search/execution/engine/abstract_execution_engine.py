from abc import ABC, abstractmethod

import torch

from global_utils.benchmark_util import Benchmarker, CUDA
from model_search.caching_service import TensorCachingService
from model_search.execution.planning.execution_plan import ExecutionPlan, ExecutionStep
from model_search.executionsteplogger import ExecutionStepLogger


class ExecutionEngine(ABC):

    def __init__(self, cachingService: TensorCachingService):
        self.cachingService: TensorCachingService = cachingService
        self.logger = ExecutionStepLogger()
        self.bench = Benchmarker(torch.device(CUDA))

    def execute_plan(self, execution_plan: ExecutionPlan):
        for exex_step in execution_plan.execution_steps:
            self.execute_step(exex_step)

    @abstractmethod
    def execute_step(self, exex_step: ExecutionStep):
        pass
