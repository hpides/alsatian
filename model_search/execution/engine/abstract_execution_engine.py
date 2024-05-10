from abc import ABC, abstractmethod

import torch

from global_utils.benchmark_util import Benchmarker, CUDA
from model_search.caching_service import CachingService
from model_search.execution.planning.execution_plan import ExecutionPlan, ExecutionStep
from model_search.executionsteplogger import ExecutionStepLogger


class ExecutionEngine(ABC):

    def __init__(self, cachingService: CachingService):
        self.caching_service: CachingService = cachingService
        self.logger = ExecutionStepLogger()
        self.bench = Benchmarker(torch.device(CUDA))

    def execute_plan(self, execution_plan: ExecutionPlan):
        exex_step_number = 0
        for exex_step in execution_plan.execution_steps:
            print("EXEC STEP NUMBER:", exex_step_number)
            self.execute_step(exex_step)
            exex_step_number += 1

    @abstractmethod
    def execute_step(self, exex_step: ExecutionStep):
        pass
