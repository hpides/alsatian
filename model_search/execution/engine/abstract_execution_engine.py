from abc import ABC, abstractmethod

import torch

from experiments.model_search.benchmark_level import BenchmarkLevel
from global_utils.benchmark_util import Benchmarker, CUDA
from model_search.caching_service import CachingService
from model_search.execution.planning.execution_plan import ExecutionPlan, ExecutionStep
from model_search.executionsteplogger import ExecutionStepLogger


def _init_benchmarker(benchmark_level: BenchmarkLevel):
    ignore_micro_bench = \
        (benchmark_level == BenchmarkLevel.END_TO_END) or (benchmark_level == BenchmarkLevel.SH_PHASES)
    return Benchmarker(torch.device('cuda'), ignore_end_to_end=ignore_micro_bench)


class ExecutionEngine(ABC):

    def __init__(self, cachingService: CachingService):
        self.caching_service: CachingService = cachingService
        self.logger = ExecutionStepLogger()
        # TODO change this for now ignore all details that happen in the execution engines
        self.bench = Benchmarker(torch.device(CUDA), ignore_micro_bench=True, ignore_end_to_end=True)

    def execute_plan(self, execution_plan: ExecutionPlan, benchmark_level=None):
        benchmarker = _init_benchmarker(benchmark_level)
        measurements = {}

        exex_step_number = 0
        for exex_step in execution_plan.execution_steps:
            print("EXEC STEP NUMBER:", exex_step_number, "===", type(exex_step).__name__)
            measure, _ = benchmarker.benchmark_end_to_end(self.execute_step, exex_step)
            exex_step_number += 1
            measurements[f'{exex_step_number}-{type(exex_step).__name__}'] = measure

        return measurements


    @abstractmethod
    def execute_step(self, exex_step: ExecutionStep, benchmark_level=None):
        pass
