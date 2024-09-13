from abc import ABC, abstractmethod

import torch

from experiments.main_experiments.model_search.benchmark_level import BenchmarkLevel
from global_utils.benchmark_util import Benchmarker
from model_search.caching_service import CachingService
from model_search.execution.planning.execution_plan import ExecutionPlan, ExecutionStep
from model_search.execution_step_logger import ExecutionStepLogger


def _init_benchmarker(benchmark_level: BenchmarkLevel):
    if (benchmark_level is None or
            (benchmark_level == BenchmarkLevel.END_TO_END) or (benchmark_level == BenchmarkLevel.SH_PHASES)):
        return Benchmarker(torch.device('cuda'), ignore_end_to_end=True, ignore_micro_bench=True)
    elif benchmark_level == BenchmarkLevel.EXECUTION_STEPS:
        return Benchmarker(torch.device('cuda'), ignore_end_to_end=False, ignore_micro_bench=True)
    elif benchmark_level == BenchmarkLevel.STEPS_DETAILS:
        return Benchmarker(torch.device('cuda'), ignore_end_to_end=False, ignore_micro_bench=False)


class ExecutionEngine(ABC):

    def __init__(self, cachingService: CachingService):
        self.caching_service: CachingService = cachingService
        self.logger = ExecutionStepLogger()

    def execute_plan(self, execution_plan: ExecutionPlan, benchmark_level=None):
        measurements = {}
        self.bench = _init_benchmarker(benchmark_level)

        exex_step_number = 1
        for exex_step in execution_plan.execution_steps:
            print("EXEC STEP NUMBER:", exex_step_number, "===", type(exex_step).__name__)
            measure, step_detailed_measure = self.bench.benchmark_end_to_end(self.execute_step, exex_step)
            measurements[f'{exex_step_number}-{type(exex_step).__name__}'] = measure
            measurements[f'{exex_step_number}-{type(exex_step).__name__}-details'] = step_detailed_measure
            exex_step_number += 1

        return measurements


@abstractmethod
def execute_step(self, exex_step: ExecutionStep):
    pass
