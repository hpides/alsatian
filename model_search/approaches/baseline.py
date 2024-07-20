import torch

from experiments.model_search.benchmark_level import BenchmarkLevel
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import EXEC_STEP_MEASUREMENTS, GEN_EXEC_PLAN, MODEL_RANKING
from global_utils.global_constants import TRAIN
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.approaches.shift import get_sorted_model_scores
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine
from model_search.execution.planning.baseline_planner import TEST, BaselineExecutionPlanner, PlannerConfig
from model_search.model_snapshots.base_snapshot import ModelSnapshot


def find_best_model(model_snapshots: [ModelSnapshot], planner_config, caching_path,
                    benchmark_level: BenchmarkLevel = None):
    measurements = {}
    ignore_micro_bench = ((benchmark_level == BenchmarkLevel.END_TO_END) or
                          (benchmark_level == BenchmarkLevel.SH_PHASES))
    benchmarker = Benchmarker(torch.device('cuda'), ignore_micro_bench=ignore_micro_bench)

    planner = BaselineExecutionPlanner(planner_config)

    cachingService = CachingService(caching_path)
    exec_engine = BaselineExecutionEngine(cachingService)

    measure, execution_plan = benchmarker.micro_benchmark_cpu(planner.generate_execution_plan, model_snapshots)
    measurements[GEN_EXEC_PLAN] = measure
    measurements[EXEC_STEP_MEASUREMENTS] = exec_engine.execute_plan(execution_plan, benchmark_level=benchmark_level)

    ranking = get_sorted_model_scores(execution_plan.execution_steps)
    measurements[f'{MODEL_RANKING}'] = ranking

    return measurements, ranking


if __name__ == '__main__':
    num_workers = 12

    save_path = '/mount-fs/tmp-dir'
    model_snapshots, model_store = dummy_snap_and_mstore_four_models(save_path)

    # datasets
    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }

    caching_path = '/mount-ssd/cache-dir'
    planner_config = PlannerConfig(12, 128, 100, DatasetClass.CUSTOM_IMAGE_FOLDER, dataset_paths)

    ranking = find_best_model(model_snapshots, planner_config, caching_path)

    print(ranking)
