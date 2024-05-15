from global_utils.global_constants import TRAIN
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.approaches.shift import get_sorted_model_scores
from model_search.caching_service import CachingService
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine
from model_search.execution.planning.baseline_planner import TEST, BaselineExecutionPlanner, BaselinePlannerConfig
from model_search.model_snapshots.base_snapshot import ModelSnapshot


def find_best_model(model_snapshots: [ModelSnapshot], planner_config, caching_path):
    planner = BaselineExecutionPlanner(planner_config)

    cachingService = CachingService(caching_path)
    exec_engine = BaselineExecutionEngine(cachingService)

    execution_plan = planner.generate_execution_plan(model_snapshots, dataset_paths)
    exec_engine.execute_plan(execution_plan)

    ranking = get_sorted_model_scores(execution_plan.execution_steps)

    return ranking


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
    planner_config = BaselinePlannerConfig(12, 128, 100)

    ranking = find_best_model(model_snapshots, planner_config, caching_path)

    print(ranking)
