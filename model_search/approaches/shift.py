import math
import os

from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.constants import SCORE
from global_utils.deterministic import DETERMINISTIC_EXECUTION, check_deterministic_env_var_set, set_deterministic
from global_utils.global_constants import TRAIN
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.shift_execution_engine import ShiftExecutionEngine
from model_search.execution.planning.baseline_planner import TEST
from model_search.execution.planning.execution_plan import ScoreModelStep
from model_search.execution.planning.planner_config import AdvancedPlannerConfig
from model_search.execution.planning.shift_planner import ShiftExecutionPlanner
from model_search.model_snapshots.base_snapshot import ModelSnapshot


def get_sorted_model_scores(execution_steps):
    scores = []
    for step in execution_steps:
        if isinstance(step, ScoreModelStep):
            for snapshot_id in step.scored_models:
                scores.append([step.execution_result[SCORE], snapshot_id])

    return sorted(scores)


def divide_snapshots(execution_steps):
    ranking = get_sorted_model_scores(execution_steps)
    print(ranking)
    snapshot_ids = [s[1] for s in ranking]
    cut = len(ranking) // 2
    return snapshot_ids[:cut], snapshot_ids[cut:]


def get_data_ranges(search_space_len, train_data_len) -> [int]:
    ranges = []
    iterations = math.ceil(math.log(search_space_len, 2))
    items_to_process = math.ceil(train_data_len / 2 ** iterations)
    items_seen = 0
    prev_end = 0

    for i in range(iterations):
        new_end = items_to_process + items_seen
        ranges.append([prev_end, new_end])
        prev_end = new_end
        items_seen += items_to_process
        items_to_process = items_to_process * 2

    if ranges[-1][1] < train_data_len:
        # make sure the last range covers all data
        ranges[-1][1] = train_data_len
    else:
        message = (f"train data to small for search space: for a search space with length {search_space_len},"
                   f" we need a dataset with at least {ranges[-1][1]} items")
        raise Exception(message)

    return ranges


def prune_snapshots(model_snapshots, keep_snapshot_ids):
    return [snap for snap in model_snapshots if snap.id in keep_snapshot_ids]


def find_best_model(model_snapshots: [ModelSnapshot], train_data_length, planner_config, caching_path):
    planner = ShiftExecutionPlanner(planner_config)

    cachingService = CachingService(caching_path)
    exec_engine = ShiftExecutionEngine(cachingService)

    data_ranges = get_data_ranges(len(model_snapshots), train_data_length)

    ranking = None

    first_iteration = True
    for _range in data_ranges:
        execution_plan = planner.generate_execution_plan(model_snapshots, _range, first_iteration)
        exec_engine.execute_plan(execution_plan)
        _, keep_snapshot_ids = divide_snapshots(execution_plan.execution_steps)
        model_snapshots = prune_snapshots(model_snapshots, keep_snapshot_ids)
        first_iteration = False
        ranking = get_sorted_model_scores(execution_plan.execution_steps)

    return ranking


if __name__ == '__main__':
    # basically for this we do not need deterministic execution, leave the flag here if we want to debug
    os.environ[DETERMINISTIC_EXECUTION] = ""
    # os.environ[DETERMINISTIC_EXECUTION] = TRUE

    if check_deterministic_env_var_set():
        num_workers = 0
        set_deterministic()
    else:
        num_workers = 12

    save_path = '/mount-fs/tmp-dir'
    _model_snapshots, model_store = dummy_snap_and_mstore_four_models(save_path)

    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }
    train_data_length = len(CustomImageFolder(dataset_paths[TRAIN]))

    planner_config = AdvancedPlannerConfig(num_workers, 128, 100, DatasetClass.CUSTOM_IMAGE_FOLDER, dataset_paths)
    caching_path = '/mount-ssd/cache-dir'

    find_best_model(_model_snapshots, train_data_length, planner_config, caching_path)
