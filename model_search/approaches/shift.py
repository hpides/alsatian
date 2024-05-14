import os
import math
import random

import numpy as np
import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.global_constants import TRAIN
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore
from model_search.caching_service import CachingService
from model_search.execution.engine.shift_execution_engine import ShiftExecutionEngine
from model_search.execution.planning.baseline_planner import TEST
from model_search.execution.planning.shift_planner import ShiftPlannerConfig, ShiftExecutionPlanner, \
    get_sorted_model_scores

CULABS_CONFIG = 'CUBLAS_WORKSPACE_CONFIG'


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


def prune_snapshots(model_snapshots, plan):
    # print(model_snapshots)
    ranking = get_sorted_model_scores(plan.execution_steps)
    print(ranking)
    for _, s_id in ranking[:len(ranking) // 2]:
        model_snapshots.pop(s_id)
    # print(model_snapshots)


if __name__ == '__main__':
    deterministic = True

    if deterministic:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        num_workers = 0
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        num_workers = 12
        assert CULABS_CONFIG not in os.environ


    save_path = '/mount-fs/tmp-dir'
    _model_snapshots, _ = dummy_snap_and_mstore(save_path)

    model_snapshots = {}
    for snap in _model_snapshots:
        model_snapshots[snap.id] = snap

    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }

    caching_path = '/mount-ssd/cache-dir'
    cachingService = CachingService(caching_path)
    planner_config = ShiftPlannerConfig(num_workers, 128)
    planner = ShiftExecutionPlanner(planner_config)
    exec_engine = ShiftExecutionEngine(cachingService)

    train_data = CustomImageFolder(dataset_paths[TRAIN])
    ranges = get_data_ranges(len(list(model_snapshots.values())), len(train_data))

    first_iteration = True
    for _range in ranges:
        plan = planner.generate_execution_plan(list(model_snapshots.values()), dataset_paths, _range, first_iteration)
        exec_engine.execute_plan(plan)
        prune_snapshots(model_snapshots, plan)
        first_iteration = False

    print('done')
