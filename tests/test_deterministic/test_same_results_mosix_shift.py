import os
import unittest

from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.deterministic import DETERMINISTIC_EXECUTION, TRUE, check_deterministic_env_var_set, set_deterministic
from global_utils.global_constants import TEST
from global_utils.global_constants import TRAIN
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.approaches.mosix import find_best_model
from model_search.approaches.shift import get_data_ranges, prune_snapshots
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.shift_execution_engine import ShiftExecutionEngine
from model_search.execution.planning.baseline_planner import TEST
from model_search.execution.planning.mosix_planner import MosixPlannerConfig
from model_search.execution.planning.shift_planner import ShiftPlannerConfig, ShiftExecutionPlanner, \
    get_sorted_model_scores


def _execute_mosix():
    os.environ[DETERMINISTIC_EXECUTION] = TRUE

    if check_deterministic_env_var_set():
        num_workers = 0
        set_deterministic()
    else:
        num_workers = 12

    save_path = '/mount-fs/tmp-dir'
    model_snapshots, model_store = dummy_snap_and_mstore_four_models(save_path)

    # datasets
    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }
    train_data = CustomImageFolder(dataset_paths[TRAIN])

    planner_config = MosixPlannerConfig(num_workers, 128, DatasetClass.CUSTOM_IMAGE_FOLDER, dataset_paths)
    persistent_caching_path = '/mount-ssd/cache-dir'

    return find_best_model(model_snapshots, model_store, dataset_paths, len(train_data), planner_config, persistent_caching_path)


def _execute_shift():
    os.environ[DETERMINISTIC_EXECUTION] = TRUE

    if check_deterministic_env_var_set():
        num_workers = 0
        set_deterministic()
    else:
        num_workers = 12

    save_path = '/mount-fs/tmp-dir'
    _model_snapshots, _ = dummy_snap_and_mstore_four_models(save_path)

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

    ranking = None

    first_iteration = True
    for _range in ranges:
        plan = planner.generate_execution_plan(list(model_snapshots.values()), dataset_paths, _range, first_iteration)
        exec_engine.execute_plan(plan)
        prune_snapshots(model_snapshots, plan)
        first_iteration = False
        ranking = get_sorted_model_scores(plan.execution_steps)

    return ranking

class TestDeterministicOutput(unittest.TestCase):

    def test_mosix_shift(self):
        mosix_out = _execute_mosix()
        shift_out = _execute_shift()
        print(mosix_out)
        print(shift_out)
        self.assertEqual(mosix_out, shift_out)


