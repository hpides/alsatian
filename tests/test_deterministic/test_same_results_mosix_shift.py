import os
import unittest

from custom.data_loaders.custom_image_folder import CustomImageFolder
from global_utils.deterministic import DETERMINISTIC_EXECUTION, TRUE, check_deterministic_env_var_set, set_deterministic
from global_utils.global_constants import TEST
from global_utils.global_constants import TRAIN
from model_search.approaches import shift, mosix
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.planning.baseline_planner import TEST
from model_search.execution.planning.planner_config import AdvancedPlannerConfig


def get_search_model_inputs():
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
    planner_config = AdvancedPlannerConfig(num_workers, 128, 100, DatasetClass.CUSTOM_IMAGE_FOLDER, dataset_paths)
    persistent_caching_path = '/mount-ssd/cache-dir'
    return dataset_paths, model_snapshots, model_store, persistent_caching_path, planner_config, train_data


def _execute_mosix():
    dataset_paths, model_snapshots, model_store, persistent_caching_path, planner_config, train_data = get_search_model_inputs()

    return mosix.find_best_model(model_snapshots, model_store, len(train_data), planner_config, persistent_caching_path)


def _execute_shift():
    dataset_paths, model_snapshots, model_store, persistent_caching_path, planner_config, train_data = get_search_model_inputs()

    return shift.find_best_model(model_snapshots, model_store, len(train_data), planner_config, persistent_caching_path)


class TestDeterministicOutput(unittest.TestCase):

    def test_mosix_shift(self):
        mosix_out = _execute_mosix()
        shift_out = _execute_shift()
        print(shift_out)
        print(mosix_out)
        self.assertEqual(mosix_out, shift_out)
