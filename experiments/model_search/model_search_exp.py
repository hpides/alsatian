import os

from custom.data_loaders.custom_image_folder import CustomImageFolder
from experiments.model_search.experiment_args import ExpArgs
from experiments.snapshots.twenty_resnet_152 import twenty_resnet_152_snapshots
from global_utils.constants import TRAIN, TEST
from model_search.approaches import baseline, mosix, shift
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.planning.planner_config import PlannerConfig


def get_snapshots(snapshot_set_string, base_save_path):
    if snapshot_set_string == 'twenty_resnet_152':
        snapshot_save_path = os.path.join(base_save_path, snapshot_set_string)
        return twenty_resnet_152_snapshots(snapshot_save_path)


def run_model_search(exp_args: ExpArgs):
    # set some hardcoded values
    persistent_caching_path = exp_args.persistent_caching_path

    num_workers = exp_args.num_workers
    batch_size = exp_args.batch_size
    num_target_classes = exp_args.num_target_classes

    dataset_paths = {
        TRAIN: exp_args.train_data,
        TEST: exp_args.test_data
    }
    dataset_class = DatasetClass.CUSTOM_IMAGE_FOLDER
    train_data = CustomImageFolder(dataset_paths[TRAIN])
    len_train_data = len(train_data)

    model_snapshots, model_store = get_snapshots(exp_args.snapshot_set_string, exp_args.base_snapshot_save_path)

    planner_config = PlannerConfig(num_workers, batch_size, num_target_classes, dataset_class, dataset_paths)

    if exp_args.approach == 'mosix':
        mosix.find_best_model(model_snapshots, len_train_data, planner_config, persistent_caching_path, model_store)
    elif exp_args.approach == 'shift':
        shift.find_best_model(model_snapshots, len_train_data, planner_config, persistent_caching_path)
    elif exp_args.approach == 'baseline':
        baseline.find_best_model(model_snapshots, planner_config, persistent_caching_path)

    measurements = {}

    return measurements
