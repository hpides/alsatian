import os
import pathlib

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder, create_sub_dataset
from experiments.model_search.experiment_args import ExpArgs
from experiments.prevent_caching.watch_utils import clear_caches_and_check_io_limit
from experiments.snapshots.generate_sets.generate_set import get_architecture_models
from experiments.snapshots.generate_sets.twenty_resnet_152 import twenty_resnet_152_snapshots
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import TRAIN, TEST, END_TO_END, DETAILED_TIMES
from global_utils.model_names import VISION_MODEL_CHOICES
from model_search.approaches import baseline, mosix, shift
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.planning.planner_config import PlannerConfig


def get_snapshots(snapshot_set_string, num_models, distribution, base_save_path):
    if snapshot_set_string == 'twenty_resnet_152':
        snapshot_save_path = os.path.join(base_save_path, snapshot_set_string)
        return twenty_resnet_152_snapshots(snapshot_save_path)
    elif snapshot_set_string in VISION_MODEL_CHOICES:
        return get_architecture_models(base_save_path, distribution, num_models, [snapshot_set_string])
    else:
        # TODO sets are already implemented, just need to add the right strings and a parameter for the number of models
        raise NotImplementedError


def _prepare_datasets(exp_args):
    train_data_path = exp_args.train_data
    if exp_args.num_train_items > 0:
        train_data_path = create_sub_dataset(exp_args.train_data, exp_args.num_train_items)

    test_data_path = exp_args.test_data
    if exp_args.num_test_items > 0:
        test_data_path = create_sub_dataset(exp_args.test_data, exp_args.num_test_items)

    dataset_paths = {
        TRAIN: train_data_path,
        TEST: test_data_path
    }
    return dataset_paths


def run_model_search(exp_args: ExpArgs):
    # set some hardcoded values
    persistent_caching_path = exp_args.persistent_caching_path
    caching_loc = exp_args.default_cache_location
    num_workers = exp_args.num_workers
    batch_size = exp_args.batch_size
    num_target_classes = exp_args.num_target_classes
    cache_size = exp_args.cache_size

    # prepare datasets
    dataset_paths = _prepare_datasets(exp_args)
    dataset_class = DatasetClass.CUSTOM_IMAGE_FOLDER
    train_data = CustomImageFolder(dataset_paths[TRAIN])
    len_train_data = len(train_data)
    test_data = CustomImageFolder(dataset_paths[TEST])
    len_test_data = len(test_data)

    model_snapshots, model_store = get_snapshots(exp_args.snapshot_set_string, exp_args.num_models,
                                                 exp_args.distribution, exp_args.base_snapshot_save_path)

    layer_output_info = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                     '../model_resource_info/outputs/layer_output_infos.json')
    model_store.add_output_sizes_to_rich_snapshots(layer_output_info)

    # after generating the snapshots make sure they are not in the caches
    clear_caches_and_check_io_limit()

    planner_config = PlannerConfig(num_workers, batch_size, num_target_classes, dataset_class, dataset_paths,
                                   caching_loc, cache_size)

    benchmark_level = exp_args.benchmark_level
    benchmarker = Benchmarker(torch.device('cuda'))

    if exp_args.approach == 'mosix':
        args = [model_snapshots, len_train_data, len_test_data, planner_config, persistent_caching_path, model_store,
                benchmark_level]
        measure, (sub_measurements, result) = benchmarker.benchmark_end_to_end(mosix.find_best_model, *args)
    elif exp_args.approach == 'shift':
        args = [model_snapshots, len_train_data, planner_config, persistent_caching_path, benchmark_level]
        measure, (sub_measurements, result) = benchmarker.benchmark_end_to_end(shift.find_best_model, *args)
    elif exp_args.approach == 'baseline':
        args = [model_snapshots, planner_config, persistent_caching_path, benchmark_level]
        measure, (sub_measurements, result) = benchmarker.benchmark_end_to_end(baseline.find_best_model, *args)
    else:
        raise NotImplementedError

    measurements = {END_TO_END: measure, DETAILED_TIMES: sub_measurements}

    return measurements, result
