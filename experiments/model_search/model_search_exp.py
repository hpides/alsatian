import os

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from experiments.model_search.benchmark_level import BenchmarkLevel
from experiments.model_search.experiment_args import ExpArgs
from experiments.snapshots.generate_sets.generate_set import generate_snapshot_set, get_architecture_models
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
        return get_architecture_models(base_save_path,distribution,num_models,[snapshot_set_string])
    else:
        # TODO sets are already implemented, just need to add the right strings and a parameter for the number of models
        raise NotImplementedError


def run_model_search(exp_args: ExpArgs):
    # set some hardcoded values
    persistent_caching_path = exp_args.persistent_caching_path
    caching_loc = exp_args.default_cache_location
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

    model_snapshots, model_store = get_snapshots(exp_args.snapshot_set_string, exp_args.num_models,
                                                 exp_args.distribution, exp_args.base_snapshot_save_path)

    planner_config = PlannerConfig(num_workers, batch_size, num_target_classes, dataset_class, dataset_paths, caching_loc)

    benchmark_level = exp_args.benchmark_level
    benchmarker = Benchmarker(torch.device('cuda'))

    if exp_args.approach == 'mosix':
        args = [model_snapshots, len_train_data, planner_config, persistent_caching_path, model_store, benchmark_level]
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
