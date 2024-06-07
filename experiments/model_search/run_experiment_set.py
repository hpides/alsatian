import argparse
import configparser
import os
import sys
import time
import traceback

import torch

from experiments.model_search.experiment_args import ExpArgs, _str_to_distribution, _str_to_cache_location, \
    _str_to_benchmark_level
from experiments.model_search.model_search_exp import run_model_search
from experiments.prevent_caching.watch_utils import LIMIT_IO
from global_utils.deterministic import TRUE
from global_utils.model_names import RESNET_152, RESNET_18, MOBILE_V2, VIT_L_32, EFF_NET_V2_L, RESNET_50, RESNET_101, \
    VIT_B_16, EFF_NET_V2_S, RESNET_34
from global_utils.write_results import write_measurements_and_args_to_json_file

BENCHMARK_LEVELS = "benchmark_levels"
NUMS_MODELS = "nums_models"
SNAPSHOT_SET_STRINGS = "snapshot_set_strings"
DEFAULT_CACHE_LOCATIONS = "default_cache_locations"
APPROACHES = "approaches"
DISTRIBUTIONS = "distributions"


def run_exp(exp_args):
    return run_model_search(exp_args)


def run_exp_set(base_exp_args, eval_space, base_file_id):
    for distribution in eval_space[DISTRIBUTIONS]:
        base_exp_args.distribution = _str_to_distribution(distribution)
        for approach in eval_space[APPROACHES]:
            base_exp_args.approach = approach
            for cache_location in eval_space[DEFAULT_CACHE_LOCATIONS]:
                base_exp_args.default_cache_location = _str_to_cache_location(cache_location)
                for snapshot_set in eval_space[SNAPSHOT_SET_STRINGS]:
                    base_exp_args.snapshot_set_string = snapshot_set
                    for num_models in eval_space[NUMS_MODELS]:
                        base_exp_args.num_models = num_models
                        for bench_level in eval_space[BENCHMARK_LEVELS]:
                            base_exp_args.benchmark_level = _str_to_benchmark_level(bench_level)

                            file_id = (f"{base_file_id}-distribution-{distribution}-approach-{approach}"
                                       f"-cache-{cache_location}-snapshot-{snapshot_set}"
                                       f"-models-{num_models}-level-{bench_level}")

                            print("RUN:", file_id)

                            try:
                                run_experiment(base_exp_args, file_id)
                            except AssertionError as e:
                                print("RUN FAILED:", file_id)
                                _, _, tb = sys.exc_info()
                                traceback.print_tb(tb)  # Fixed format
                                tb_info = traceback.extract_tb(tb)
                                filename, line, func, text = tb_info[-1]

                                print('An error occurred on line {} in statement {}'.format(line, text))

                            # sleep and clean up
                            time.sleep(2)
                            torch.cuda.empty_cache()
                            time.sleep(2)


def run_experiment(exp_args, file_id):
    print(f'run experiment:{exp_args}')

    if exp_args.limit_fs_io:
        os.environ[LIMIT_IO] = TRUE

    # code to start experiment here
    measurements, result = run_exp(exp_args)

    write_measurements_and_args_to_json_file(
        measurements=measurements,
        args=exp_args.get_dict(),
        dir_path=exp_args.result_dir,
        file_id=file_id
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config.ini', help='Configuration file path')
    parser.add_argument('--base_config_section', default='des-gpu-imagenette-base-1000')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.base_config_section)

    # Call the main function with parsed arguments
    # run_experiment_section(exp_args, args.config_section)
    eval_space = {
        DISTRIBUTIONS: ["LAST_ONE_LAYER"],
        APPROACHES: ["baseline", "shift"],
        DEFAULT_CACHE_LOCATIONS: ["GPU"],
        SNAPSHOT_SET_STRINGS: [RESNET_18, RESNET_152, VIT_L_32, MOBILE_V2, RESNET_50, RESNET_101,
                               RESNET_34, EFF_NET_V2_S, EFF_NET_V2_L, VIT_B_16],
        NUMS_MODELS: [35],
        BENCHMARK_LEVELS: ["STEPS_DETAILS", "EXECUTION_STEPS"]
    }
    run_exp_set(exp_args, eval_space, base_file_id=args.base_config_section)
