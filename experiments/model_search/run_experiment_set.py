import argparse
import configparser
import time

import torch

from experiments.model_search.experiment_args import ExpArgs, _str_to_distribution, _str_to_cache_location, \
    _str_to_benchmark_level
from experiments.model_search.model_search_exp import run_model_search
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

                            run_experiment(base_exp_args, file_id)

                            # sleep and clean up
                            time.sleep(2)
                            torch.cuda.empty_cache()
                            time.sleep(2)


def run_experiment(exp_args, file_id):
    print(f'run experiment:{exp_args}')

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
    parser.add_argument('--base_config_section', default='des-gpu-imagenette-base')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.base_config_section)

    # Call the main function with parsed arguments
    # run_experiment_section(exp_args, args.config_section)
    eval_space = {
        DISTRIBUTIONS: ["TOP_LAYERS"],
        APPROACHES: ["baseline", "shift", "mosix"],
        DEFAULT_CACHE_LOCATIONS: ["GPU"],
        SNAPSHOT_SET_STRINGS: ["resnet18"],
        NUMS_MODELS: [4],
        BENCHMARK_LEVELS: ["END_TO_END", "EXECUTION_STEPS"]
    }
    run_exp_set(exp_args, eval_space, base_file_id=args.base_config_section)
