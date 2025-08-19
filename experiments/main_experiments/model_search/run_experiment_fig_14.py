import argparse
import configparser
import os
import sys
import time
import traceback

import torch

from experiments.main_experiments.model_search.experiment_args import ExpArgs, _str_to_cache_location, \
    _str_to_benchmark_level
from experiments.main_experiments.model_search.model_search_exp import run_model_search
from experiments.main_experiments.prevent_caching.watch_utils import LIMIT_IO
from experiments.main_experiments.snapshots.hugging_face.init_hf_models import *
from global_utils.deterministic import TRUE
from global_utils.write_results import write_measurements_and_args_to_json_file

DATA_ITEMS = "data_itmes"

BENCHMARK_LEVELS = "benchmark_levels"
NUMS_MODELS = "nums_models"
SNAPSHOT_SET_STRINGS = "snapshot_set_strings"
DEFAULT_CACHE_LOCATIONS = "default_cache_locations"
APPROACHES = "approaches"
DISTRIBUTIONS = "distributions"


def identify_missing_experiments(base_exp_args, eval_space, base_file_id, num_iterations, result_directory):
    # get a list of all files in the result_directory
    result_files = [f for f in os.listdir(result_directory) if os.path.isfile(os.path.join(result_directory, f))]
    # split result files by '#' and only keep the second part
    result_files = ["#".join(f.split('#')[2:]).replace(".json", "") for f in result_files]

    found = {}
    for file in result_files:
        if file not in found:
            found[file] = 0
        found[file] += 1

    # generate expected file dict
    expected = expected_experiment_files(base_exp_args, eval_space, base_file_id, num_iterations)

    print("found")
    print(found)
    print("expected")
    print(expected)

    # identify missing experiments
    diff_experiments = {}
    for key in list(expected.keys()):
        if key not in found:
            diff_experiments[key] = expected[key]
        else:
            diff = expected[key] - found[key]
            if diff > 0:
                diff_experiments[key] = diff

    print("diff_experiments")
    print(diff_experiments)
    return diff_experiments


def expected_experiment_files(base_exp_args, eval_space, base_file_id, num_iterations):
    result = {}
    for train_items, test_items in eval_space[DATA_ITEMS]:
        base_exp_args.num_train_items = train_items
        base_exp_args.num_test_items = test_items
        for snapshot_set in eval_space[SNAPSHOT_SET_STRINGS]:
            base_exp_args.snapshot_set_string = snapshot_set
            for approach in eval_space[APPROACHES]:
                base_exp_args.approach = approach
                for cache_location in eval_space[DEFAULT_CACHE_LOCATIONS]:
                    base_exp_args.default_cache_location = _str_to_cache_location(cache_location)
                    for num_models in eval_space[NUMS_MODELS]:
                        base_exp_args.num_models = num_models
                        for bench_level in eval_space[BENCHMARK_LEVELS]:
                            base_exp_args.benchmark_level = _str_to_benchmark_level(bench_level)

                            if approach == "baseline" or approach == "shift":
                                base_exp_args.load_full = True
                            else:
                                base_exp_args.load_full = False

                            file_id = (f"approach#{approach}"
                                       f"#cache#{cache_location}#snapshot#{snapshot_set.replace('/', '-')}"
                                       f"#models#{num_models}#items#{train_items + test_items}#level#{bench_level}")

                            result[file_id] = num_iterations

    return result


def run_exp(exp_args):
    return run_model_search(exp_args)


def run_exp_set(base_exp_args, eval_space, base_file_id):
    for train_items, test_items in eval_space[DATA_ITEMS]:
        base_exp_args.num_train_items = train_items
        base_exp_args.num_test_items = test_items
        for snapshot_set in eval_space[SNAPSHOT_SET_STRINGS]:
            base_exp_args.snapshot_set_string = snapshot_set
            for approach in eval_space[APPROACHES]:
                base_exp_args.approach = approach
                for cache_location in eval_space[DEFAULT_CACHE_LOCATIONS]:
                    base_exp_args.default_cache_location = _str_to_cache_location(cache_location)
                    for num_models in eval_space[NUMS_MODELS]:
                        base_exp_args.num_models = num_models
                        for bench_level in eval_space[BENCHMARK_LEVELS]:
                            base_exp_args.benchmark_level = _str_to_benchmark_level(bench_level)

                            if "," in snapshot_set:
                                # prevent that the file-name is too long
                                file_id = (f"{base_file_id}#approach#{approach}"
                                           f"#cache#{cache_location}#snapshot#combined"
                                           f"#models#{num_models}#items#{train_items + test_items}#level#{bench_level}")
                            else:
                                file_id = (f"{base_file_id}#approach#{approach}"
                                           f"#cache#{cache_location}#snapshot#{snapshot_set.replace('/', '-')}"
                                           f"#models#{num_models}#items#{train_items + test_items}#level#{bench_level}")

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
    parser.add_argument('--config_file')
    parser.add_argument('--base_config_section')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.base_config_section)

    for snapshot_set_string in [FACEBOOK_DETR_RESNET_101, FACEBOOK_DINOV2_LARGE, MICROSOFT_RESNET_152,
                                GOOGLE_VIT_BASE_PATCH16_224_IN21K, FACEBOOK_DETR_RESNET_50]:
        for data_items in [(1600, 400), (6400, 1600)]:
            for approach in ["shift", "baseline", "mosix"]:

                eval_space = {
                    APPROACHES: [approach],
                    DEFAULT_CACHE_LOCATIONS: ["CPU"],
                    SNAPSHOT_SET_STRINGS: [snapshot_set_string],
                    NUMS_MODELS: [exp_args.num_models],
                    BENCHMARK_LEVELS: ["EXECUTION_STEPS"],
                    DATA_ITEMS: [data_items]
                }

                num_runs = 1
                missing_exps = identify_missing_experiments(exp_args, eval_space, args.base_config_section, num_runs,
                                                            exp_args.result_dir)

                while len(missing_exps) > 0:
                    run_exp_set(exp_args, eval_space, base_file_id=args.base_config_section)

                    missing_exps = identify_missing_experiments(exp_args, eval_space, args.base_config_section,
                                                                num_runs,
                                                                exp_args.result_dir)
