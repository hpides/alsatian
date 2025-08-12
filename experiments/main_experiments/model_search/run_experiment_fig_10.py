import argparse
import configparser
import os
import sys
import time
import traceback

import torch

from experiments.main_experiments.model_search.experiment_args import ExpArgs, _str_to_distribution, \
    _str_to_cache_location, \
    _str_to_benchmark_level
from experiments.main_experiments.model_search.model_search_exp_synthetic import run_model_search
from experiments.main_experiments.prevent_caching.watch_utils import LIMIT_IO
from experiments.main_experiments.snapshots.synthetic.generate import TWENTY_FIVE_PERCENT, FIFTY_PERCENT, TOP_LAYERS
from global_utils.deterministic import TRUE
from global_utils.model_names import RESNET_18
from global_utils.write_results import write_measurements_and_args_to_json_file

DATA_ITEMS = "data_itmes"

BENCHMARK_LEVELS = "benchmark_levels"
NUMS_MODELS = "nums_models"
SNAPSHOT_SET_STRINGS = "snapshot_set_strings"
DEFAULT_CACHE_LOCATIONS = "default_cache_locations"
APPROACHES = "approaches"
DISTRIBUTIONS = "distributions"


def run_exp(exp_args):
    return run_model_search(exp_args)


def identify_missing_experiments(base_exp_args, eval_space, base_file_id, num_iterations, result_directory):
    # get a list of all files in the result_directory
    result_files = [f for f in os.listdir(result_directory) if os.path.isfile(os.path.join(result_directory, f))]
    # split result files by '#' and only keep the second part
    result_files = [f.split('#')[-1].replace(".json", "") for f in result_files]

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

                                if approach == "baseline" or approach == "shift":
                                    base_exp_args.load_full = True
                                else:
                                    base_exp_args.load_full = False

                                file_id = (f"{base_file_id}-distribution-{distribution}-approach-{approach}"
                                           f"-cache-{cache_location}-snapshot-{snapshot_set}"
                                           f"-models-{num_models}-items-{train_items + test_items}-level-{bench_level}")

                                result[file_id] = num_iterations

        return result


def run_exp_set(base_exp_args, eval_space, base_file_id):
    for train_items, test_items in eval_space[DATA_ITEMS]:
        base_exp_args.num_train_items = train_items
        base_exp_args.num_test_items = test_items
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

                                if approach == "baseline" or approach == "shift":
                                    base_exp_args.load_full = True
                                else:
                                    base_exp_args.load_full = False

                                file_id = (f"{base_file_id}-distribution-{distribution}-approach-{approach}"
                                           f"-cache-{cache_location}-snapshot-{snapshot_set}"
                                           f"-models-{num_models}-items-{train_items + test_items}-level-{bench_level}")

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

def prune_eval_sapce(eval_space, missing_exps):
    """
    Prune eval_space to only contain values present in missing_exps.
    """
    # 1. Initialize a copy of eval_space where each value is an empty set
    pruned = {k: set() for k in eval_space}

    # 2. Iterate over missing_exps keys and parse the string
    for key in missing_exps.keys():
        # Example key:
        # "{base_file_id}-distribution-{distribution}-approach-{approach}-cache-{cache}-snapshot-{snapshot}-models-{models}-items-{items}-level-{level}"
        # We want to extract the values between the known tags.
        # We'll split by the tags.
        def extract_between(s, before, after):
            # returns the substring between before and after
            i = s.index(before) + len(before)
            j = s.index(after, i)
            return s[i:j]

        try:
            # Find all values
            distribution = extract_between(key, "-distribution-", "-approach-")
            approach = extract_between(key, "-approach-", "-cache-")
            cache = extract_between(key, "-cache-", "-snapshot-")
            snapshot = extract_between(key, "-snapshot-", "-models-")
            models = extract_between(key, "-models-", "-items-")
            items_str = extract_between(key, "-items-", "-level-")
            level = key.split("-level-")[-1]
        except Exception:
            # If parsing fails, skip this key
            continue

        # 3. Convert items_str back into a tuple (train_items, test_items)
        total_items = int(items_str)
        items_tuple = None
        for tup in eval_space.get("data_itmes", []):
            if isinstance(tup, (tuple, list)) and len(tup) == 2:
                if sum(tup) == total_items:
                    items_tuple = tuple(tup)
                    break
        if items_tuple is None:
            # If not found, skip
            continue

        # 4. Append each value into the respective set in pruned
        pruned.get("distributions", set()).add(distribution)
        pruned.get("approaches", set()).add(approach)
        pruned.get("default_cache_locations", set()).add(cache)
        pruned.get("snapshot_set_strings", set()).add(snapshot)
        pruned.get("nums_models", set()).add(int(models))
        pruned.get("benchmark_levels", set()).add(level)
        pruned.get("data_itmes", set()).add(items_tuple)

    # 5. Convert each set back into a list
    for k in pruned:
        pruned[k] = list(pruned[k])

    # 6. Return the pruned eval space dictionary
    return pruned

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

    # Call the main function with parsed arguments
    # run_experiment_section(exp_args, args.config_section)

    # run once to for detailed numbers
    eval_space = {
        DISTRIBUTIONS: [TOP_LAYERS, TWENTY_FIVE_PERCENT, FIFTY_PERCENT],
        APPROACHES: ["baseline", "shift", "mosix"],
        DEFAULT_CACHE_LOCATIONS: ["CPU"],
        # SNAPSHOT_SET_STRINGS: [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32],
        SNAPSHOT_SET_STRINGS: [RESNET_18],  # TODO extend to other models as well
        NUMS_MODELS: [35],
        BENCHMARK_LEVELS: ["EXECUTION_STEPS"],
        DATA_ITEMS: [(1600, 400), (6400, 1600)]
    }

    # TODO put in readme how to extend to multiple runs
    # for reproducibility probably enough if we run once
    num_runs = 1
    missing_exps = identify_missing_experiments(exp_args, eval_space, args.base_config_section, num_runs,
                                                exp_args.result_dir)

    while len(missing_exps) > 0:
        pruned_eval_space = prune_eval_sapce(eval_space, missing_exps)
        print("pruned_eval_space")
        print(pruned_eval_space)

        run_exp_set(exp_args, pruned_eval_space, base_file_id=args.base_config_section)

        missing_exps = identify_missing_experiments(exp_args, eval_space, args.base_config_section, num_runs,
                                                    exp_args.result_dir)

