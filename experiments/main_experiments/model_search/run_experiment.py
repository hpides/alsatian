import argparse
import configparser
import os

from experiments.model_search.experiment_args import ExpArgs
from experiments.model_search.model_search_exp import run_model_search
from experiments.main_experiments.prevent_caching.watch_utils import LIMIT_IO
from global_utils.constants import SSD_CACHING_ACTIVE
from global_utils.deterministic import TRUE
from global_utils.write_results import write_measurements_and_args_to_json_file

def run_exp(exp_args):
    return run_model_search(exp_args)


def run_experiment_section(exp_args, section, file_id=None):
    print(f'run experiment:{exp_args}')

    # code to start experiment here
    measurements, result = run_exp(exp_args)

    if file_id is None:
        file_id = f'score_model_exp_section_{section}'

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
    parser.add_argument('--config_section', default='debug-des-gpu-out-of-memory',
                        help='Exact Configuration identified by the section in the configuration file')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.config_section)

    if exp_args.limit_fs_io:
        os.environ[LIMIT_IO] = TRUE

    if exp_args.ssd_caching_active:
        os.environ[SSD_CACHING_ACTIVE] = TRUE

    # Call the main function with parsed arguments
    run_experiment_section(exp_args, args.config_section)
