import argparse
import configparser

from experiments.bottlenecks.model_rank.experiment_args import ExpArgs
from experiments.bottlenecks.model_rank.score_model import score_model_exp
from global_utils.write_results import write_measurements_and_args_to_json_file


def run_exp(exp_args):
    return score_model_exp(exp_args)


def main(exp_args, section):
    print(f'run experiment:{exp_args}')

    # code to start experiment here
    measurements = run_exp(exp_args)

    write_measurements_and_args_to_json_file(
        measurements=measurements,
        args=exp_args.get_dict(),
        dir_path=exp_args.result_dir,
        file_id=f'score_model_exp_section_{section}'
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config.ini', help='Configuration file path')
    parser.add_argument('--config_section', default='debug-local',
                        help='Exact Configuration identified by the section in the configuration file')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.config_section)

    # Call the main function with parsed arguments
    main(exp_args, args.config_section)
