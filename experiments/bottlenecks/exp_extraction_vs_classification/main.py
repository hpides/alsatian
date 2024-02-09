import argparse
import configparser

from global_utils.write_results import write_measurements_and_args_to_json_file


def run_exp(exp_args):
    return {'dummy_result': 123}


def main(exp_args):
    print(f'run experiment:{exp_args}')

    # code to start experiment here
    measurements = run_exp(exp_args)

    write_measurements_and_args_to_json_file(
        measurements=measurements,
        args=exp_args.get_dict(),
        dir_path=exp_args.result_dir,
        file_id=f'SPECIFY-GOOD_FILE-NAME'
    )


class ExpArgs:
    def __init__(self, args, section):
        self.model_name = args[section]['model_name']
        self.result_dir = args[section]['result_dir']
        self.dataset_path = args[section]['dataset_path']
        self.extract_batch_size = args.getint(section, 'extract_batch_size')
        self.classify_batch_size = args.getint(section, 'classify_batch_size')
        self.num_items = args.getint(section, 'num_items')

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_file', default='./config.ini', help='Configuration file path')
    parser.add_argument('--config_section', default='debug-local',
                        help='Exact Configuration identified by the section in the configuration file')
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    exp_args = ExpArgs(config, args.config_section)

    # Call the main function with parsed arguments
    main(exp_args)
