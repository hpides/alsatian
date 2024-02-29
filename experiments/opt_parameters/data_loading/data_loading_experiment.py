import configparser
import os
import time

import torch

from experiments.opt_parameters.data_loading.experiment_args import ExpArgs
from global_utils.constants import LOAD_DATA
from global_utils.dummy_dataset import DummyDataset
from global_utils.write_results import write_measurements_and_args_to_json_file

if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../config.ini')
    config_section = 'data-loading-exp-params-des-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 32, 48, 64]
    sleeps = [None, 2]

    dataset_size = 1024 * 10
    data_set = DummyDataset(1024 * 10, (3, 224, 224), (1,), exp_args.dummy_input_dir, saved_items=dataset_size)

    for batch_size in batch_sizes:
        for num_workers in nums_workers:
            for sleep in sleeps:
                print(f"{batch_size} - {num_workers}")
                exp_args.batch_size = batch_size
                exp_args.data_workers = num_workers

                data_loader = torch.utils.data.DataLoader(data_set, batch_size=exp_args.batch_size,
                                                          shuffle=False,
                                                          num_workers=exp_args.data_workers)

                measurements = {}
                measurements[LOAD_DATA] = []

                end_to_end_start = time.perf_counter()

                with torch.no_grad():
                    start = time.perf_counter()

                    for i, (inputs, l) in enumerate(data_loader):
                        # measure data loading time
                        measurements[LOAD_DATA].append(time.perf_counter() - start)

                        # simulate the time it takes to compute inference
                        if sleep is not None:
                            time.sleep(sleep)

                        start = time.perf_counter()

                        if i > 9:
                            break

                time.sleep(2)

                exp_id = f'data-loading-exp-params-des-gpu-{num_workers}-batch_size-{batch_size}-sleep-{sleep}'
                write_measurements_and_args_to_json_file(
                    measurements=measurements,
                    args=exp_args.get_dict(),
                    dir_path=exp_args.result_dir,
                    file_id=exp_id
                )
