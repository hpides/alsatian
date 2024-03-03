import configparser
import os
import time

import torch

from experiments.bottlenecks.model_rank.experiment_args import ExpArgs
from experiments.bottlenecks.model_rank.run_experiment import run_experiment_section
from global_utils.model_names import VISION_MODEL_CHOICES

if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'num_workers_and_batch_size/config.ini')
    config_section = 'debug-des-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    # start with batch size 16 then double
    batch_size = 32
    model_names = VISION_MODEL_CHOICES
    max_batch_sizes = {}

    while len(model_names) > 0:
        for model_name in model_names:
            try:
                exp_args.model_name = model_name
                exp_args.extract_batch_size = batch_size
                exp_args.num_items = exp_args.extract_batch_size

                run_experiment_section(exp_args, config_section)
                torch.cuda.empty_cache()
                time.sleep(2)
                max_batch_sizes[model_name] = batch_size
                print(f'{model_name} passed batch size {batch_size}')
            except:
                model_names.remove(model_name)

        batch_size = batch_size * 2
        print('new batch size')

    print("MAX BATCH SIZES")
    print(max_batch_sizes)
