import configparser
import os
import time

import torch

from custom.models.init_models import initialize_model
from experiments.opt_parameters.inf_times.experiment_args import ExpArgs
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import INFERENCE
from global_utils.device import get_device
from global_utils.dummy_dataset import DummyDataset
from global_utils.model_names import VISION_MODEL_CHOICES
from global_utils.write_results import write_measurements_and_args_to_json_file

if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), './config.ini')
    config_section = 'inference-time-des-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 32, 48, 64]
    sleeps = [None, 2]

    dataset_size = exp_args.dataset_size
    input_shape = (3, 224, 224)
    data_set = DummyDataset(dataset_size, input_shape, (1,), exp_args.dummy_input_dir, saved_items=dataset_size)

    device = get_device(exp_args.device)
    bench = Benchmarker(device)
    bench.add_task(INFERENCE)

    for batch_size in batch_sizes:
        for model_name in VISION_MODEL_CHOICES:
            print(f"{batch_size} - {model_name}")
            model = initialize_model(model_name, pretrained=True, features_only=True)
            model = model.to(device)
            model.eval()
            data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

            measurements = {}
            measurements[INFERENCE] = []

            with torch.no_grad():

                warm_up_cycles = 10
                random_batch = torch.randn(size=[batch_size] + list(input_shape), dtype=torch.float)
                random_batch = random_batch.to(device)
                for i in range(warm_up_cycles):
                    _ = model(random_batch)

                for i, (inputs, l) in enumerate(data_loader):
                    batch = inputs.to(device)

                    bench.register_cuda_start(INFERENCE, i)
                    out = model(batch)
                    bench.register_cuda_end(INFERENCE, i)

                    if i >= 9:
                        break

            bench.sync_and_summarize_tasks()
            measurements[INFERENCE] = bench.get_task_times(INFERENCE)
            time.sleep(2)

            exp_id = f'inference_time-des_gpu-model_name-{model_name}-batch_size-{batch_size}'
            write_measurements_and_args_to_json_file(
                measurements=measurements,
                args=exp_args.get_dict(),
                dir_path=exp_args.result_dir,
                file_id=exp_id
            )
