import configparser
import os
import time

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform
from custom.models.init_models import initialize_model
from experiments.opt_parameters.num_workers_and_batch_size.experiment_args import ExpArgs
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, END_TO_END, CUDA
from global_utils.device import get_device
from global_utils.dummy_dataset import DummyDataset
from global_utils.model_names import VISION_MODEL_CHOICES
from global_utils.write_results import write_measurements_and_args_to_json_file


def _load_data_to_device(batch, device):
    # if the device is GPU load, otherwise do nothing
    if device.type == CUDA:
        batch = batch.to(device)
    return batch


def _inference(batch, model):
    model.eval()
    with torch.set_grad_enabled(False):
        out = model(batch)
        return out


if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
    config_section = 'num-workers-and-batch-size-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 12, 16, 32, 48, 64]
    model_names = VISION_MODEL_CHOICES

    for dataset_type in ['imagenette', 'preprocessed_ssd']:

        data_set = CustomImageFolder(os.path.join(exp_args.dataset_path, 'train'), imagenet_inference_transform)
        dataset_len = int(len(data_set) / max(batch_sizes)) * max(batch_sizes)
        data_set.set_subrange(0, dataset_len)

        if dataset_type == 'preprocessed_ssd':
            data_set = DummyDataset(dataset_len, (3, 224, 224), (1,), exp_args.dummy_input_dir, saved_items=dataset_len)

        for model_name in model_names:
            for batch_size in batch_sizes:
                for num_workers in nums_workers:
                    print(f"{batch_size} - {num_workers} - {model_name}")

                    model = initialize_model(model_name, pretrained=True, features_only=True)
                    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                                              shuffle=False,
                                                              num_workers=num_workers)
                    device = get_device(exp_args.device)
                    bench = Benchmarker(device)
                    bench.add_task(DATA_TO_DEVICE)
                    bench.add_task(INFERENCE)
                    bench.warm_up_gpu()

                    model = model.to(device)

                    measurements = {}
                    measurements[LOAD_DATA] = []
                    measurements[DATA_TO_DEVICE] = []
                    measurements[INFERENCE] = []

                    end_to_end_start = time.perf_counter()

                    with torch.no_grad():
                        start = time.perf_counter()

                        for i, (inputs, l) in enumerate(data_loader):
                            # measure data loading time
                            measurements[LOAD_DATA].append(time.perf_counter() - start)

                            bench.register_cuda_start(DATA_TO_DEVICE, i)
                            batch = _load_data_to_device(inputs, device)
                            bench.register_cuda_end(DATA_TO_DEVICE, i)

                            bench.register_cuda_start(INFERENCE, i)
                            out = _inference(batch, model)
                            bench.register_cuda_end(INFERENCE, i)

                            start = time.perf_counter()

                    bench.sync_and_summarize_tasks()
                    measurements[END_TO_END] = time.perf_counter() - end_to_end_start
                    measurements[DATA_TO_DEVICE] = bench.get_task_times(DATA_TO_DEVICE)
                    measurements[INFERENCE] = bench.get_task_times(INFERENCE)

                    torch.cuda.empty_cache()
                    time.sleep(2)

                    exp_id = f'param-analysis-{model_name}-workers-{num_workers}-batch_size-{batch_size}-dataset_type-{dataset_type}'
                    write_measurements_and_args_to_json_file(
                        measurements=measurements,
                        args=exp_args.get_dict(),
                        dir_path=exp_args.result_dir,
                        file_id=exp_id
                    )
