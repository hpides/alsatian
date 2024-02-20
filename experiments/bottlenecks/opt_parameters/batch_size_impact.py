import configparser
import os
import time

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform
from custom.models.init_models import initialize_model
from experiments.bottlenecks.model_rank.experiment_args import ExpArgs
from experiments.bottlenecks.model_rank.score_model import _load_data_to_device, _inference
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, END_TO_END
from global_utils.device import get_device
from global_utils.model_names import VISION_MODEL_CHOICES
from global_utils.write_results import write_measurements_and_args_to_json_file

if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
    config_section = 'batch-size-impact-des-gpu'

    # Read configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    exp_args = ExpArgs(config, config_section)

    batch_sizes = [32, 128, 512, 1024]
    model_names = VISION_MODEL_CHOICES

    measurements = {model_name: {} for model_name in model_names}

    for model_name in model_names:
        for batch_size in batch_sizes:
            print(f"{batch_size} - {model_name}")
            exp_args.model_name = model_name
            exp_args.extract_batch_size = batch_size
            exp_args.num_items = 10 * max(batch_sizes)

            model = initialize_model(exp_args.model_name, pretrained=True, features_only=True)
            data_set = CustomImageFolder(os.path.join(exp_args.dataset_path, 'train'), imagenet_inference_transform)
            data_loader = torch.utils.data.DataLoader(data_set, batch_size=exp_args.extract_batch_size, shuffle=False,
                                                      num_workers=exp_args.data_workers)
            device = get_device(exp_args.device)
            bench = Benchmarker(device)
            bench.add_task(DATA_TO_DEVICE)
            bench.add_task(INFERENCE)
            bench.warm_up_gpu()
            time.sleep(2)

            model = model.to(device)

            single_config_measures = {}
            single_config_measures[LOAD_DATA] = []
            single_config_measures[DATA_TO_DEVICE] = []
            single_config_measures[INFERENCE] = []

            end_to_end_start = time.perf_counter()

            with torch.no_grad():
                start = time.perf_counter()

                for i, (inputs, l) in enumerate(data_loader):
                    # measure data loading time
                    single_config_measures[LOAD_DATA].append(time.perf_counter() - start)

                    bench.register_cuda_start(DATA_TO_DEVICE, i)
                    batch = _load_data_to_device(inputs, device)
                    bench.register_cuda_end(DATA_TO_DEVICE, i)

                    bench.register_cuda_start(INFERENCE, i)
                    out = _inference(batch, model)
                    bench.register_cuda_end(INFERENCE, i)

                    start = time.perf_counter()

            bench.sync_and_summarize_tasks()
            single_config_measures[END_TO_END] = time.perf_counter() - end_to_end_start
            single_config_measures[DATA_TO_DEVICE] = bench.get_task_times(DATA_TO_DEVICE)
            single_config_measures[INFERENCE] = bench.get_task_times(INFERENCE)

            torch.cuda.empty_cache()
            time.sleep(2)
            measurements[model_name][str(batch_size)] = single_config_measures

    write_measurements_and_args_to_json_file(
        measurements=measurements,
        args=exp_args.get_dict(),
        dir_path=exp_args.result_dir,
        file_id=f'batch_size_impact'
    )
