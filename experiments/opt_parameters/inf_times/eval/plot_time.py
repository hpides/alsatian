from statistics import median, mean

from experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import INFERENCE
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES, CONVOLUTION_MODELS, TRANSFORMER_MODELS
import matplotlib.pyplot as plt
import numpy as np


def get_file_id(model_name, batch_size):
    return f"inference_time-des_gpu-model_name-{model_name}-batch_size-{batch_size}"


def collect_inf_time_data(root_dir, model_names, batch_sizes, agg_func=None, normalize=False):
    inf_times = {}
    for model_name in model_names:
        inf_times[model_name] = {}
        for batch_size in batch_sizes:
            file_id = get_file_id(model_name, batch_size)
            measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]

            _data = measurements[INFERENCE]

            if agg_func:
                _data = agg_func(measurements[INFERENCE])

            if normalize:
                _data = _data / batch_size

            inf_times[model_name][batch_size] = _data


    return inf_times

def plot_inf_times(data, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    models = list(data.keys())
    batch_sizes = [32, 128, 256, 512, 1024]

    bar_width = 0.1
    index = range(len(models))

    for i, batch_size in enumerate(batch_sizes):
        values = [data[model][batch_size] for model in models]
        ax.bar([x + i * bar_width for x in index], values, bar_width, label=f'Batch Size: {batch_size}')

    ax.set_xlabel('Model names')
    ax.set_ylabel('Time in seconds')
    ax.set_xticks([i + (len(batch_sizes) - 1) * bar_width / 2 for i in index])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    # Save plot as SVG
    plt.savefig(f'{save_path}.svg', format='svg')

    # Save plot as PNG
    plt.savefig(f'{save_path}.png', format='png')


if __name__ == '__main__':
    root_dir = '/Users/nils/Downloads/inference-time-exp'
    batch_sizes = [32, 128, 256, 512, 1024]

    save_path = '../plots/inf_times_'

    model_names = CONVOLUTION_MODELS
    data = collect_inf_time_data(root_dir, model_names, batch_sizes, agg_func=median, normalize=True)
    plot_inf_times(data, f'{save_path}_conv_models')

    model_names = TRANSFORMER_MODELS
    data = collect_inf_time_data(root_dir, model_names, batch_sizes, agg_func=median, normalize=True)
    plot_inf_times(data, f'{save_path}_trans_models')


    # for batch_size in batch_sizes:
    #         for dataset_type in dataset_types:
    #             _id = f'batch_size-{batch_size}-sleep-{sleep}-data-{dataset_type}'
    #             output_path = f'../plots/workers_impact/{_id}'
    #             data = collect_batch_size_data(root_dir, batch_size, sleep, dataset_type, nums_workers, last_batch=9)
    #             multi_line_plot(data, output_path)
