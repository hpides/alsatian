import matplotlib.pyplot as plt

from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import END_TO_END
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import *


def get_metric_numbers_per_model_vary_batch_size(root_dir, metrics, model_names, batch_sizes, num_workers,
                                                 dataset_type,
                                                 batch_size_normalized=False, agg_func=None):
    data = {}
    for model_name in model_names:
        data[model_name] = {}
        for batch_size in batch_sizes:
            data[model_name][batch_size] = {}
            file_id = get_file_id(batch_size, model_name, num_workers, dataset_type)
            measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
            for metric in metrics:
                _times = measurements[metric]
                if batch_size_normalized:
                    _times = [t / batch_size for t in _times]
                if agg_func:
                    _times = agg_func(_times)
                else:
                    data[model_name][batch_size][metric] = _times
    return data


def get_metric_numbers_per_model_vary_workers_size(root_dir, metrics, model_names, batch_size, nums_workers,
                                                   dataset_type,
                                                   batch_size_normalized=False, agg_func=None):
    data = {}
    for model_name in model_names:
        data[model_name] = {}
        for num_workers in nums_workers:
            data[model_name][num_workers] = {}
            file_id = get_file_id(batch_size, model_name, num_workers, dataset_type)
            measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
            for metric in metrics:
                _times = measurements[metric]
                if batch_size_normalized:
                    _times = [t / batch_size for t in _times]
                if agg_func:
                    _times = agg_func(_times)
                else:
                    data[model_name][num_workers][metric] = _times
    return data


def get_metric_numbers_one_model(root_dir, metrics, model_name, batch_sizes, nums_workers, dataset_type,
                                 batch_size_normalized=False, agg_func=None):
    data = {}
    for batch_size in batch_sizes:
        data[batch_size] = {}
        for num_workers in nums_workers:
            data[batch_size][num_workers] = {}
            file_id = get_file_id(batch_size, model_name, num_workers, dataset_type)
            measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
            for metric in metrics:
                _times = measurements[metric]
                if batch_size_normalized:
                    _times = [t / batch_size for t in _times]
                if agg_func:
                    _times = agg_func(_times)
                else:
                    data[batch_size][num_workers][metric] = _times
    return data


def get_file_id(batch_size, model_name, num_workers, dataset_type):
    file_id = f"param-analysis-{model_name}-workers-{num_workers}-batch_size-{batch_size}-dataset_type-{dataset_type}"
    return file_id


def plot_single_aggregated_metric(data, save_path, metric_name, x_axis_label):
    fig, ax = plt.subplots(figsize=(12, 8))

    models = list(data.keys())
    sub_category = list(data[models[0]].keys())

    bar_width = 0.1
    index = range(len(models))

    for i, batch_size in enumerate(sub_category):
        values = [data[model][batch_size][metric_name] for model in models]
        ax.bar([x + i * bar_width for x in index], values, bar_width, label=f'{batch_size}')

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel('Time in seconds')
    ax.set_xticks([i + (len(sub_category) - 1) * bar_width / 2 for i in index])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    # Save plot as SVG
    plt.savefig(f'{save_path}.svg', format='svg')

    # Save plot as PNG
    plt.savefig(f'{save_path}.png', format='png')

    plt.close()


if __name__ == '__main__':
    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 12, 16, 32, 48, 64]
    model_names = VISION_MODEL_CHOICES
    dataset_types = ['imagenette', 'preprocessed_ssd']

    root_dir = '/Users/nils/Downloads/worker_batch_size_impact'

    # plot analysis of batch size impact
    for num_workers in nums_workers:
        for dataset_type in dataset_types:
            data = get_metric_numbers_per_model_vary_batch_size(
                root_dir, [END_TO_END], VISION_MODEL_CHOICES, batch_sizes, num_workers, dataset_type,
                batch_size_normalized=False
            )
            save_path = f'../plots/end_to_end-dataset_type-{dataset_type}-num_workers-{num_workers}'
            plot_single_aggregated_metric(data, save_path, END_TO_END, 'model architectures')

    for batch_size in batch_sizes:
        for dataset_type in dataset_types:
            data = get_metric_numbers_per_model_vary_workers_size(
                root_dir, [END_TO_END], VISION_MODEL_CHOICES, batch_size, nums_workers, dataset_type,
                batch_size_normalized=False
            )
            save_path = f'../plots/end_to_end-dataset_type-{dataset_type}-batch_size-{batch_size}'
            plot_single_aggregated_metric(data, save_path, END_TO_END, 'model architectures')

    for model_name in model_names:
        for dataset_type in dataset_types:
            data = get_metric_numbers_one_model(
                root_dir, [END_TO_END], model_name, batch_sizes, nums_workers, dataset_type,
                batch_size_normalized=False
            )
            save_path = f'../plots/end_to_end-dataset_type-{dataset_type}-model_name-{model_name}'
            plot_single_aggregated_metric(data, save_path, END_TO_END, 'batch_sizes')
