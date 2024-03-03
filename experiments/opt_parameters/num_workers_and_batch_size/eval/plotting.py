import os.path
from statistics import median, mean

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import END_TO_END, LOAD_DATA, DATA_TO_DEVICE, INFERENCE
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import *


def collect_and_aggregate_data(root_dir, model_names, batch_sizes, nums_workers):
    data = {}
    for model_name in model_names:
        data[model_name] = {}
        for num_workers in nums_workers:
            data[model_name][num_workers] = {}
            for batch_size in batch_sizes:
                data[model_name][num_workers][batch_size] = {}
                file_id = get_file_id(batch_size, model_name, num_workers)
                measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
                data[model_name][num_workers][batch_size][END_TO_END] = measurements[END_TO_END]
                sum_end_to_end = 0
                for metric in [LOAD_DATA, DATA_TO_DEVICE, INFERENCE]:
                    _times = measurements[metric]
                    sum_times = sum(_times)
                    median_times = median(_times)
                    avg_times = mean(_times)
                    data[model_name][num_workers][batch_size][f'raw_numbers_{metric}'] = sum_times
                    data[model_name][num_workers][batch_size][f'sum_{metric}'] = sum_times
                    data[model_name][num_workers][batch_size][f'median_{metric}'] = median_times
                    data[model_name][num_workers][batch_size][f'avg_{metric}'] = avg_times
                    data[model_name][num_workers][batch_size][f'norm_sum_{metric}'] = sum_times / 10 * 1024
                    data[model_name][num_workers][batch_size][f'norm_median_{metric}'] = median_times / batch_size
                    data[model_name][num_workers][batch_size][f'norm_avg_{metric}'] = avg_times / batch_size
                    sum_end_to_end += sum_times
                data[model_name][num_workers][batch_size][f'sum_{END_TO_END}'] = sum_end_to_end

    return data


def get_raw_metric_numbers_per_model(root_dir, metric, model_names, batch_sizes, num_workers,
                                     batch_size_normalized=False):
    data = {}
    for model_name in model_names:
        data[model_name] = {}
        for batch_size in batch_sizes:
            data[model_name][batch_size] = {}
            file_id = get_file_id(batch_size, model_name, num_workers)
            measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
            _times = measurements[metric]
            if batch_size_normalized:
                _times = [t / batch_size for t in _times]
            data[model_name][batch_size] = _times
    return data


def collect_end_to_end_data(root_dir, model_names, batch_sizes, nums_workers):
    data = {}
    for model_name in model_names:
        data[model_name] = {}
        for num_workers in nums_workers:
            data[model_name][num_workers] = {}
            for batch_size in batch_sizes:
                file_id = get_file_id(batch_size, model_name, num_workers)
                measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
                data[model_name][num_workers][batch_size] = measurements[END_TO_END]
    return data


def get_file_id(batch_size, model_name, num_workers):
    file_id = f"param-analysis-{model_name}-workers-{num_workers}-batch_size-{batch_size}"
    return file_id


def plot_times_for_one_model(root_dir, model_name, batch_sizes, nums_workers, file_path):
    data = collect_end_to_end_data(root_dir, [model_name], batch_sizes, nums_workers)
    # Sample data
    workers = nums_workers
    values = [[data[model_name][w][batch_size] for batch_size in batch_sizes] for w in nums_workers]
    labels = batch_sizes

    num_categories = len(workers)
    num_values = len(values[0])  # Assuming all x_values have the same number of values

    width = 0.8 / num_values  # Adjust the width dynamically based on the number of values

    x = np.arange(num_categories)  # the label locations

    fig, ax = plt.subplots()

    # Plotting each set of values
    for i in range(num_values):
        ax.bar(x + i * width - 0.4, [val[i] for val in values], width, label=labels[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()

    plt.xlabel("Number of workers")
    plt.ylabel("End-to-end time")
    plt.legend(title="Batch sizes")
    plt.tight_layout()  # Adjust layout to prevent overlapping labels

    file_path = os.path.join(file_path, f'batch_size_worker_bench_{model_name}')
    plt.savefig(f'{file_path}.png')
    plt.savefig(f'{file_path}.svg', format='svg')

    # plt.show()

    plt.close()

def boxplot_normalized_inference_times(data):
    # Prepare data for plotting
    models = list(data.keys())
    batch_sizes = sorted(list(data[models[0]].keys()))
    colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'lightskyblue']
    num_models = len(models)

    plt.figure(figsize=(14, 8))

    for i, batch_size in enumerate(batch_sizes):
        x = [model for model in models]
        y_means = [np.mean(data[model][batch_size]) for model in models]
        y_errors = [stats.sem(data[model][batch_size]) * stats.t.ppf(0.975, len(data[model][batch_size]) - 1) for model
                    in models]
        plt.bar(np.arange(num_models) + i * 0.15, y_means, yerr=y_errors, capsize=5, color=colors[i], width=0.15,
                label=f'Batch Size: {batch_size}')
        for j, model in enumerate(models):
            outliers = [datapoint for datapoint in data[model][batch_size] if
                        np.abs(datapoint - np.mean(data[model][batch_size])) > 2 * np.std(data[model][batch_size])]
            for outlier in outliers:
                plt.plot(np.arange(num_models)[j] + i * 0.15, outlier, marker='o', markersize=2, color='red')

    plt.xlabel('Model Name')
    plt.ylabel('Average time per item')
    plt.xticks(np.arange(num_models) + 0.3 / 2, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48]

    root_dir = '/Users/nils/Downloads/opt-parameters-64-cores'
    data = get_raw_metric_numbers_per_model(root_dir, INFERENCE, VISION_MODEL_CHOICES, batch_sizes, 4, batch_size_normalized=True)
    boxplot_normalized_inference_times(data)


    # agg_data = collect_and_aggregate_data(root_dir, VISION_MODEL_CHOICES,
    #                                       batch_sizes, nums_workers)
    #
    # base_path = '/Users/nils/Downloads/opt-parameters-'
    # for config in ['32-cores', '64-cores']:
    #     data_path = base_path + config
    #     plot_path = f'plots-{config}'
    #
    #     for model_name in VISION_MODEL_CHOICES:
    #         plot_times_for_one_model(data_path, model_name, batch_sizes, nums_workers, plot_path)
