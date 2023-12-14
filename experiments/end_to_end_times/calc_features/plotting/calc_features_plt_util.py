import statistics

from experiments.plot_shared.data_transform import aggregate_measurements
from experiments.plot_shared.file_parsing import get_raw_data
from experiments.plot_shared.plotting_util import plot_stacked_bar_chart
from global_utils.global_constants import *


def get_categorized_times(root_dir, env_name, model_name, train_size, val_size, iteration=None):
    search_strings = [f'env_name-{env_name}', f'model_name-{model_name}', f'train_size-{train_size}-',
                      f'val_size-{val_size}', f'env_name-{env_name}']
    data = get_raw_data(root_dir, search_strings)

    extracted_iterations = {}
    measurements = data[MEASUREMENTS]

    for i in measurements.keys():
        iteration = measurements[i]
        extracted_iterations[i] = {}
        train_numbers = iteration[TRAIN]
        val_numbers = iteration[VAL]

        extracted_iterations[i][TRAIN_DATA_LOAD] = sum(train_numbers['time-data-loading'])
        extracted_iterations[i][VAL_DATA_LOAD] = sum(val_numbers['time-data-loading'])

        extracted_iterations[i][TRAIN] = train_numbers[TIME] - extracted_iterations[i][TRAIN_DATA_LOAD]
        extracted_iterations[i][VAL] = val_numbers[TIME] - extracted_iterations[i][VAL_DATA_LOAD]

    aggregated_median = aggregate_measurements(extracted_iterations, statistics.median)

    return aggregated_median


def get_aggregated_times_per_model(model_names, root_dir, env_name, train_size, val_size, iteration=None):
    measurements = {}
    for model in model_names:
        measurements[model] = get_categorized_times(root_dir, env_name, model, train_size, val_size, iteration)
    return measurements


def bar_plot_calc_features(model_names, root_dir, env_name, train_size, val_size, iteration=None, plot_kwargs=None):
    model_times = get_aggregated_times_per_model(model_names, root_dir, env_name, train_size, val_size, iteration)
    plot_stacked_bar_chart(model_times, **plot_kwargs)


if __name__ == '__main__':
    get_categorized_times(
        '/Users/nils/uni/programming/model-search-paper/experiments/end_to_end_times/calc_features/results',
        'DES-GPU-SERVER', 'vit_b_16', 800, 200, iteration=None)
