import statistics

from experiments.side_experiments.plot_shared.data_transform import aggregate_measurements
from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from experiments.side_experiments.plot_shared.plotting_util import plot_stacked_bar_chart
from global_utils.global_constants import *


def aggregate_over_epochs(measurements):
    # aggregate within epoch
    extracted_iterations = {}
    for i in measurements.keys():
        iteration = measurements[i]
        extracted_iterations[i] = {}
        train_numbers = iteration[TRAIN]
        val_numbers = iteration[VAL]

        extracted_iterations[i][TRAIN_DATA_LOAD] = sum(train_numbers['time-data-loading'])
        extracted_iterations[i][VAL_DATA_LOAD] = sum(val_numbers['time-data-loading'])

        extracted_iterations[i][TRAIN] = train_numbers[TIME] - extracted_iterations[i][TRAIN_DATA_LOAD]
        extracted_iterations[i][VAL] = val_numbers[TIME] - extracted_iterations[i][VAL_DATA_LOAD]

    # aggregate across epochs
    result = aggregate_measurements(extracted_iterations, sum)

    return result


def get_categorized_times(root_dir, env_name, model_name, train_size, val_size, fine_tuning_variant):
    search_strings = [f'env_name-{env_name}', f'model_name-{model_name}', f'train_size-{train_size}-',
                      f'val_size-{val_size}', f"fine_tuning_variant-{fine_tuning_variant}"]
    data = get_raw_data(root_dir, search_strings, expected_files=5)

    aggregated_epochs = []
    for rep in data:
        measurements = rep[MEASUREMENTS]
        aggregated_epochs.append(aggregate_over_epochs(measurements))

    aggregated_median = aggregate_measurements(aggregated_epochs, statistics.median)

    return aggregated_median


def get_aggregated_times_per_model(model_names, root_dir, env_name, train_size, val_size, fine_tuning_variant):
    measurements = {}
    for model in model_names:
        measurements[model] = get_categorized_times(root_dir, env_name, model, train_size, val_size,
                                                    fine_tuning_variant)
    return measurements


def bar_plot_fine_tuning(model_names, root_dir, env_name, train_size, val_size, fine_tuning_variant, plot_kwargs=None):
    model_times = get_aggregated_times_per_model(model_names, root_dir, env_name, train_size, val_size,
                                                 fine_tuning_variant)
    plot_stacked_bar_chart(model_times, **plot_kwargs)
