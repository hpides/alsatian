import os.path

import numpy as np
from matplotlib import pyplot as plt

from experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from global_utils.constants import END_TO_END, DETAILED_TIMES
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import RESNET_18, RESNET_152

SH_RANK_ITERATION_DETAILS = 'sh_rank_iteration_details'

SH_RANK_ITERATION = 'sh_rank_iteration'
SUM_SH_RANK_ITERATION = 'sum_sh_rank_iteration'

SH_ITERATIONS = "sh_iterations"

SUM_CLEAN_TIMES = "sum_clean_times"

SUM_DETAILED_TIMES_NO_CLEANUP = "sum_detailed_times_no_cleanup"

SUM_DETAILED_TIMES = "sum_detailed_times"

CLEAR_CACHES = "clear_caches"


def _non_relevant_key(key, ignore_prefixes):
    for pre in ignore_prefixes:
        if pre in key:
            return True
    return False


def sum_up_level(data, levels=1, ignore_prefixes=[]):
    total_sum = 0
    for key, value in data.items():
        if not _non_relevant_key(key, ignore_prefixes):
            if isinstance(value, (int, float)):
                total_sum += value
            elif levels > 1 and isinstance(value, dict):
                total_sum += sum_up_level(value, levels - 1)

    return total_sum


def extract_metrics_of_interest(measurements):
    detailed_times = measurements[DETAILED_TIMES]
    sum_detailed_times = sum_up_level(detailed_times)
    sum_detailed_times_no_cleanup = sum_up_level(detailed_times, levels=1, ignore_prefixes=[CLEAR_CACHES])
    sum_clean_times = sum_up_level(detailed_times, levels=1, ignore_prefixes=["sh_rank"])

    result = {
        END_TO_END: measurements[END_TO_END],
        SUM_DETAILED_TIMES: sum_detailed_times,
        SUM_DETAILED_TIMES_NO_CLEANUP: sum_detailed_times_no_cleanup,
        SUM_CLEAN_TIMES: sum_clean_times,
        SH_ITERATIONS: {}
    }

    sh_i = 0

    while f'{SH_RANK_ITERATION}_{sh_i}' in detailed_times:
        sh_iteration = detailed_times[f'{SH_RANK_ITERATION}_{sh_i}']
        sum_iteration_times = sum_up_level(detailed_times[f'{SH_RANK_ITERATION_DETAILS}_{sh_i}'], levels=2)
        result[SH_ITERATIONS][sh_i] = {
            SH_RANK_ITERATION: sh_iteration,
            SUM_SH_RANK_ITERATION: sum_iteration_times
        }
        sh_i += 1

    return result


def extract_times_of_interest(root_dir, file_id):
    # find file
    files = extract_files_by_name(root_dir, [file_id])
    # TODO so far we expect only 1 file
    assert len(files) == 1

    # parse file
    data = parse_json_file(files[0])
    measurements = data[MEASUREMENTS]

    # actual extraction
    metrics_of_interest = extract_metrics_of_interest(measurements)
    print(metrics_of_interest)

    # check validity of data
    # check diff between measured end to end time and the sum of the more detailed times
    diff_end_to_end_vs_details = metrics_of_interest[END_TO_END] - metrics_of_interest[SUM_DETAILED_TIMES]
    assert diff_end_to_end_vs_details > 0 and diff_end_to_end_vs_details < 1
    # check if the time summed up times without
    no_cleanup_plus_cleanup = \
        metrics_of_interest[SUM_DETAILED_TIMES_NO_CLEANUP] + metrics_of_interest[SUM_CLEAN_TIMES]
    assert (metrics_of_interest[END_TO_END] - no_cleanup_plus_cleanup) > 0
    assert (metrics_of_interest[END_TO_END] - no_cleanup_plus_cleanup) < 1

    return metrics_of_interest


def end_to_end_plot_times(root_dir, models, approaches, distribution, caching_location, num_models, measure_type):
    model_measurements = {}
    for model in models:
        model_measurements[model] = {}
        for approach in approaches:
            config = [distribution, approach, caching_location, model, num_models, measure_type]
            file_id = file_template.format(*config)
            times = extract_times_of_interest(root_dir, file_id)
            model_measurements[model][approach] = times[SUM_DETAILED_TIMES_NO_CLEANUP]

    return model_measurements


def plot_end_to_end_times(data_root_dir, models, approaches, distribution, caching_location, num_models, measure_type,
                          plot_save_path):
    # Extracting the data
    data = end_to_end_plot_times(
        data_root_dir, models, approaches, distribution, caching_location, num_models, measure_type)
    models = list(data.keys())
    methods = list(next(iter(data.values())).keys())
    # Number of models and methods
    n_models = len(models)
    n_methods = len(methods)
    # Creating a bar plot
    bar_width = 0.2
    index = np.arange(n_models)
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Plot each method
    for i, method in enumerate(methods):
        method_values = [data[model][method] for model in models]
        ax.bar(index + i * bar_width, method_values, bar_width, label=method)
    # Adding labels and title
    ax.set_xlabel('Model Architectures')
    ax.set_ylabel('Time in seconds')
    ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend()
    # Save the plot as SVG and PNG
    plt.tight_layout()
    plot_file_name = f'end_to_end-{distribution}-{caching_location}-{num_models}-{measure_type}-{models}'
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))


if __name__ == '__main__':
    root_dir = '/Users/nils/uni/programming/model-search-paper/experiments/model_search/results/dummy'
    file_template = 'des-gpu-imagenette-base-distribution-{}-approach-{}-cache-{}-snapshot-{}-models-{}-level-{}.json'

    config = ['TOP_LAYERS', 'mosix', 'CPU', 'resnet152', '35', 'EXECUTION_STEPS']
    file_id = file_template.format(*config)
    times = extract_times_of_interest(root_dir, file_id)
    print(times)

    models = [RESNET_18, RESNET_152]
    approaches = ['baseline', 'shift', 'mosix']
    distribution = 'TOP_LAYERS'
    caching_location = 'CPU'
    num_models = 35
    measure_type = 'EXECUTION_STEPS'
    t = end_to_end_plot_times(root_dir, models, approaches, distribution, caching_location, num_models, measure_type)
    plot_save_path = './plots'

    plot_end_to_end_times(root_dir, models, approaches, distribution, caching_location, num_models, measure_type,
                          plot_save_path)
