import os.path
from statistics import median

import numpy as np
from matplotlib import pyplot as plt

from experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from experiments.snapshots.synthetic.generate import TOP_LAYERS, TWENTY_FIVE_PERCENT, FIFTY_PERCENT
from global_utils.constants import GEN_EXEC_PLAN, GET_COMPOSED_MODEL, MODEL_TO_DEVICE, LOAD_DATA, DATA_TO_DEVICE, \
    CALC_PROXY_SCORE, LOAD_STATE_DICT, INIT_MODEL, STATE_TO_MODEL, INFERENCE, END_TO_END, DETAILED_TIMES, \
    EXEC_STEP_MEASUREMENTS
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import RESNET_18, RESNET_152, VIT_L_32, EFF_NET_V2_L

CALC_PROXY_SCORE_NUMBERS = "calc_proxy_score_numbers"

INFERENCE_DETAILED_NUMBERS = "inference_detailed_numbers"

STEP_DETAILED_NUMS_AGG = "step_detailed_numbers_agg"
SUM_OVER_STEPS_DETAILED_NUMS_AGG = "sum_over_steps_detailed_numbers_agg"

BASELINE = 'baseline'

SH_RANK_ITERATION_DETAILS = 'sh_rank_iteration_details'

SH_RANK_ITERATION = 'sh_rank_iteration'
SUM_SH_RANK_ITERATION = 'sum_sh_rank_iteration'

SH_ITERATIONS = "sh_iterations"

SUM_CLEAN_TIMES = "sum_clean_times"

SUM_DETAILED_TIMES_NO_CLEANUP = "sum_detailed_times_no_cleanup"

SUM_DETAILED_TIMES = "sum_detailed_times"

CLEAR_CACHES = "clear_caches"

DETAILED_METRICS_OF_INTEREST = [GEN_EXEC_PLAN, GET_COMPOSED_MODEL, MODEL_TO_DEVICE, LOAD_DATA, DATA_TO_DEVICE,
                                INFERENCE, STATE_TO_MODEL, INIT_MODEL, LOAD_STATE_DICT, CALC_PROXY_SCORE]

MODEL_NAME_MAPPING = {
    RESNET_18: "ResNet-18",
    RESNET_152: "ResNet-152",
    VIT_L_32: "ViT-L-32",
    EFF_NET_V2_L: "EffNetV2-L"
}



def _non_relevant_key(key, ignore_prefixes):
    for pre in ignore_prefixes:
        if pre in key:
            return True
    return False


def sum_identical_keys(data, metrics_of_interest, accumulator=None):
    if accumulator is None:
        accumulator = {}

    for key, value in data.items():
        if isinstance(value, dict):
            sum_identical_keys(value, metrics_of_interest, accumulator=accumulator)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    sum_identical_keys(item, metrics_of_interest, accumulator=accumulator)
        elif isinstance(value, (int, float)):
            if key in metrics_of_interest:
                if key in accumulator:
                    accumulator[key] += value
                else:
                    accumulator[key] = value

    return accumulator


def sum_up_level(data, levels=1, ignore_prefixes=[]):
    total_sum = 0
    for key, value in data.items():
        if not _non_relevant_key(key, ignore_prefixes):
            if isinstance(value, (int, float)):
                total_sum += value
            elif levels > 1 and isinstance(value, dict):
                total_sum += sum_up_level(value, levels - 1)

    return total_sum


def extract_metrics_of_interest(measurements, approach, include_exec_step_details=True):
    if approach == BASELINE:
        result = {
            END_TO_END: measurements[END_TO_END]
        }

        detailed_times = measurements[DETAILED_TIMES]
        exec_step_measurements = detailed_times[EXEC_STEP_MEASUREMENTS]

        ef_i = 1

        result[INFERENCE_DETAILED_NUMBERS] = {}
        result[CALC_PROXY_SCORE_NUMBERS] = {}
        while (f'{ef_i}-{"BaselineExtractFeaturesStep-details"}' in exec_step_measurements or
               f'{ef_i}-{"ScoreModelStep-details"}' in exec_step_measurements):
            if f'{ef_i}-{"BaselineExtractFeaturesStep-details"}' in exec_step_measurements:
                result[INFERENCE_DETAILED_NUMBERS][ef_i] = {}
                b_step = exec_step_measurements[f'{ef_i}-{"BaselineExtractFeaturesStep-details"}']
                if include_exec_step_details:
                    aggregated_numbers = sum_identical_keys(b_step, DETAILED_METRICS_OF_INTEREST)

                    result[INFERENCE_DETAILED_NUMBERS][ef_i][STEP_DETAILED_NUMS_AGG] = aggregated_numbers
            elif f'{ef_i}-{"ScoreModelStep-details"}' in exec_step_measurements:
                result[CALC_PROXY_SCORE_NUMBERS][ef_i] = {}
                score_step = exec_step_measurements[f'{ef_i}-{"ScoreModelStep-details"}']

                if include_exec_step_details:
                    aggregated_numbers = sum_identical_keys(score_step, DETAILED_METRICS_OF_INTEREST)

                    result[CALC_PROXY_SCORE_NUMBERS][ef_i][STEP_DETAILED_NUMS_AGG] = aggregated_numbers

            ef_i += 1

        result[SUM_OVER_STEPS_DETAILED_NUMS_AGG] = sum_identical_keys(result, DETAILED_METRICS_OF_INTEREST)



    else:
        detailed_times = measurements[DETAILED_TIMES]
        sum_detailed_times = sum_up_level(detailed_times)

        result = {
            END_TO_END: measurements[END_TO_END],
            SUM_DETAILED_TIMES: sum_detailed_times,
            SH_ITERATIONS: {}
        }

        sh_i = 0

        while f'{SH_RANK_ITERATION}_{sh_i}' in detailed_times:
            sh_iteration = detailed_times[f'{SH_RANK_ITERATION}_{sh_i}']
            detailed_sh_iteration_times = detailed_times[f'{SH_RANK_ITERATION_DETAILS}_{sh_i}']
            sum_iteration_times = sum_up_level(detailed_sh_iteration_times, levels=2)
            result[SH_ITERATIONS][sh_i] = {
                SH_RANK_ITERATION: sh_iteration,
                SUM_SH_RANK_ITERATION: sum_iteration_times
            }
            if include_exec_step_details:
                aggregated_numbers = sum_identical_keys(detailed_sh_iteration_times, DETAILED_METRICS_OF_INTEREST)

                result[SH_ITERATIONS][sh_i][STEP_DETAILED_NUMS_AGG] = aggregated_numbers

            sh_i += 1

        result[SUM_OVER_STEPS_DETAILED_NUMS_AGG] = sum_identical_keys(result, DETAILED_METRICS_OF_INTEREST)

    return result


def extract_times_of_interest(root_dir, file_ids, approach, measure_type):
    files = []
    for file_id in file_ids:
        # find file
        files += extract_files_by_name(root_dir, [file_id])
    assert len(files) >= 2

    collected_metrics = []
    for file in files:

        # parse file
        data = parse_json_file(file)
        measurements = data[MEASUREMENTS]

        # actual extraction
        metrics_of_interest = extract_metrics_of_interest(measurements, approach)
        collected_metrics.append(metrics_of_interest)
        print(metrics_of_interest)

        if not approach == BASELINE and not measure_type == "STEPS_DETAILS":
            # check validity of data
            # check diff between measured end to end time and the sum of the more detailed times
            diff_end_to_end_vs_details = metrics_of_interest[END_TO_END] - metrics_of_interest[SUM_DETAILED_TIMES]
            assert diff_end_to_end_vs_details > 0 and diff_end_to_end_vs_details < 2

    return collected_metrics


def regroup_and_rename_times(times):
    grouped_times = {}

    grouped_times['prepare data'] = times[LOAD_DATA]
    grouped_times['prepare data'] += times[DATA_TO_DEVICE]

    grouped_times['prepare model'] = 0
    if LOAD_STATE_DICT in times:
        grouped_times['prepare model'] += times[LOAD_STATE_DICT]
    if INIT_MODEL in times:
        grouped_times['prepare model'] += times[INIT_MODEL]
    if STATE_TO_MODEL in times:
        grouped_times['prepare model'] += times[STATE_TO_MODEL]
    if MODEL_TO_DEVICE in times:
        grouped_times['prepare model'] += times[MODEL_TO_DEVICE]
    if GET_COMPOSED_MODEL in times:
        grouped_times['prepare model'] += times[GET_COMPOSED_MODEL]

    grouped_times[INFERENCE] = times[INFERENCE]

    grouped_times["proxy score"] = times[CALC_PROXY_SCORE]

    if GEN_EXEC_PLAN in times:
        grouped_times['exec planning'] = times[GEN_EXEC_PLAN]

    return grouped_times


def end_to_end_plot_times(root_dir, file_template, models, approaches, distribution, data_items, measure_type):
    model_measurements = {}
    for model in models:
        model_measurements[model] = {}
        for approach in approaches:
            config = [distribution, approach, model, data_items, measure_type]
            file_ids = [file_template.format(*config)]

            times = extract_times_of_interest(root_dir, file_ids, approach, measure_type)

            # get median of end to end
            end_to_end_times = [x[END_TO_END] for x in times]

            model_measurements[model][approach] = median(end_to_end_times)

    return model_measurements


def sh_iteration_plot_times(root_dir, model, approaches, distribution, caching_location, num_models, measure_type):
    model_measurements = {}
    model_measurements[model] = {}
    for approach in approaches:
        config = [distribution, approach, caching_location, model, num_models, measure_type]
        file_id = file_template.format(*config)
        times = extract_times_of_interest(root_dir, file_id, approach, measure_type)
        if approach == BASELINE:
            model_measurements[model][BASELINE] = times[END_TO_END]
        else:
            model_measurements[model][approach] = {}
            for k, v in times[SH_ITERATIONS].items():
                model_measurements[model][approach][k] = v[SH_RANK_ITERATION]

    return model_measurements


def plot_end_to_end_times(data_root_dir, file_template, models, approaches, distribution, data_items, measure_type,
                          plot_save_path):

    plt.rcParams.update({'font.size': 20})
    colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
    colors = ['#bae4bc', '#43a2ca', '#0868ac']

    # Extracting the data
    data = end_to_end_plot_times(
        data_root_dir, file_template, models, approaches, distribution, data_items, measure_type)
    models = list(data.keys())
    methods = list(next(iter(data.values())).keys())
    # Number of models and methods
    n_models = len(models)
    n_methods = len(methods)
    # Creating a bar plot
    bar_width = 0.3
    index = np.arange(n_models)
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Plot each method
    # Create a figure and an axis with a larger width
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, method in enumerate(methods):
        method_values = [data[model][method] for model in models]
        bars = ax.bar(index + i * bar_width, method_values, bar_width, label=method, color=colors[i])

        # Add annotations for shift and mosix
        if method in ['shift', 'mosix']:
            for bar, model in zip(bars, models):
                baseline_value = data[model]['baseline']
                speedup = baseline_value / data[model][method]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{speedup:.2f}x', ha='center',
                        va='bottom')

    # Adding labels and title
    # ax.set_xlabel('Model Architectures')
    ax.set_ylabel('Time in seconds')
    ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([MODEL_NAME_MAPPING[model] for model in models], rotation=15, ha='right')  # Rotate x-axis labels
    # ax.set_xticklabels([MODEL_NAME_MAPPING[model] for model in models])
    ax.tick_params(axis='x')
    ax.legend()
    # Save the plot as SVG and PNG
    plt.tight_layout()
    plot_file_name = f'end_to_end-{distribution}-{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))


def plot_sh_iterations(root_dir, model, approach, distribution, caching_location, num_models, measure_type,
                       plot_save_path):
    data = sh_iteration_plot_times(root_dir, model, approach, distribution, caching_location, num_models, measure_type)
    shift_data = data[model]['shift']
    mosix_data = data[model]['mosix']
    x = list(shift_data.keys())
    shift_values = list(shift_data.values())
    mosix_values = list(mosix_data.values())
    # Number of bars per group
    n_bars = len(x)
    # Baseline value divided by the number of keys
    baseline_value = data[model]['baseline']
    baseline_divided = baseline_value / n_bars
    # Creating a bar plot
    bar_width = 0.35
    index = np.arange(n_bars)
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Plot each group
    bars_shift = ax.bar(index, shift_values, bar_width, label='Shift')
    bars_mosix = ax.bar(index + bar_width, mosix_values, bar_width, label='Mosix')
    # Add a horizontal gray line at the baseline divided value
    ax.axhline(y=baseline_divided, color='gray', linestyle='--', linewidth=1)
    # Adding label to the right of the plot within the frame
    ax.text(n_bars - 2.5, baseline_divided + 10, 'baseline / num iterations', color='gray', ha='left', va='center')
    # Adding labels and title
    ax.set_xlabel('Key')
    ax.set_ylabel('Values')
    ax.set_title(f'{model}-{distribution}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x)
    ax.legend()
    # Save the plot as SVG and PNG
    plt.tight_layout()
    plot_file_name = f'sh_iterations-{distribution}-{caching_location}-{num_models}-{measure_type}-{model}'
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))


if __name__ == '__main__':
    file_template = 'des-gpu-imagenette-synthetic-distribution-{}-approach-{}-cache-CPU-snapshot-{}-models-35-items-{}-level-{}'

    models = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
    approaches = ['baseline', 'shift', 'mosix']
    distributions = [TOP_LAYERS, TWENTY_FIVE_PERCENT, FIFTY_PERCENT]
    measure_type = 'EXECUTION_STEPS'

    for distribution in distributions:
        for data_items in [1000, 2000, 4000, 8000]:
            root_dir = "/Users/nils/Downloads/des-gpu-imagenette-synthetic-snapshots"
            plot_save_path = f'./plots-synthetic-snapshots/{data_items}'
            plot_end_to_end_times(root_dir, file_template, models, approaches, distribution, data_items, measure_type,
                                  plot_save_path)
