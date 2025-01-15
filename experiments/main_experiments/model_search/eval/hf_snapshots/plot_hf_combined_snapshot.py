import os.path
from statistics import median

import numpy as np
from matplotlib import pyplot as plt

from experiments.main_experiments.snapshots.hugging_face.init_hf_models import *
from experiments.side_experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from global_utils.constants import GEN_EXEC_PLAN, GET_COMPOSED_MODEL, MODEL_TO_DEVICE, LOAD_DATA, DATA_TO_DEVICE, \
    CALC_PROXY_SCORE, LOAD_STATE_DICT, INIT_MODEL, STATE_TO_MODEL, INFERENCE, END_TO_END, DETAILED_TIMES, \
    EXEC_STEP_MEASUREMENTS, MEASUREMENTS

MOSIX = "mosix"

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

SHIFT = "shift"

#            MICROSOFT_TABLE_TRANSFORMER_DETECTION, FACEBOOK_DETR_RESNET_101, FACEBOOK_DETR_RESNET_50_DC5,
#            SENSE_TIME_DEFORMABLE_DETR], 14)

MODEL_NAME_MAPPING = {
    GOOGLE_VIT_BASE_PATCH16_224_IN21K: 'ViT-base',
    FACEBOOK_DETR_RESNET_50: 'detr-50',
    CONDITIONAL_DETR_RESNET_50: 'c-detr-50',
    FACEBOOK_DINOV2_BASE: 'dino2-b',
    FACEBOOK_DINOV2_LARGE: 'dino2-l',
    MICROSOFT_RESNET_18: 'm-res-18',
    MICROSOFT_RESNET_152: 'm-res-152',
    MICROSOFT_TABLE_STRUCTURE_RECOGNITION: 'tab-rec',
    MICROSOFT_TABLE_TRANSFORMER_DETECTION: 'tab-det',
    FACEBOOK_DETR_RESNET_101: 'detr-101',
    FACEBOOK_DETR_RESNET_50_DC5: 'detr-50-dc',
    SENSE_TIME_DEFORMABLE_DETR: 'def-detr',
    'combined': 'combined'
}

APPROACH_NAME_MAPPING = {
    BASELINE: "Base",
    SHIFT: "SHiFT",
    MOSIX: "Alsatian",
    "base": "Base",
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
    assert len(files) >= 1

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


def end_to_end_plot_times(root_dir, file_template, models, approaches, data_items, measure_type,
                          not_aggregated=False):
    model_measurements = {}
    for model in models:
        model_measurements[model] = {}
        for approach in approaches:
            config = [approach, model, data_items, measure_type]
            file_ids = [file_template.format(*config)]

            times = extract_times_of_interest(root_dir, file_ids, approach, measure_type)

            # get median of end to end
            end_to_end_times = [x[END_TO_END] for x in times]

            if not_aggregated:
                model_measurements[model][approach] = end_to_end_times
            else:
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


def has_first_decimal_zero(number):
    # Convert the number to string and split at the decimal point
    number_str = str(number)

    # Check if there's a decimal part
    if '.' in number_str:
        decimal_part = number_str.split('.')[1]

        # Check if the first digit in the decimal part is '0'
        return decimal_part[0] == '0'

    # If no decimal part exists, return False
    return False


def plot_end_to_end_times(data_root_dir, file_template, models, approaches, data_items, measure_type,
                          plot_save_path, plot_width):
    plt.rcParams.update({'font.size': 46})

    plt.rcParams.update({'text.usetex': True
                            , 'pgf.rcfonts': False
                            , 'text.latex.preamble': r"""\usepackage{iftex}
                                                  \ifxetex
                                                      \usepackage[libertine]{newtxmath}
                                                      \usepackage[tt=false]{libertine}
                                                      \setmonofont[StylisticSet=3]{inconsolata}
                                                  \else
                                                      \RequirePackage[tt=false, type1=true]{libertine}
                                                  \fi"""
                         })

    colors = ['#bae4bc', '#43a2ca', '#0868ac']

    # Extracting the data
    data = end_to_end_plot_times(
        data_root_dir, file_template, models, approaches, data_items, measure_type)
    data = data["combined"]

    # Change the names of the approaches to 'base', 'shift', 'mosix'
    ordered_approaches = ['base', 'shift', 'mosix']
    # Map the original approach names to the new ones
    original_approaches = ['baseline', 'shift', 'mosix']
    times = [data[approach] /60 for approach in original_approaches]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(7, 9))
    bars = plt.bar([APPROACH_NAME_MAPPING[x] for x in ordered_approaches], times, color=colors)

    # Add annotations for shift and mosix
    for bar, approach, original_approach in zip(bars, ordered_approaches, original_approaches):
        if approach in ['shift', 'mosix']:
            baseline_value = data['baseline']
            speedup = baseline_value / data[original_approach]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{speedup:.1f}x', ha='center',
                    va='bottom', rotation=0)

    if data['baseline'] < data['shift']:
        plt.ylim(0, 90)

    plt.ylabel('Time in minutes', labelpad=20)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot without legend
    plt.tight_layout()
    plot_file_name = f'end_to_end-{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.png'))

    plt.close(fig)


def plot_end_to_end_times_error(data_root_dir, file_template, models, approaches, distribution, data_items,
                                measure_type,
                                plot_save_path):
    plt.rcParams.update({'font.size': 20})

    colors = ['#bae4bc', '#43a2ca', '#0868ac']

    # Extracting the data
    data = end_to_end_plot_times(
        data_root_dir, file_template, models, approaches, distribution, data_items, measure_type, not_aggregated=True)

    models = list(data.keys())
    methods = list(next(iter(data.values())).keys())

    # Number of models and methods
    n_models = len(models)
    n_methods = len(methods)

    # Creating a bar plot
    bar_width = 0.3
    index = np.arange(n_models)

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, method in enumerate(methods):
        # Get the list of values for each model and method
        method_values = [data[model][method] for model in models]

        # Calculate the median and the 2.5th and 97.5th percentiles
        medians = [np.median(values) for values in method_values]
        percentiles_2_5 = [np.percentile(values, 2.5) for values in method_values]
        percentiles_97_5 = [np.percentile(values, 97.5) for values in method_values]

        # Calculate the error bars: the difference between the median and the 2.5th/97.5th percentiles
        lower_errors = [median - p2_5 for median, p2_5 in zip(medians, percentiles_2_5)]
        upper_errors = [p97_5 - median for median, p97_5 in zip(medians, percentiles_97_5)]
        error_bars = [lower_errors, upper_errors]

        # Plot the bars with error bars
        bars = ax.bar(index + i * bar_width, medians, bar_width, label=method, color=colors[i],
                      yerr=error_bars, capsize=5)

        # Add annotations for shift and mosix speedups
        if method in ['shift', 'mosix']:
            for bar, model in zip(bars, models):
                baseline_value = np.median(data[model]['baseline'])
                speedup = baseline_value / np.median(data[model][method])
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{speedup:.2f}x', ha='center',
                        va='bottom')

        # Adding labels and title
        ax.set_ylabel('Time in seconds')
        ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels([MODEL_NAME_MAPPING[model] for model in models], rotation=15, ha='right')
        ax.tick_params(axis='x')
        ax.legend()

        # Save the plot as SVG and PNG
        plt.tight_layout()
        plot_file_name = f'percentile-end_to_end-{distribution}-{measure_type}'
        plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
        plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))

def plot_end_to_end_times_with_error(
    root_dir, file_template, models, approaches, data_items, measure_type, plot_save_path, plot_width
):
    plt.rcParams.update({'font.size': 24})
    plt.rcParams.update({'text.usetex': True, 'pgf.rcfonts': False,
                         'text.latex.preamble': r"""\usepackage{iftex}
                          \ifxetex
                              \usepackage[libertine]{newtxmath}
                              \usepackage[tt=false]{libertine}
                              \setmonofont[StylisticSet=3]{inconsolata}
                          \else
                              \RequirePackage[tt=false, type1=true]{libertine}
                          \fi"""
                        })

    colors = ['#bae4bc', '#43a2ca', '#0868ac']

    # Extracting the data
    data = end_to_end_plot_times(
        root_dir, file_template, models, approaches, data_items, measure_type, not_aggregated=True
    )
    models = list(data.keys())
    methods = list(next(iter(data.values())).keys())
    n_models = len(models)
    n_methods = len(methods)

    # Creating a bar plot
    bar_width = 0.3
    index = np.arange(n_models)
    fig, ax = plt.subplots(figsize=(plot_width, 4))

    max_method_value = 0
    for i, method in enumerate(methods):
        method_values = [data[model][method] for model in models]
        medians = [np.median(values) for values in method_values]
        percentiles_2_5 = [np.percentile(values, 2.5) for values in method_values]
        percentiles_97_5 = [np.percentile(values, 97.5) for values in method_values]
        lower_errors = [median - p2_5 for median, p2_5 in zip(medians, percentiles_2_5)]
        upper_errors = [p97_5 - median for median, p97_5 in zip(medians, percentiles_97_5)]
        error_bars = [lower_errors, upper_errors]

        max_method_value = max(max_method_value, max(medians))

        bars = ax.bar(
            index + i * bar_width,
            [m / 60 for m in medians],  # Convert medians to minutes
            bar_width,
            label=APPROACH_NAME_MAPPING[method],
            color=colors[i],
            yerr=np.array(error_bars) / 60,  # Convert errors to minutes
            capsize=5
        )

        # Add annotations for speedup
        if method in ['shift', 'mosix']:
            for bar, model in zip(bars, models):
                baseline_value = np.median(data[model]['baseline'])
                speedup = baseline_value / np.median(data[model][method])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{speedup:.1f}x',
                    ha='center',
                    va='bottom',
                    rotation=0
                )

    ax.set_ylabel('Time in minutes')
    ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([MODEL_NAME_MAPPING[model.replace("-", "/", 1)] for model in models], rotation=15, ha='right')
    ax.tick_params(axis='x')

    if max_method_value > 100:
        y_ticks = list(range(0, int(max_method_value / 60) + 5, 50))
    else:
        y_ticks = list(range(0, int(max_method_value / 60) + 5, 20))
    ax.set_yticks(y_ticks)

    plt.tight_layout()
    plot_file_name = f'end_to_end_error-{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.png'))

    # Extract legend
    fig_legend = plt.figure(figsize=(8, 2))
    legend = fig_legend.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3, frameon=False)
    fig_legend.tight_layout()

    legend_file_name = 'legend'
    fig_legend.savefig(os.path.join(plot_save_path, f'{legend_file_name}.svg'))
    fig_legend.savefig(os.path.join(plot_save_path, f'{legend_file_name}.png'))

    plt.close(fig_legend)
    plt.close(fig)


if __name__ == '__main__':

    file_template = 'des-gpu-imagenette-huggingface-all-hf-architecture-search#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}'

    models = ['combined']
    approaches = ['baseline', 'shift', 'mosix']
    measure_type = 'EXECUTION_STEPS'

    for data_items in [2000, 8000]:
        root_dir = os.path.abspath('./results/des-gpu-imagenette-huggingface-combined-architecture-search')
        plot_save_path = os.path.abspath(f'./plots/combined_models/{data_items}/')
        os.makedirs(plot_save_path, exist_ok=True)
        plot_end_to_end_times(root_dir, file_template, models, approaches, data_items, measure_type, plot_save_path,
                              4)
        plot_end_to_end_times_with_error(root_dir, file_template, models, approaches, data_items, measure_type, plot_save_path,
                              4)
