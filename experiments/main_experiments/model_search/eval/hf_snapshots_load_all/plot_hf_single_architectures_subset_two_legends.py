import os.path
from statistics import median

import numpy as np
from matplotlib import pyplot as plt

from experiments.main_experiments.model_search.eval.hf_snapshots.plot_hf_combined_snapshot import sum_up_level
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
    SENSE_TIME_DEFORMABLE_DETR: 'def-detr'
}

APPROACH_NAME_MAPPING = {
    BASELINE: "Base",
    SHIFT: "SHiFT",
    MOSIX: "Alsatian",
    "base": "Base",
}


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


def extract_times_of_interest(root_dir, file_ids, approach, measure_type, fallback_path=None,
                              backup_file_ids=None):
    files = []
    for file_id in file_ids:
        # find file
        files += extract_files_by_name(root_dir, [file_id])
        if len(files) == 0:
            for file_id in backup_file_ids:
                files += extract_files_by_name(fallback_path, [file_id])
    assert len(files) >= 1 and len(files) < 4

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


def end_to_end_plot_times(root_dir, file_template, models, approaches, data_items, measure_type,
                          not_aggregated=False, fallback_path=None, backup_file_template=None):
    model_measurements = {}
    for model in models:
        model = model.replace("/", "-", 1)
        model_measurements[model] = {}
        for approach in approaches:
            config = [approach, model, data_items, measure_type]
            file_ids = [file_template.format(*config)]

            backup_file_ids = [backup_file_template.format(*config)]

            times = extract_times_of_interest(root_dir, file_ids, approach, measure_type, fallback_path=fallback_path,
                                              backup_file_ids=backup_file_ids)

            # get median of end to end
            end_to_end_times = [x[END_TO_END] for x in times]

            if not_aggregated:
                model_measurements[model][approach] = end_to_end_times
            else:
                model_measurements[model][approach] = median(end_to_end_times)

    return model_measurements


def _merge_data(data):
    model_names = list(data['baseline'].keys())
    merged_data = {model_name: {} for model_name in model_names}
    for approach, app_data in data.items():
        for model_name in merged_data.keys():
            merged_data[model_name][approach] = app_data[model_name][approach]

    return merged_data


def collect_combined_data(model_subsets, data_root_dirs, fallback_path, file_templates, backup_file_template,
                          measure_type):
    data = {}
    for model_set, models in model_subsets.items():
        data[model_set] = {}
        for approach, data_root_dir in data_root_dirs.items():
            # Extracting the data
            data[model_set][approach] = end_to_end_plot_times(
                data_root_dir, file_templates[approach], models, [approach], data_items, measure_type,
                fallback_path=fallback_path, backup_file_template=backup_file_template)
        data[model_set] = _merge_data(data[model_set])

    return data


def plot_end_to_end_times(data, plot_save_path):
    plt.rcParams.update({'font.size': 24})

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

    _, first_data_item = next(iter(data.items()))
    methods = list(next(iter(first_data_item.values())).keys())
    # Number of models and methods
    n_methods = len(methods)
    # Creating a bar plot
    bar_width = 0.3
    # Create a figure and an axis with a larger width
    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 3]})

    bi = -1
    for data_group, group_data in data.items():
        max_method_value = 0
        bi += 1
        models = list(group_data.keys())
        n_models = len(models)
        bar_index = np.arange(n_models)  # Positions for the bars

        for i, method in enumerate(methods):
            method_values = [group_data[model][method] / 60 for model in models]
            max_method_value = max(max_method_value, max(method_values))
            print(max_method_value)
            bars = axs[bi].bar(bar_index + i * bar_width, method_values, bar_width,
                               label=APPROACH_NAME_MAPPING[method],
                               color=colors[i])
            # Add annotations for shift and mosix
            if method in ['shift', 'mosix']:
                for bar, model in zip(bars, models):
                    baseline_value = group_data[model]['baseline']
                    speedup = baseline_value / group_data[model][method]
                    axs[bi].text(bar.get_x() + bar.get_width() / 2 + 0.04, bar.get_height(), f'{speedup:.1f}x',
                                 ha='center',
                                 va='bottom', rotation=0)

        # Adding labels and title
        axs[bi].set_ylabel('Time in minutes')
        if max_method_value > 5 and max_method_value < 7:
            y_ticks = list(range(0, 8, 2))
            axs[bi].set_yticks(y_ticks)
        elif max_method_value > 10 and max_method_value < 20:
            y_ticks = list(range(0, 20, 5))
            axs[bi].set_yticks(y_ticks)


        # Set x-ticks at the center of each group of bars
        tick_positions = bar_index + bar_width * (n_methods - 1) / 2
        axs[bi].set_xticks(tick_positions)

        # Map and assign labels to the x-ticks
        models_ticks = [MODEL_NAME_MAPPING[model.replace("-", "/", 1)] for model in models]
        axs[bi].set_xticklabels(models_ticks, rotation=15, ha='right')  # Rotate x-axis labels

    # Ensure layout is tight to avoid label overlap
    plt.tight_layout()

    plot_file_name = f'end_to_end-merged_subset_models_{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{data_items}-{plot_file_name}.png'))

    plt.close(fig)


if __name__ == '__main__':
    model_sets = {
        "subset-long": [GOOGLE_VIT_BASE_PATCH16_224_IN21K, FACEBOOK_DETR_RESNET_50],
        "subset-short": [FACEBOOK_DETR_RESNET_101, FACEBOOK_DINOV2_LARGE, MICROSOFT_RESNET_152],
    }

    root_dirs = {
        "baseline": os.path.abspath('./results/des-gpu-imagenette-huggingface-load-full-models'),
        "shift": os.path.abspath('./results/des-gpu-imagenette-huggingface-load-full-models'),
        "mosix": os.path.abspath(
            '../hf_snapshots/results/des-gpu-imagenette-huggingface-single-architecture-search')
    }

    fallback_path = os.path.abspath(
        '../hf_snapshots/results/des-gpu-imagenette-huggingface-single-architecture-search')

    file_templates = {
        "baseline": "des-gpu-imagenette-huggingface-load-full-models#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}",
        "shift": "des-gpu-imagenette-huggingface-load-full-models#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}",
        "mosix": "des-gpu-imagenette-huggingface-single-architecture-search#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}"
    }

    backup_file_template = "des-gpu-imagenette-huggingface-single-architecture-search#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}"

    measure_type = 'EXECUTION_STEPS'
    for data_items in [2000, 8000]:
        data = collect_combined_data(model_sets, root_dirs, fallback_path, file_templates, backup_file_template,
                                     measure_type)

        plot_save_path = os.path.abspath(f'./plots/single_models/{data_items}/subset-merged')
        os.makedirs(plot_save_path, exist_ok=True)
        plot_end_to_end_times(data, plot_save_path)
