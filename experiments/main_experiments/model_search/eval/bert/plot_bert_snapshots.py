import os.path
from statistics import median

from matplotlib import pyplot as plt

from experiments.main_experiments.model_search.eval.synthetic_snapshots.plot_synthetic_snapshots import \
    APPROACH_NAME_MAPPING
from experiments.main_experiments.snapshots.synthetic.generate import TOP_LAYERS, TWENTY_FIVE_PERCENT, FIFTY_PERCENT
from experiments.side_experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from global_utils.constants import GEN_EXEC_PLAN, GET_COMPOSED_MODEL, MODEL_TO_DEVICE, LOAD_DATA, DATA_TO_DEVICE, \
    CALC_PROXY_SCORE, LOAD_STATE_DICT, INIT_MODEL, STATE_TO_MODEL, INFERENCE, END_TO_END, DETAILED_TIMES, \
    EXEC_STEP_MEASUREMENTS, MEASUREMENTS
from global_utils.model_names import RESNET_18, RESNET_152, VIT_L_32, EFF_NET_V2_L, BERT

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

    return model_measurements[models[0]]


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

    data = end_to_end_plot_times(data_root_dir, file_template, models, approaches, distribution,
                                 data_items, measure_type)

    # Change the names of the approaches to 'base', 'shift', 'mosix'
    ordered_approaches = ['base', 'shift', 'mosix']
    # Map the original approach names to the new ones
    original_approaches = ['baseline', 'shift', 'mosix']
    times = [data[approach] for approach in original_approaches]

    # Create the bar plot
    plt.figure(figsize=(7, 9))
    bars = plt.bar([APPROACH_NAME_MAPPING[x] for x in ordered_approaches], times, color=colors)

    # Add annotations for shift and mosix
    for bar, approach, original_approach in zip(bars, ordered_approaches, original_approaches):
        if approach in ['shift', 'mosix']:
            baseline_value = data['baseline']
            speedup = baseline_value / data[original_approach]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{speedup:.2f}x', ha='center', va='bottom')

    # plt.xlabel('Approach')
    plt.ylabel('Time in sec', labelpad=20)

    # Rotate the x-ticks by 45 degrees
    plt.xticks(rotation=45)

    # Add a legend
    # plt.legend(title='Approach')

    # Adjust the layout to ensure nothing is cut off
    plt.tight_layout()

    # Save the plot as SVG and PNG
    plot_file_name = f'bert-end_to_end-{distribution}-{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))


if __name__ == '__main__':
    file_template = 'des-gpu-bert-synthetic-distribution-{}-approach-{}-cache-CPU-snapshot-{}-models-35-items-{}-level-{}'

    models = [BERT]
    approaches = ['baseline', 'shift', 'mosix']
    distributions = [TOP_LAYERS, TWENTY_FIVE_PERCENT, FIFTY_PERCENT]

    for distribution in distributions:
        for data_items in [2000, 8000]:
            if data_items == 8000:
                measure_type = 'STEPS_DETAILS'
            else:
                measure_type = 'EXECUTION_STEPS'
            root_dir = os.path.abspath("./results/des-gpu-bert-synthetic/")
            plot_save_path = f'./plots/{data_items}'
            plot_end_to_end_times(root_dir, file_template, models, approaches, distribution, data_items, measure_type,
                                  plot_save_path)
