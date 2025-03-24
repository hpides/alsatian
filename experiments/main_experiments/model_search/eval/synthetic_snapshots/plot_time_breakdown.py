import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.main_experiments.model_search.eval.synthetic_snapshots.plot_synthetic_snapshots import \
    extract_times_of_interest, SUM_OVER_STEPS_DETAILED_NUMS_AGG, regroup_and_rename_times, APPROACH_NAME_MAPPING, \
    BASELINE, SHIFT, MOSIX
from experiments.plot_util import HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE
from global_utils.model_names import RESNET_152, VIT_L_32


def stacked_bar_plot_three_configurations(config_1, config_2, config_3, file_path, file_name, approaches):
    APPROACH_NAME_MAPPING = {
        BASELINE: "B",
        SHIFT: "S",
        MOSIX: "A",
    }

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

    # colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#b30086']
    # colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
    colors = [HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE]

    # Maintain the order of the keys as they appear in the JSON inputs
    categories = (list(config_1.keys()) +
                  [key for key in config_2.keys() if key not in config_1] +
                  [key for key in config_3.keys() if key not in config_1 and key not in config_2])

    if "exec planning" in categories:
        categories.remove("exec planning")

    # for conf in [config_1, config_2, config_3]:
    #     if "exec_planning" in conf:
    #         del conf["exec_planning"]

    # Data preparation
    baseline_values = [config_1.get(category, 0) for category in categories]
    shift_values = [config_2.get(category, 0) for category in categories]
    mosix_values = [config_3.get(category, 0) for category in categories]

    # do not include exec planning because it is so small

    # Combine the values
    values = [baseline_values, shift_values, mosix_values]
    # Transpose the values for stacking
    values = np.array(values).T
    # Plotting
    bar_width = 0.8
    indices = np.arange(len(approaches))
    plt.figure(figsize=(3, 4))
    # Stack the bars
    bottom = np.zeros(len(approaches))
    bars = []
    for i, category in enumerate(categories):
        print(i)
        bar = plt.bar(indices, values[i], bar_width, bottom=bottom, label=category, color=colors[i])
        bars.append(bar)
        bottom += values[i]
    # plt.xlabel(x_label)
    plt.ylabel('Time in seconds')
    # plt.title(title)
    if approaches[0] in APPROACH_NAME_MAPPING:
        plt.xticks(indices, [APPROACH_NAME_MAPPING[x] for x in approaches])
    else:
        plt.xticks(indices, approaches)
    # if title in ["shift", "baseline"]:
    #     y_ticks = list(range(0, 2000, 500))
    #     plt.yticks(y_ticks)
    # Reverse the order of handles and labels for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # legend = plt.legend(handles[::-1], labels[::-1], ncol=3)

    # Save the legend to a separate SVG file
    fig_legend = plt.figure(figsize=(3, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles[::-1], labels[::-1], loc='center', ncol=1)
    ax_legend.axis('off')  # Hide the axes
    legend_file_path = os.path.join(file_path, f'breakdown-legend.svg')
    fig_legend.savefig(legend_file_path, bbox_inches='tight', format='svg')
    legend_file_path = os.path.join(file_path, f'breakdown-legend.pdf')
    fig_legend.savefig(legend_file_path, bbox_inches='tight', format='pdf')
    plt.close(fig_legend)

    # Remove the legend from the original plot
    # legend.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{file_name}.svg'))
    plt.savefig(os.path.join(file_path, f'{file_name}.pdf'))
    plt.savefig(os.path.join(file_path, f'{file_name}.png'))


cache_size_mapping = {
    62000: "64",
    8000: "10",
    3000: "5"
}


def plot_approaches_across_memory_config(root_dir, model, items, device, model_dist, output_path, approaches):
    collected_data = []
    for approach in approaches:
        file_id = f'des-gpu-imagenette-synthetic-distribution-{model_dist}-approach-{approach}-cache-{device}-snapshot-{model}-models-35-items-{items}-level-STEPS_DETAILS'
        detailed_numbers = extract_times_of_interest(root_dir, [file_id], approach, "STEPS_DETAILS")[0][
            SUM_OVER_STEPS_DETAILED_NUMS_AGG]
        detailed_numbers = regroup_and_rename_times(detailed_numbers)
        collected_data.append(detailed_numbers)
    plot_file_name = f'time_breakdown-{model}-{model_dist}-{items}'
    stacked_bar_plot_three_configurations(collected_data[0], collected_data[1], collected_data[2], output_path,
                                          plot_file_name, approaches)


def plot_fixed_approach_changed_config(root_dir, model, items, device, distributions, output_path, approach):
    collected_data = []
    for model_dist in distributions:
        file_id = f'des-gpu-imagenette-synthetic-distribution-{model_dist}-approach-{approach}-cache-{device}-snapshot-{model}-models-35-items-{items}-level-STEPS_DETAILS'
        detailed_numbers = extract_times_of_interest(root_dir, [file_id], approach, "STEPS_DETAILS")[0][
            SUM_OVER_STEPS_DETAILED_NUMS_AGG]
        detailed_numbers = regroup_and_rename_times(detailed_numbers)
        collected_data.append(detailed_numbers)
    plot_file_name = f'time_breakdown-distributions-{model}-{approach}-{items}'
    stacked_bar_plot_three_configurations(collected_data[0], collected_data[1], collected_data[2], output_path,
                                          plot_file_name, ["50", "25", "top"])


if __name__ == '__main__':
    output_path = './plots/breakdown_plots'
    device = "CPU"
    root_dir = os.path.abspath('./results/des-gpu-imagenette-synthetic-snapshots')
    model = RESNET_152
    approaches = ['baseline', 'shift', 'mosix']
    plot_configurations = [
        ["TOP_LAYERS", 2000],
        ["TOP_LAYERS", 8000],
        ["FIFTY_PERCENT", 8000]
    ]

    for model_dist, items in plot_configurations:
        plot_approaches_across_memory_config(root_dir, model, items, device, model_dist, output_path, approaches)

    approach = 'mosix'
    distributions = ["FIFTY_PERCENT", "TWENTY_FIVE_PERCENT", "TOP_LAYERS"]
    items = 8000
    plot_configurations = [
        [distributions, RESNET_152],
        [distributions, VIT_L_32]
    ]
    for distributions, model in plot_configurations:
        plot_fixed_approach_changed_config(root_dir, model, items, device, distributions, output_path, approach)
