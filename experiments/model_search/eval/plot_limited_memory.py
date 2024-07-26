import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from experiments.model_search.eval.plotting import extract_times_of_interest, SUM_OVER_STEPS_DETAILED_NUMS_AGG, \
    regroup_and_rename_times
from global_utils.model_names import VIT_L_32


def stacked_bar_plot_three_configurations(config_1, config_2, config_3, file_path, file_name, config_names, x_label,
                                          title):
    FONT_SIZE = 18
    plt.rcParams.update({
        'font.size': FONT_SIZE,  # General font size
        'axes.titlesize': FONT_SIZE,  # Title font size
        'axes.labelsize': FONT_SIZE,  # X and Y label font size
        'xtick.labelsize': FONT_SIZE,  # X tick labels font size
        'ytick.labelsize': FONT_SIZE,  # Y tick labels font size
        'legend.fontsize': FONT_SIZE,  # Legend font size
    })

    colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac', '#b30086']

    # Maintain the order of the keys as they appear in the JSON inputs
    categories = list(config_1.keys()) + [key for key in config_2.keys() if key not in config_1] + [key for key in
                                                                                                    config_3.keys() if
                                                                                                    key not in config_1 and key not in config_2]
    # Data preparation
    baseline_values = [config_1.get(category, 0) for category in categories]
    shift_values = [config_2.get(category, 0) for category in categories]
    mosix_values = [config_3.get(category, 0) for category in categories]
    # Combine the values
    values = [baseline_values, shift_values, mosix_values]
    # Transpose the values for stacking
    values = np.array(values).T
    # Plotting
    approaches = config_names
    bar_width = 0.8
    indices = np.arange(len(approaches))
    plt.figure(figsize=(3, 5))
    # Stack the bars
    bottom = np.zeros(len(approaches))
    bars = []
    for i, category in enumerate(categories):
        print(i)
        bar = plt.bar(indices, values[i], bar_width, bottom=bottom, label=category, color=colors[i])
        bars.append(bar)
        bottom += values[i]
    # plt.xlabel(x_label)
    plt.ylabel('time in seconds')
    plt.xlabel('avail. memory in GB',labelpad=20, x=0.1)
    plt.title(title)
    plt.xticks(indices, approaches)
    # Reverse the order of handles and labels for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # legend = plt.legend(handles[::-1], labels[::-1], ncol=3)

    # Save the legend to a separate SVG file
    fig_legend = plt.figure(figsize=(3, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles[::-1], labels[::-1], loc='center', ncol=3)
    ax_legend.axis('off')  # Hide the axes
    legend_file_path = os.path.join(file_path, f'legend.svg')
    fig_legend.savefig(legend_file_path, bbox_inches='tight', format='svg')
    plt.close(fig_legend)

    # Remove the legend from the original plot
    # legend.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{file_name}.svg'))
    plt.savefig(os.path.join(file_path, f'{file_name}.png'))


cache_size_mapping = {
    62000: "64",
    8000: "10",
    3000: "5"
}


def plot_approaches_across_memory_config(root_dir, model_distribution, model_dist, output_path, approach):
    for model in [VIT_L_32]:
        for data_items, device in [(8000, "CPU")]:
            collected_data = []
            for mosix_cache_size in [62000, 8000, 3000]:
                file_id = f"distribution-{model_dist}-approach-{approach}-cache-{device}-snapshot-{model}-cache_size-{mosix_cache_size}-models-35-level-STEPS_DETAILS"
                detailed_numbers = extract_times_of_interest(root_dir, [file_id], approach, "STEPS_DETAILS")[
                    SUM_OVER_STEPS_DETAILED_NUMS_AGG]
                detailed_numbers = regroup_and_rename_times(detailed_numbers)
                collected_data.append(detailed_numbers)
            plot_file_name = f'approaches_across_memory_config-{approach}-{model}-{model_distribution}'
            stacked_bar_plot_three_configurations(collected_data[0], collected_data[1], collected_data[2], output_path,
                                                  plot_file_name, list(cache_size_mapping.values()), "available memory",
                                                  approach)




if __name__ == '__main__':
    output_path = './plots-limited-memory'
    model_distribution = "FIFTY_PERCENT"
    root_dir = f'/Users/nils/Downloads/limited-memory-exps'
    model_dist = "FIFTY_PERCENT"

    for approach in ['mosix','baseline', 'shift']:
        plot_approaches_across_memory_config(root_dir, model_distribution, model_dist, output_path, approach)

