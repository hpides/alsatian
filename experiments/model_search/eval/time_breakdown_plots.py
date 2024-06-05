import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.model_search.eval.plotting import extract_times_of_interest, SUM_OVER_STEPS_DETAILED_NUMS_AGG
from global_utils.model_names import VIT_L_32, RESNET_152


def stacked_bar_plot_three_approaches(baseline, shift, mosix, file_path, file_name):
    # Maintain the order of the keys as they appear in the JSON inputs
    categories = list(baseline.keys()) + [key for key in shift.keys() if key not in baseline] + [key for key in
                                                                                                 mosix.keys() if
                                                                                                 key not in baseline and key not in shift]
    # Data preparation
    baseline_values = [baseline.get(category, 0) for category in categories]
    shift_values = [shift.get(category, 0) for category in categories]
    mosix_values = [mosix.get(category, 0) for category in categories]
    # Combine the values
    values = [baseline_values, shift_values, mosix_values]
    # Transpose the values for stacking
    values = np.array(values).T
    # Plotting
    approaches = ['Baseline', 'Shift', 'Mosix']
    bar_width = 0.5
    indices = np.arange(len(approaches))
    plt.figure(figsize=(10, 8))
    # Stack the bars
    bottom = np.zeros(len(approaches))
    bars = []
    for i, category in enumerate(categories):
        bar = plt.bar(indices, values[i], bar_width, bottom=bottom, label=category)
        bars.append(bar)
        bottom += values[i]
    plt.xlabel('Approaches')
    plt.ylabel('Values')
    plt.title('Stacked Bar Chart of JSON Data')
    plt.xticks(indices, approaches)
    # Reverse the order of handles and labels for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{file_name}.svg'))
    plt.savefig(os.path.join(file_path, f'{file_name}.png'))


if __name__ == '__main__':
    output_path = './time_breakdown'
    model_distribution = "LAST_ONE_LAYER"
    for model in [RESNET_152, VIT_L_32]:
        for data_items, device in [(1000, "GPU"), (2000, "GPU"), (4000, "CPU")]:
            root_dir = f'/Users/nils/Downloads/imagenette-{data_items}-gpu-des-gpu-server'
            collected_data = []
            for approach in ['baseline', 'shift', 'mosix']:
                file_id = f"des-gpu-imagenette-base-{data_items}-distribution-{model_distribution}-approach-{approach}-cache-{device}-snapshot-{model}-models-35-level-STEPS_DETAILS"
                detailed_numbers = extract_times_of_interest(root_dir, file_id, approach, "STEPS_DETAILS")[
                    SUM_OVER_STEPS_DETAILED_NUMS_AGG]
                collected_data.append(detailed_numbers)
            plot_file_name = f'time-distribution-items-{data_items}-distribution-{model_distribution}-snapshot-{model}'
            stacked_bar_plot_three_approaches(collected_data[0], collected_data[1], collected_data[2], output_path,
                                              plot_file_name)
