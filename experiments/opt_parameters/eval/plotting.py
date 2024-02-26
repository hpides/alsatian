import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.opt_parameters.eval.eval_batch_size_util import read_json, aggregate_data
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES


def plot_end_to_end_batch_size(data, file_path):
    # Extracting data for plotting
    X = VISION_MODEL_CHOICES
    subgroups = [str(x) for x in [32, 128, 512, 1024]]
    values = np.array([[data[MEASUREMENTS][key][subgroup]["end_to_end"] for subgroup in subgroups] for key in X])

    # Plotting
    bar_width = 0.2
    X_axis = np.arange(len(X))

    # Adjusting figure size
    plt.figure(figsize=(10, 6))

    for i, subgroup in enumerate(subgroups):
        plt.bar(X_axis + (i - 1.5) * bar_width, values[:, i], bar_width, label=subgroup)

    plt.xticks(X_axis, X, rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.xlabel("Models")
    plt.ylabel("End to End Time")
    plt.legend(title="Batch Sizes")
    plt.tight_layout()  # Adjust layout to prevent overlapping labels

    plt.savefig(f'{file_path}.png')
    plt.savefig(f'{file_path}.svg', format='svg')

if __name__ == '__main__':
    des_gpu_vision_measurements = '../results/batch_size_impact/2024-02-20-13:30:29#batch_size_impact.json'
    data = read_json(des_gpu_vision_measurements)
    aggregate_data(data)
    out = os.path.join('plots', 'des_gpu_vision_measurements')
    plot_end_to_end_batch_size(data, out)