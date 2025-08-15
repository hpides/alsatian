import os
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES, VIT_L_32, RESNET_152


def plot_cmp_param_size_inf_time(x_values, param_size, inf_time, ignore, model_name, x_interval):
    FONT_SIZE = 20
    plt.rcParams.update({
        'font.size': FONT_SIZE,  # General font size
        'axes.titlesize': FONT_SIZE,  # Title font size
        'axes.labelsize': FONT_SIZE,  # X and Y label font size
        'xtick.labelsize': FONT_SIZE,  # X tick labels font size
        'ytick.labelsize': FONT_SIZE,  # Y tick labels font size
        'legend.fontsize': FONT_SIZE,  # Legend font size
    })

    x = x_values
    y1 = param_size
    y2 = [x * 1000 for x in inf_time]

    for i in range(ignore[0]):
        x.pop(0)
        y1.pop(0)
        y2.pop(0)

    for i in range(ignore[1]):
        x.pop()
        y1.pop()
        y2.pop()

    x = range(0,len(y1))

    # Function for simple moving average (smoothing)
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    # Apply smoothing using a moving average
    window_size = 5  # Change this to adjust smoothing intensity
    y2_smoothed = moving_average(y2, window_size)

    # Create the figure and first y-axis
    fig, ax1 = plt.subplots(figsize=(4, 2))

    # Plot the first line (y1) as a dotted line on the first y-axis
    y1_color = '#7bccc4'
    ax1.plot(x, y1, 'o', color=y1_color,  alpha=0.5)
    ax1.set_xlabel('Block index')
    ax1.set_ylabel('Num. of params', color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax1.set_ylim(- 0.2 * max(y1), 1.2 * max(y1))

    ax1.set_xticks(list(range(x[0],x[-1]+x_interval, x_interval)))

    # Create the second y-axis, sharing the same x-axis
    ax2 = ax1.twinx()

    y2_color = '#0868ac'
    ax2.plot(x, y2, 'o', color=y2_color,  alpha=0.5)
    ax2.set_ylabel('Inf. time in ms', color=y2_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    ax2.set_ylim(- 0.2 * max(y2), 2 * max(y2))

    # Optional: add titles and a grid
    # plt.title("Line Plot with Two Y-Axes, Noise, and Smoothing (Dots for Noisy Data and Dotted Line for sin(x))")
    # ax1.grid(True)

    path = f'/mount-fs/plots/fig12/param_inf_time_dist_{model_name}'
    plt.savefig(f'{path}.png', format='png', bbox_inches='tight')
    plt.savefig(f'{path}.svg', format='svg', bbox_inches='tight')

    # Show the plot
    plt.show()


def get_keys_values(data, metric):
    keys = list(data.keys())
    values = [v[metric] for v in list(data.values())]
    if metric == 'gpu_inf_times':
        values = [median(v) for v in values]

    return keys, values


if __name__ == '__main__':
    root_dir = os.path.abspath('/mount-fs/results/fig12')
    model_names = VISION_MODEL_CHOICES
    batch_size = 32

    # (RESNET_152, [1, 1])
    for model_name, ignore, x_interval in [(VIT_L_32, [1, 1], 3), (RESNET_152, [0, 3], 10)]:
        _id = f'model_name-{model_name}-batch_size-{batch_size}'
        measurements = get_raw_data(root_dir, [_id], expected_files=1)[MEASUREMENTS]
        keys, inf_times = get_keys_values(measurements, 'gpu_inf_times')
        _, params_mb = get_keys_values(measurements, 'num_params')
        plot_cmp_param_size_inf_time(keys, params_mb, inf_times, ignore, model_name, x_interval)
