import os

import numpy as np
from matplotlib import pyplot as plt

from experiments.model_search.eval.plotting import extract_times_of_interest
from global_utils.constants import END_TO_END
from global_utils.model_names import EFF_NET_V2_L, VIT_L_32, RESNET_152, RESNET_18


def end_to_end_plot_times(root_dir, models, approach, distribution, caching_locations, num_models, measure_type):
    model_measurements = {}
    for model in models:
        model_measurements[model] = {}
        for caching_location in caching_locations:
            config = [distribution, approach, caching_location, model, num_models, measure_type]
            file_id = file_template.format(*config)
            times = extract_times_of_interest(root_dir, file_id, approach, measure_type)
            model_measurements[model][caching_location] = times[END_TO_END]

    return model_measurements


def plot(data_root_dir, models, approaches, distribution, caching_location, num_models, measure_type,
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
    # Create a figure and an axis with a larger width
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.rcParams.update({'font.size': 16})

    for i, method in enumerate(methods):
        method_values = [data[model][method] for model in models]
        bars = ax.bar(index + i * bar_width, method_values, bar_width, label=method)

        # Add annotations for shift and mosix
        if method in ['shift', 'mosix']:
            for bar, model in zip(bars, models):
                baseline_value = data[model]['baseline']
                speedup = baseline_value / data[model][method]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{speedup:.2f}x', ha='center',
                        va='bottom')

    # Adding labels and title
    ax.set_xlabel('Model Architectures')
    ax.set_ylabel('Time in seconds')
    ax.set_xticks(index + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(models, fontsize=14, rotation=45, ha='right')  # Rotate x-axis labels
    ax.tick_params(axis='x', labelsize=18)
    ax.legend()
    # Save the plot as SVG and PNG
    plt.tight_layout()
    plot_file_name = f'end_to_end-{distribution}-{caching_location}-{num_models}-{measure_type}'
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.svg'))
    plt.savefig(os.path.join(plot_save_path, f'{plot_file_name}.png'))


if __name__ == '__main__':
    data_items = 4000
    root_dir = f'/Users/nils/Downloads/imagenette-{data_items}-cpu-gpu'
    file_template = f'des-gpu-imagenette-base-{data_items}' + '-distribution-{}-approach-{}-cache-{}-snapshot-{}-models-{}-level-{}.json'

    # config = ['TOP_LAYERS', 'mosix', 'CPU', 'resnet152', '35', 'EXECUTION_STEPS']
    # file_id = file_template.format(*config)

    models = [RESNET_18, RESNET_152, VIT_L_32, EFF_NET_V2_L]
    approach = 'mosix'
    distributions = ['LAST_ONE_LAYER']
    caching_locations = ['GPU', 'CPU']
    num_models = 35
    measure_type = 'EXECUTION_STEPS'
    plot_save_path = f'plots-compare-gpu-cpu'

    for distribution in distributions:
        plot(root_dir, models, approach, distribution, caching_locations, num_models, measure_type,
             plot_save_path)
