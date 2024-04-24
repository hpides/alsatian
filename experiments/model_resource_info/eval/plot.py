from statistics import mean

import matplotlib.pyplot as plt

from experiments.plot_shared.file_parsing import get_raw_data
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES


def plot(data, metric, output_path=None):
    # Extract keys and values from the dictionary
    keys = list(data.keys())
    values = [v[metric] for v in list(data.values())]
    if metric == 'gpu_inf_times':
        values = [mean(v) for v in values]

    # Plot the data as connected dots
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.xlabel('Layers')
    plt.ylabel(metric)
    plt.grid(True)
    if output_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = f'{output_path}.svg'
        plt.savefig(path, format="svg", bbox_inches='tight')
        path = f'{output_path}.png'
        plt.savefig(path, format="png", bbox_inches='tight')

    plt.close()


if __name__ == '__main__':
    root_dir = '/Users/nils/Downloads/model_resource_info'
    model_names = VISION_MODEL_CHOICES
    batch_size = 32
    metrics = ['num_params', 'num_params_mb', 'output_size_mb', 'gpu_inf_times']

    for model_name in model_names:
        _id = f'model_name-{model_name}-batch_size-{batch_size}'

        measurements = get_raw_data(root_dir, [_id], expected_files=1)[MEASUREMENTS]
        for metric in metrics:
            output_path = f'../plots/{metric}-{_id}'
            plot(measurements, metric, output_path)
