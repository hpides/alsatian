import matplotlib.pyplot as plt

from experiments.side_experiments.opt_parameters.data_loading.eval.plot_shared import get_file_id
from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import LOAD_DATA
from global_utils.global_constants import MEASUREMENTS


def collect_batch_size_data(root_dir, num_workers, sleep, dataset_type, batch_sizes, last_batch=None):
    load_times = {}
    for batch_size in batch_sizes:
        file_id = get_file_id(batch_size, dataset_type, num_workers, sleep)
        measurements = get_raw_data(root_dir, [file_id], expected_files=1)[MEASUREMENTS]
        load_times[batch_size] = measurements[LOAD_DATA]
        if last_batch is not None:
            load_times[batch_size] = load_times[batch_size][:last_batch]

    return load_times


def multi_line_plot(data, output_path=None):
    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height as needed

    # Extracting x-values (assuming same for all lines)
    x = range(1, len(next(iter(data.values()))) + 1)

    # Plotting the lines
    for label, y in data.items():
        plt.plot(x, y, label=label)

    # Adding labels and removing title
    plt.xlabel('Batch number')
    plt.ylabel('Time in seconds')

    # Reversing the order of legend labels
    legend_labels = list(data.keys())[::-1]

    # Creating a reversed legend
    handles, _ = plt.gca().get_legend_handles_labels()
    reversed_handles = [handles[list(data.keys()).index(label)] for label in legend_labels]

    # Adding legend outside the plot
    plt.legend(reversed_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if output_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = f'{output_path}.svg'
        plt.savefig(path, format="svg", bbox_inches='tight')
        path = f'{output_path}.png'
        plt.savefig(path, format="png", bbox_inches='tight')

    plt.close()


if __name__ == '__main__':
    root_dir = '/Users/nils/Downloads/data-loading-exp'
    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 32, 48, 64]
    sleeps = [None, 2]
    dataset_types = ['imagenette', 'preprocessed_ssd']

    for num_workers in nums_workers:
        for sleep in sleeps:
            for dataset_type in dataset_types:
                _id = f'workers-{num_workers}-sleep-{sleep}-data-{dataset_type}'
                output_path = f'../plots/batch_size_impact/{_id}'
                data = collect_batch_size_data(root_dir, num_workers, sleep, dataset_type, batch_sizes, last_batch=9)
                multi_line_plot(data, output_path)
