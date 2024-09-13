import json
from statistics import mean

from experiments.micro_benchmarks.layer_io_vs_compute_bound.plot import plot
from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES


def get_input_sizes(initial_input_size_mb, measurements):
    result = {}
    # input size of i is output size of i-1
    previous_output_size = initial_input_size_mb
    for i in measurements.keys():
        result[i] = previous_output_size
        previous_output_size = measurements[i]['output_size_mb']
    return result


def get_load_times(assumed_read_speed, input_sizes):
    result = {}
    for k, v in input_sizes.items():
        result[k] = v / assumed_read_speed
    return result


def get_layers_until_io_bound(load_times, measurements):
    result = {}
    keys = list(load_times.keys())
    for i in range(len(keys)):
        time_budget = load_times[keys[i]]
        j = i
        current_layer_i = keys[j]
        result[current_layer_i] = []
        while time_budget >= mean(measurements[keys[j]]['gpu_inf_times']):
            time_budget -= mean(measurements[keys[j]]['gpu_inf_times'])
            result[current_layer_i].append(keys[j])
            j += 1
            if j >= len(keys):
                result[current_layer_i].append(f'remaining: {time_budget}s of initial budget: {load_times[keys[i]]}s')
                break

    return result


def generate_count_metric(data, model_names):
    result = {}
    for model_name in model_names:
        result[model_name] = {}
        for k, v in data[model_name].items():
            if len(v) > 0 and "remaining" in v[-1]:
                result[model_name][k] = len(list(data[model_name].keys()))
            else:
                result[model_name][k] = len(v)
    return result


if __name__ == '__main__':
    root_dir = '/Users/nils/Downloads/model_resource_info'
    batch_size = 32
    initial_input_size_mb = (batch_size * 3 * 224 * 224 * 4) / 1000000
    assumed_read_speed = 1000  # in MB/s

    result = {}
    for model_name in VISION_MODEL_CHOICES:
        _id = f'model_name-{model_name}-batch_size-{batch_size}'
        measurements = get_raw_data(root_dir, [_id], expected_files=1)[MEASUREMENTS]

        input_sizes = get_input_sizes(initial_input_size_mb, measurements)
        load_times = get_load_times(assumed_read_speed, input_sizes)

        result[model_name] = get_layers_until_io_bound(load_times, measurements)

    with open('./result-layer-ids.json', 'w') as json_file:
        json.dump(result, json_file)

    counts = generate_count_metric(result, VISION_MODEL_CHOICES)

    with open('./result-layer-counts.json', 'w') as json_file:
        json.dump(counts, json_file)

    for model_name in VISION_MODEL_CHOICES:
        data = counts[model_name]
        plot(data, f'./plots/{model_name}-number-compute-bound-layers')


    print('test')
