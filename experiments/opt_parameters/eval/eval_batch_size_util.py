import json
from statistics import median

from global_utils.constants import LOAD_DATA, DATA_TO_DEVICE, INFERENCE, END_TO_END
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES


def read_json(_file):
    with open(_file, 'r') as file:
        data = json.load(file)
    return data


def aggregate_data(_data):
    batch_sizes = [str(x) for x in [32, 128, 512, 1024]]
    model_names = VISION_MODEL_CHOICES

    for model_name in model_names:
        for batch_size in batch_sizes:
            sum_end_to_end = 0
            for metric in [LOAD_DATA, DATA_TO_DEVICE, INFERENCE]:
                _times = _data[MEASUREMENTS][model_name][batch_size][metric]
                sum_times = sum(_times)
                median_times = median(_times)
                _data[MEASUREMENTS][model_name][batch_size][f'sum_{metric}'] = sum_times
                _data[MEASUREMENTS][model_name][batch_size][f'median_{metric}'] = median_times
                sum_end_to_end += sum_times
            _data[MEASUREMENTS][model_name][batch_size][f'sum_{END_TO_END}'] = sum_end_to_end
