import random

from experiments.side_experiments.plot_shared.file_parsing import get_raw_data
from global_utils.constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES
from model_search.approaches.shift import get_data_ranges


def decision(probability):
    return int(random.random() <= probability)


def format_number(number):
    # Remove the decimal part and convert to integer
    integer_number = int(number)

    # Convert to string and pad with leading zeros to ensure it's 10 characters long
    formatted_number = f"{integer_number:010d}"

    # Replace leading zeros with spaces
    formatted_number_with_spaces = formatted_number.replace('0', ' ', 10 - len(str(integer_number)))

    # Insert spaces every three digits from the right
    parts = []
    while formatted_number_with_spaces:
        parts.append(formatted_number_with_spaces[-3:])
        formatted_number_with_spaces = formatted_number_with_spaces[:-3]

    formatted_number_with_spaces = ' '.join(reversed(parts))

    return formatted_number_with_spaces


def get_max_linear_path_size_mb(model_name, num_items, percent_ignore=0.0, branch_probability=1.0):
    root_dir = '/Users/nils/Downloads/model_resource_info'
    batch_size = 32
    _id = f'model_name-{model_name}-batch_size-{batch_size}'
    measurements = get_raw_data(root_dir, [_id], expected_files=1)[MEASUREMENTS]

    parameter_size = 0
    intermediate_size = 0

    keys = list(measurements.keys())
    for k in keys:
        parameter_size += measurements[k]['num_params_mb']

    keys = keys[int(len(keys) * percent_ignore):]
    for k in keys:
        intermediate_size += measurements[k]['output_size_mb'] * decision(branch_probability)

    intermediate_size = (intermediate_size / batch_size) * num_items

    combined = parameter_size + intermediate_size

    return {
        # 'num_params_mb': parameter_size,
        # 'output_size_mb': intermediate_size,
        'combined': format_number(combined)
    }


def print_size_overview(model_names, num_items, percent_ignore, branch_probability=1.0):
    print()
    print(f"num_items:{num_items}")
    print(f"percent_ignore:{percent_ignore}")
    print(f"branch_probability:{branch_probability}")
    for model_name in model_names:
        res = get_max_linear_path_size_mb(model_name, num_items, percent_ignore=percent_ignore, branch_probability=branch_probability)
        if len(model_name) > 9:
            print(model_name, '\t', res['combined'])
        else:
            print(model_name, '\t\t', res['combined'])


if __name__ == '__main__':
    random.seed(42)
    model_names = VISION_MODEL_CHOICES

    ranges = get_data_ranges(100, 10050)
    num_items = [range[1] - range[0] for range in ranges]
    print(ranges)
    print(num_items)

    num_items = 5000
    percent_ignore = 0
    print_size_overview(model_names, num_items, percent_ignore)

    ranges = get_data_ranges(100, 20000)
    num_items = [range[1] - range[0] for range in ranges]
    print(ranges)
    print(num_items)

    num_items = 10100
    percent_ignore = 0
    print_size_overview(model_names, num_items, percent_ignore)

    num_items = 5000
    percent_ignore = 0.5
    print_size_overview(model_names, num_items, percent_ignore)

    num_items = 5000
    percent_ignore = 0.75
    print_size_overview(model_names, num_items, percent_ignore)

    ranges = get_data_ranges(100, 10050)
    num_items = [range[1] - range[0] for range in ranges]

    for n in num_items:
        percent_ignore = 0.5
        print_size_overview(model_names, n, percent_ignore)


