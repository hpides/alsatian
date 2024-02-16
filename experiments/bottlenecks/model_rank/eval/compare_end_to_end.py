import json

from global_utils.constants import MODEL_TO_DEVICE, STATE_TO_MODEL, CALC_PROXY_SCORE, LOAD_DATA, DATA_TO_DEVICE, \
    INFERENCE


def parse_json_file_to_dict(file_path):
    try:
        with open(file_path, 'r') as file:
            # Load JSON data from file into a Python dictionary
            data_dict = json.load(file)
            return data_dict
    except FileNotFoundError:
        print("File not found:", file_path)
        return None
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None


def compare_end_to_end(json_file_path):
    parsed_dict = parse_json_file_to_dict(json_file_path)

    measurements = parsed_dict['measurements']
    end_to_end_time = measurements['end_to_end_time']
    sum_times = 0
    sum_times += measurements[MODEL_TO_DEVICE]
    sum_times += measurements[STATE_TO_MODEL]
    sum_times += measurements[CALC_PROXY_SCORE]
    sum_times += sum(measurements[LOAD_DATA][10:])
    sum_times += sum(measurements[DATA_TO_DEVICE][10:])
    sum_times += sum(measurements[INFERENCE][10:])

    print(f'end-to-end\t: {end_to_end_time * 10 ** -9:.3f} s ({end_to_end_time * 10 ** -6:.3f} ms)')
    print(f'sum of times: {sum_times * 10 ** -9:.3f} s ({sum_times * 10 ** -6:.3f} ms)')
    abs_diff = abs(end_to_end_time - sum_times)
    print(f'Abs difference: {abs_diff * 10 ** -9:.3f} s ({abs_diff * 10 ** -6:.3f} ms)')
    print(f'Per difference: {calculate_percentage_difference(sum_times, end_to_end_time):.3f} %')


def calculate_percentage_difference(old_value, new_value):
    percentage_difference = ((new_value - old_value) / old_value) * 100
    return percentage_difference


if __name__ == '__main__':
    for file in [
        '/Users/nils/Downloads/consistency/2024-02-16-12:13:03#score_model_exp_section_debug-des-consistent-results-w1.json',
        '/Users/nils/Downloads/consistency/2024-02-16-12:24:59#score_model_exp_section_debug-des-consistent-results-w2.json',
        '/Users/nils/Downloads/consistency/2024-02-16-12:29:36#score_model_exp_section_debug-des-consistent-results-w4.json',
        '/Users/nils/Downloads/consistency/2024-02-16-12:33:15#score_model_exp_section_debug-des-consistent-results-w8.json'
    ]:
        compare_end_to_end(file)
        print()
