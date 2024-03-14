from statistics import median

from experiments.plot_shared.data_transform import aggregate_measurements
from experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from experiments.plot_shared.horizontal_normalized_bars import plot_horizontal_normalized_bar_chart
from experiments.plot_shared.plotting_util import plot_stacked_bar_chart
from global_utils.constants import STATE_DICT_SIZE, LOAD_DATA, DATA_TO_DEVICE, INFERENCE, MODEL_TO_DEVICE, \
    CALC_PROXY_SCORE, STATE_TO_MODEL, LOAD_STATE_DICT
from global_utils.global_constants import MEASUREMENTS
from global_utils.model_names import VISION_MODEL_CHOICES, RESNET_18, RESNET_152, VIT_L_32, EFF_NET_V2_L


def extract_and_filter(file, disk_speed):
    result = {}
    data = parse_json_file(file)
    measurements = data[MEASUREMENTS]

    result[LOAD_STATE_DICT] = measurements[STATE_DICT_SIZE] / disk_speed
    result[MODEL_TO_DEVICE] = measurements[MODEL_TO_DEVICE]
    result[STATE_TO_MODEL] = measurements[STATE_TO_MODEL]

    result[LOAD_DATA] = sum(measurements[LOAD_DATA])
    result[DATA_TO_DEVICE] = sum(measurements[DATA_TO_DEVICE])
    result[INFERENCE] = sum(measurements[INFERENCE])

    result[CALC_PROXY_SCORE] = measurements[CALC_PROXY_SCORE]

    return result


def get_aggregated_data(root_dir, file_id, agg_func, disk_speed, expected_files=5):
    files = extract_files_by_name(root_dir, [file_id])
    assert len(files) == 5, file_id
    extracted_data = [extract_and_filter(f, disk_speed) for f in files]
    agg_data = aggregate_measurements(extracted_data, agg_func)
    return agg_data


def plot_time_dist(root_dir, file_template, model_names, disk_speed, file_name_prefix=""):
    # . at the end of name is important here to distinguish between dataset types when searching for files
    for dataset_type in ['imagenette.', 'imagenette_preprocessed_ssd.']:
        for split in [str(x) for x in [None, -1, -3, 25, 50, 75]]:
            for num_items in [3 * 32, 1024, 9 * 1024]:
                if split == 'None' or dataset_type == 'imagenette_preprocessed_ssd.':
                    data = {}
                    for model_name in model_names:
                        # example_config = ['resnet152', '100', '50', 'imagenette']
                        config = [model_name, num_items, split, dataset_type]
                        file_id = file_template.format(*config)

                        data[model_name] = get_aggregated_data(root_dir, file_id, median, disk_speed)

                    # ignore = [MODEL_TO_DEVICE, STATE_TO_MODEL, DATA_TO_DEVICE]
                    ignore = []
                    file_name = f'bottleneck_analysis-items-{num_items}-split-{split}-data-{dataset_type}'.replace('.',
                                                                                                                   '')
                    plot_horizontal_normalized_bar_chart(data, save_path='./plots', file_name=f'{file_name_prefix}normalized-{file_name}',
                                                         ignore=ignore)
                    plot_stacked_bar_chart(data, save_path='./plots', file_name=f'{file_name_prefix}stacked-{file_name}')


if __name__ == '__main__':
    root_dir = '/Users/nils/Downloads/bottleneck-analysis'
    file_template = 'bottleneck_analysis-model-{}-items-{}-split-{}-dataset_type-{}'
    disk_speed = 200  # in MB/s

    model_names = VISION_MODEL_CHOICES.copy()

    plot_time_dist(root_dir, file_template, model_names, disk_speed)

    model_names = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
    plot_time_dist(root_dir, file_template, model_names, disk_speed, 'PART-')
