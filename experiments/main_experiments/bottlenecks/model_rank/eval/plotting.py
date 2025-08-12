import os
from statistics import median

from experiments.side_experiments.plot_shared.data_transform import aggregate_measurements
from experiments.side_experiments.plot_shared.file_parsing import extract_files_by_name, parse_json_file
from experiments.side_experiments.plot_shared.horizontal_normalized_bars import plot_horizontal_normalized_bar_chart
from global_utils.constants import STATE_DICT_SIZE, LOAD_DATA, DATA_TO_DEVICE, INFERENCE, MODEL_TO_DEVICE, \
    CALC_PROXY_SCORE, STATE_TO_MODEL, LOAD_STATE_DICT, MEASUREMENTS
from global_utils.model_names import RESNET_18, RESNET_152, VIT_L_32, EFF_NET_V2_L

MODEL_NAME_MAPPING = {
    RESNET_18: "ResNet-18",
    RESNET_152: "ResNet-152",
    VIT_L_32: "ViT-L-32",
    EFF_NET_V2_L: "EfficientNetV2-L"
}


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


def group_times(agg_data):
    result = {}
    result['prepare model'] = agg_data['load_state_dict'] + agg_data['model_to_device'] + agg_data[
        'load_state_to_model']
    result['prepare data'] = agg_data['load_data'] + agg_data['data_to_device']
    result['inference'] = agg_data['inference']
    result['proxy score'] = agg_data['calc_proxy_score']

    return result


def get_aggregated_data(root_dir, file_id, agg_func, disk_speed, expected_files=5):
    files = extract_files_by_name(root_dir, [file_id])
    assert len(files) == 5, file_id
    extracted_data = [extract_and_filter(f, disk_speed) for f in files]
    agg_data = aggregate_measurements(extracted_data, agg_func)
    grouped_times = group_times(agg_data)
    return grouped_times


def rename_model_names(data):
    result = {}
    for k, v in data.items():
        new_key = MODEL_NAME_MAPPING[k]
        result[new_key] = v
    return result


def plot_time_dist(root_dir, file_template, model_names, disk_speed, save_path, file_name_prefix=""):
    # . at the end of name is important here to distinguish between dataset types when searching for files
    for model_name in model_names:
        for dataset_type, split, num_items in zip(
                ['imagenette.', 'imagenette.', 'imagenette.', 'imagenette_preprocessed_ssd.'],
                [None, None, None, -3],
                [3 * 32, 1024, 9 * 1024, 1024]
        ):
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
            data = rename_model_names(data)

            plot_horizontal_normalized_bar_chart(data, save_path=save_path,
                                                 file_name=f'{file_name_prefix}normalized-{file_name}',
                                                 ignore=ignore, legend=False)


if __name__ == '__main__':
    root_dir = os.path.abspath('/mount-fs/results/bottleneck-analysis')
    save_path = os.path.abspath('/mount-fs/plots/bottleneck-analysis')
    file_template = 'bottleneck_analysis-model-{}-items-{}-split-{}-dataset_type-{}'

    disk_speed = 200  # in MB/s
    model_names = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
    plot_time_dist(root_dir, file_template, model_names, disk_speed, save_path)
