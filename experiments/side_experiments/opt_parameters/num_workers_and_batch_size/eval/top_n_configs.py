from experiments.side_experiments.opt_parameters.num_workers_and_batch_size.eval.plotting import get_metric_numbers_one_model
from global_utils.constants import END_TO_END
from global_utils.model_names import VISION_MODEL_CHOICES

NUM_WORKERS = 'num_workers'

BATCH_SIZE = 'batch_size'

PREPROCESSED_SSD = 'preprocessed_ssd'

IMAGENETTE = 'imagenette'


def transform_data(data):
    transformed_data = {}
    for key, inner_dict in data.items():
        transformed_inner_dict = {}
        for sub_key, sub_inner_dict in inner_dict.items():
            transformed_inner_dict[sub_key] = sub_inner_dict['end_to_end']
        transformed_data[key] = transformed_inner_dict
    return transformed_data


def create_inverted_index(data):
    inverted_index = {}
    for key, inner_dict in data.items():
        for sub_key, value in inner_dict.items():
            inverted_index[value] = {BATCH_SIZE: key, NUM_WORKERS: sub_key}
    return inverted_index


def get_top_n_lowest_values(inverted_index, n):
    sorted_values = sorted(inverted_index.keys())
    top_n_values = sorted_values[:n]
    return [(value, inverted_index[value]) for value in top_n_values]


def sort_dict_by_values(input_dict):
    sorted_dict = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict


def top_3_confs():
    global model_name, data, key
    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 12, 16, 32, 48, 64]
    model_names = VISION_MODEL_CHOICES
    dataset_types = [IMAGENETTE, PREPROCESSED_SSD]
    root_dir = '/Users/nils/Downloads/worker_batch_size_impact'
    selected_configs = {
        IMAGENETTE: {BATCH_SIZE: 128, NUM_WORKERS: 12},
        PREPROCESSED_SSD: {BATCH_SIZE: 256, NUM_WORKERS: 2}
    }
    for dataset_type in dataset_types:
        config_count = {}
        config_regrets = {}
        for model_name in model_names:

            data = get_metric_numbers_one_model(
                root_dir, [END_TO_END], model_name, batch_sizes, nums_workers, dataset_type,
                batch_size_normalized=False
            )

            transformed = transform_data(data)
            inverted = create_inverted_index(transformed)
            print(f'Model name: {model_name}, dataset type: {dataset_type}')
            top_3_configs = get_top_n_lowest_values(inverted, 3)
            top_config = top_3_configs[0]
            selected_config = data[selected_configs[dataset_type][BATCH_SIZE]][
                selected_configs[dataset_type][NUM_WORKERS]]
            time_regret = selected_config[END_TO_END] - top_config[0]
            print(top_3_configs)
            print(f'regret (abs): {time_regret}, regret (%): {100 * (time_regret / top_config[0])} %')
            print()
            for _time, config in top_3_configs:
                if str(config) not in config_count:
                    config_count[str(config)] = 1
                else:
                    config_count[str(config)] += 1

        print()
        print(f'CONFIG COUNT --- {dataset_type}')
        sorted_dict = sort_dict_by_values(config_count)
        for key, value in sorted_dict.items():
            print(f"{key}: {value}")
        print()


def select_lowest_regret_config(root_dir, dataset_type):
    batch_sizes = [32, 128, 256, 512, 1024]
    nums_workers = [1, 2, 4, 8, 12, 16, 32, 48, 64]

    accumulated_regrets = {}

    for batch_size in batch_sizes:
        for num_workers in nums_workers:
            sum_pct_regret = 0
            for model_name in VISION_MODEL_CHOICES:
                data = get_metric_numbers_one_model(
                    root_dir, [END_TO_END], model_name, batch_sizes, nums_workers, dataset_type,
                    batch_size_normalized=False
                )
                transformed = transform_data(data)
                inverted = create_inverted_index(transformed)
                top_config = get_top_n_lowest_values(inverted, 1)

                selected_config_time = data[batch_size][num_workers][END_TO_END]
                regret = selected_config_time - top_config[0][0]
                pct_regret = regret / top_config[0][0]
                sum_pct_regret += pct_regret
            accumulated_regrets[f'batch_size: {batch_size} - workers: {num_workers}'] = sum_pct_regret / len(
                VISION_MODEL_CHOICES)

    sorted_dict = dict(sorted(accumulated_regrets.items(), key=lambda item: item[1]))
    return sorted_dict


if __name__ == '__main__':
    top_3_confs()
    top = select_lowest_regret_config('/Users/nils/Downloads/worker_batch_size_impact', IMAGENETTE)
    print()
    print(top)
    print()
    top = select_lowest_regret_config('/Users/nils/Downloads/worker_batch_size_impact', PREPROCESSED_SSD)
    print()
    print(top)
    print()
