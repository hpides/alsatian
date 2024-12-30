import os
import pathlib

from custom.data_loaders import custom_image_folder
from custom.data_loaders.custom_image_folder import CustomImageFolder
from experiments.dummy_experiments.dummy_model_snapshots import get_three_random_two_bock_models
from experiments.main_experiments.model_search.experiment_args import _str_to_cache_location
from global_utils.constants import TRAIN, TEST
from model_search.approaches import mosix
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.planning.planner_config import PlannerConfig

if __name__ == '__main__':
    dataset_paths = {
        TRAIN: custom_image_folder.create_sub_dataset("/mount-ssd/data/imagenette2/train", 6),
        TEST: custom_image_folder.create_sub_dataset("/mount-ssd/data/imagenette2/val", 2)
    }
    dataset_class = DatasetClass.CUSTOM_IMAGE_FOLDER
    train_data = CustomImageFolder(dataset_paths[TRAIN])
    test_data = CustomImageFolder(dataset_paths[TEST])

    planner_config = PlannerConfig(1, 2, 10, dataset_class, dataset_paths,
                                   _str_to_cache_location("CPU"), 100000)

    model_snapshots, model_store = get_three_random_two_bock_models()

    layer_output_info = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                     '../side_experiments/model_resource_info/outputs/layer_output_infos.json')
    model_store.add_output_sizes_to_rich_snapshots(layer_output_info, default_size=20)

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    persistent_caching_path = "/mount-ssd/cache-dir"

    args = [model_snapshots, len_train_data, len_test_data, planner_config, persistent_caching_path, model_store]

    mosix.find_best_model(*args)
