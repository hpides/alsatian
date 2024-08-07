import os.path

import torch

from custom.models.init_models import initialize_model
from experiments.main_experiments.snapshots.trained.generate_trained_snapshots import IMAGE_WOOF, STANFORD_DOGS, STANFORD_CARS, \
    CUB_BIRDS_200, FOOD_101
from global_utils.hash import state_dict_hash
from global_utils.json_operations import write_json_to_file, read_json_to_dict
from global_utils.model_names import RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32
from model_search.model_management.model_store import ModelStore, model_store_from_dict
from model_search.model_snapshots.base_snapshot import ModelSnapshot, generate_snapshot_id


def find_model_paths(directory, start_str, end_str):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(start_str) and file.endswith(end_str):
                matched_files.append(os.path.join(root, file))
    return matched_files


def collect_snapshots(snapshot_base_path, architecture_name, dataset_names):
    state_dict_paths = []
    model_snapshots = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(snapshot_base_path, dataset_name)
        state_dict_paths += find_model_paths(dataset_path, architecture_name, "epoch-20.pth")

    for state_dict_path in state_dict_paths:
        state_dict = torch.load(state_dict_path)
        sd_hash = state_dict_hash(state_dict)
        snapshot_id = generate_snapshot_id(architecture_name, sd_hash)
        snapshot = ModelSnapshot(architecture_name, state_dict_path, sd_hash, snapshot_id)
        model_snapshots.append(snapshot)

    return model_snapshots


def build_model_store(save_path, model_snapshots):
    model_store = ModelStore(save_path)
    for snapshot in model_snapshots:
        model_store.add_snapshot(snapshot)

    return model_store


def get_existing_model_store(architecture_name, base_model_store_save_path):
    model_store_save_path = get_model_store_save_path(base_model_store_save_path, architecture_name)
    model_store_json_path = os.path.join(model_store_save_path, 'model_store.json')

    model_store_dict = read_json_to_dict(model_store_json_path)
    model_store = model_store_from_dict(model_store_dict)

    model_snapshots = list(model_store.models.values())

    return model_snapshots, model_store


def get_model_store_save_path(base_model_store_save_path, architecture_name):
    return os.path.join(base_model_store_save_path, f'{architecture_name}-model-store')


def get_trained_models_and_model_store(architecture_name, base_model_store_save_path):
    model_store_save_path = get_model_store_save_path(base_model_store_save_path, architecture_name)
    model_store_json_path = os.path.join(model_store_save_path, 'model_store.json')
    model_store_dict = read_json_to_dict(model_store_json_path)
    model_store = model_store_from_dict(model_store_dict)

    if architecture_name == EFF_NET_V2_L:
        # for this model we generated to many models (we want 36 as for the other so delete some models)
        delete_ids = [
            'eff_net_v2_l-e92a66bafef812a6afdac7f307dbf1ac-hkvbys8h',
            'eff_net_v2_l-4b2637f71e89ed721040184ea414df7a-ptekjjj3',
            'eff_net_v2_l-50416073c122743ddd2fdf13f977bbc0-78ywe5bo',
            'eff_net_v2_l-80e2c11d7856ebbeae16b83d6d820a2f-2dh8qxge',
            'eff_net_v2_l-5a12f02a9bd5c88dd6a7be8274372d41-rpkyd784',
            'eff_net_v2_l-5fc92fc4db6d3374a468ef4482e80a8d-9kyrv978'
        ]
        for _id in delete_ids:
            del model_store.models[_id]

    return list(model_store.models.values()), model_store


if __name__ == '__main__':
    dataset_names = [IMAGE_WOOF, STANFORD_DOGS, STANFORD_CARS, CUB_BIRDS_200, FOOD_101]
    model_architectures = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
    snapshot_base_path = "/mount-ssd/snapshot-dir"
    base_model_store_save_path = "/mount-fs/trained-snapshots/modelstore_savepath"
    epochs_trained = 20

    for architecture_name in model_architectures:
        print()
        print(architecture_name)

        model_store_save_path = get_model_store_save_path(base_model_store_save_path, architecture_name)
        os.makedirs(model_store_save_path)

        snapshots = collect_snapshots(snapshot_base_path, architecture_name, dataset_names)

        # next to the trained snapshots also add a snapshot that is the on form PyTorch pretrained on ImageNet
        pre_model = initialize_model(architecture_name, pretrained=True, sequential_model=True, features_only=True)
        state_dict = pre_model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.to("cpu")
        sd_save_path = os.path.join(model_store_save_path, 'imagenet_model.pth')
        torch.save(state_dict, sd_save_path)
        sd_hash = state_dict_hash(state_dict)
        snapshot_id = generate_snapshot_id(architecture_name, sd_hash)
        snapshot = ModelSnapshot(architecture_name, sd_save_path, sd_hash, snapshot_id)
        snapshots.append(snapshot)

        model_store = build_model_store(model_store_save_path, snapshots)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        model_store_json_path = os.path.join(model_store_save_path, 'model_store.json')
        write_json_to_file(model_store_dict, model_store_json_path)
