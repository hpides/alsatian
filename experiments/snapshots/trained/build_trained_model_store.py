import os.path

import torch

from custom.models.init_models import initialize_model
from experiments.snapshots.trained.generate_trained_snapshots import IMAGE_WOOF, STANFORD_DOGS, STANFORD_CARS, \
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


if __name__ == '__main__':
    dataset_names = [IMAGE_WOOF, STANFORD_DOGS, STANFORD_CARS, CUB_BIRDS_200, FOOD_101]
    model_architectures = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
    snapshot_base_path = "/mount-fs/trained-snapshots-OLD"
    base_model_store_save_path = "/mount-fs/trained-snapshots-OLD/modelstore_savepath"
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
