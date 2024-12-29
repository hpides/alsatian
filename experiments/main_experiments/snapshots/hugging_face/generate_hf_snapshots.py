import argparse
import configparser
import os

import torch

from experiments.main_experiments.snapshots.hugging_face.init_hf_models import initialize_hf_model
from experiments.main_experiments.snapshots.synthetic.generate import generate_snapshot
from global_utils.json_operations import read_json_to_dict, write_json_to_file
from global_utils.model_names import RESNET_50
from model_search.model_management.model_store import ModelStore, model_store_from_dict

HF_MODEL_CHOICES = [RESNET_50]


def build_model_store(save_path, model_snapshots):
    model_store = ModelStore(save_path)
    for snapshot in model_snapshots:
        model_store.add_snapshot(snapshot)

    return model_store


def get_existing_model_store(model_store_save_path):
    model_store_json_path = os.path.join(model_store_save_path, 'model_store.json')

    model_store_dict = read_json_to_dict(model_store_json_path)
    model_store = model_store_from_dict(model_store_dict)

    model_snapshots = list(model_store.models.values())

    return model_snapshots, model_store


def generate_hf_snapshots(base_model_id, fine_tuned_model_ids, save_path: str, hf_cache_dir: str,
                          num_models: int = -1) -> [torch.nn.Module]:
    generated_snapshots = []

    hf_model_ids = [base_model_id] + fine_tuned_model_ids
    if num_models > 0:
        hf_model_ids = hf_model_ids[:num_models]

    for hf_model_id in hf_model_ids:
        architecture_name, model = initialize_hf_model(base_model_id, hf_model_id, hf_cache_dir)
        new_snapshot = generate_snapshot(architecture_name, model, save_path, hf_id=hf_model_id.replace("/", "___"))
        generated_snapshots.append(new_snapshot)

    return generated_snapshots


class ExpArgs:
    def __init__(self, args, section):
        self.model_name = args[section].get('model_name', None)
        self.hf_caching_path = args[section].get('hf_caching_path', None)
        self.snapshot_save_path = args[section].get('snapshot_save_path', None)
        self.snapshot_ids_file = args[section].get('snapshot_ids_file', None)
        self.base_model_id = args[section].get('base_model_id', None)
        self.number_models = args.getint(section, 'number_models', fallback=1)

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__


if __name__ == '__main__':
    # TODO run this for every model we are interested in
    # TODO check that everything is written to the correct directories

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--config_section', required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    args = ExpArgs(config, args.config_section)

    model_store_json_path = os.path.join(args.snapshot_save_path, 'model_store.json')
    if os.path.exists(model_store_json_path):
        # execute just to see if we get any errors
        get_existing_model_store(args.snapshot_save_path)
    else:
        # parse snapshot_id_file
        snapshot_id_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.snapshot_ids_file)
        with open(snapshot_id_file_path, "r") as file:
            fine_tuned_model_ids = [line.strip() for line in file]

        snapshots = generate_hf_snapshots(args.base_model_id, fine_tuned_model_ids,
                                          args.snapshot_save_path,
                                          args.hf_caching_path, args.number_models)

        model_store = build_model_store(args.snapshot_save_path, snapshots)
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, model_store_json_path)

    print("test")
