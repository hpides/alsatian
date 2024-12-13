import argparse
import configparser

import torch

from custom.models.init_models import initialize_model
from experiments.main_experiments.snapshots.synthetic.generate import generate_snapshot
from global_utils.model_names import RESNET_50

HF_MODEL_CHOICES = [RESNET_50]


def generate_hf_snapshots(architecture_name: str, base_model_id, fine_tuned_model_ids, save_path: str,
                          hf_cache_dir: str,
                          num_models: int = -1) -> [torch.nn.Module]:
    generated_snapshots = []

    hf_model_ids = [base_model_id] + fine_tuned_model_ids
    if num_models > 0:
        hf_model_ids = hf_model_ids[:num_models]

    for hf_model_id in hf_model_ids:
        model = initialize_model(
            architecture_name, features_only=True, hf_identifier=hf_model_id, hf_cache_dir=hf_cache_dir)
        new_snapshot = generate_snapshot(architecture_name, model, save_path, hf_id=hf_model_id.replace("/","___"))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--config_section', required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    args = ExpArgs(config, args.config_section)

    # parse snapshot_id_file
    with open(args.snapshot_ids_file, "r") as file:
        fine_tuned_model_ids = [line.strip() for line in file]

    generate_hf_snapshots(args.model_name, args.base_model_id, fine_tuned_model_ids, args.snapshot_save_path,
                          args.hf_caching_path, args.number_models)
