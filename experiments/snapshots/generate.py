import os.path
import random
from enum import Enum

import torch.nn

from custom.models.init_models import initialize_model
from experiments.snapshots.retrain_distribution import normal_retrain_layer_dist_last_few, normal_retrain_layer_dist_25, \
    normal_retrain_layer_dist_50
from global_utils.hash import state_dict_hash
from global_utils.model_operations import split_model_in_two
from model_search.model_snapshots.base_snapshot import ModelSnapshot, generate_snapshot_id


class RetrainDistribution(Enum):
    HARD_CODED = 1
    RANDOM = 2
    TOP_LAYERS = 3
    TWENTY_FIVE_PERCENT = 4
    FIFTY_PERCENT = 5


def _num_retrained_layers(max_number, distribution: RetrainDistribution) -> int:
    if distribution == RetrainDistribution.RANDOM:
        return random.randint(0, max_number)
    else:
        raise ValueError(f"invalid distribution: {distribution}")


def _adjust_model_randomly(architecture_name: str, base_model: torch.nn.Sequential,
                           distribution: RetrainDistribution, retrain_idx=None) -> torch.nn.Module:
    if retrain_idx is None:
        retrain_idx = _num_retrained_layers(len(base_model), distribution)

    keep_layers = len(base_model) - retrain_idx
    _, (_, second_layer_names) = split_model_in_two(base_model, keep_layers, include_layer_names=True)
    copied_weights = base_model.state_dict()
    for parm_key in second_layer_names:
        del copied_weights[parm_key]

    new_model = initialize_model(architecture_name, pretrained=False, features_only=True)
    new_model.load_state_dict(copied_weights, strict=False)

    return new_model


def generate_snapshots(architecture_name: str, num_models: int, distribution: RetrainDistribution, save_path: str,
                       retrain_idxs=None, use_same_base=False, ) -> [torch.nn.Module]:
    # always start with a model pretrained on Imagenet
    model = initialize_model(architecture_name, pretrained=True, features_only=True)
    snapshot = generate_snapshot(architecture_name, model, save_path)
    base_snapshot = snapshot
    base_model = base_snapshot.init_model_from_snapshot()

    if distribution == RetrainDistribution.HARD_CODED:
        assert retrain_idxs is not None
    elif distribution == RetrainDistribution.TOP_LAYERS:
        retrain_idxs = normal_retrain_layer_dist_last_few(len(base_model), num_models - 1)
    elif distribution == RetrainDistribution.TWENTY_FIVE_PERCENT:
        retrain_idxs = normal_retrain_layer_dist_25(len(base_model), num_models - 1)
    elif distribution == RetrainDistribution.FIFTY_PERCENT:
        retrain_idxs = normal_retrain_layer_dist_50(len(base_model), num_models - 1)
    else:
        raise NotImplementedError

    print('retrain_idxs', retrain_idxs)

    generated_snapshots = [snapshot]
    for i in range(num_models - 1):
        if not use_same_base:
            base_snapshot = random.choice(generated_snapshots)
            base_model = base_snapshot.init_model_from_snapshot()

        if retrain_idxs:
            new_model = _adjust_model_randomly(architecture_name, base_model, distribution, retrain_idxs[i])
        else:
            new_model = _adjust_model_randomly(architecture_name, base_model, distribution)

        new_snapshot = generate_snapshot(architecture_name, new_model, save_path)
        generated_snapshots.append(new_snapshot)

    return generated_snapshots


def generate_snapshot(architecture_name, model, save_path):
    pre_trained_state = model.state_dict()
    sd_hash = state_dict_hash(pre_trained_state)
    snapshot_id = generate_snapshot_id(architecture_name, sd_hash)
    state_dict_path = os.path.join(save_path, f'{snapshot_id}.pt')
    if not os.path.exists(state_dict_path):
        torch.save(pre_trained_state, state_dict_path)
    snapshot = ModelSnapshot(architecture_name, state_dict_path, sd_hash, snapshot_id)
    return snapshot
