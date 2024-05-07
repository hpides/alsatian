import random
from enum import Enum

import torch.nn

from custom.models.init_models import initialize_model
from global_utils.model_names import RESNET_18
from global_utils.model_operations import split_model_in_two


class RetrainDistribution(Enum):
    HARD_CODED = 1
    RANDOM = 2
    TOP_LAYERS = 3
    TWENTY_FIVE_PERCENT = 4
    FIFTY_PERCENT = 5


def _num_retrained_layers(max_number, distribution: RetrainDistribution) -> int:
    if distribution == RetrainDistribution.RANDOM:
        return random.randint(0, max_number)
    elif distribution == RetrainDistribution.TOP_LAYERS:
        raise NotImplementedError
    elif distribution == RetrainDistribution.TWENTY_FIVE_PERCENT:
        raise NotImplementedError
    elif distribution == RetrainDistribution.FIFTY_PERCENT:
        raise NotImplementedError
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


def generate_snapshots(architecture_name: str, num_models: int, distribution: RetrainDistribution,
                       retrain_idxs=None, use_same_base=False) -> [torch.nn.Module]:
    # TODO for now its fine to not save a snapshot to disk, later we have to save them inbetween because we will run out of memory
    if distribution == RetrainDistribution.HARD_CODED:
        assert retrain_idxs is not None

    # always start with a model pretrained on Imagenet
    pre_trained = initialize_model(architecture_name, pretrained=True, features_only=True)
    generated_models = [pre_trained]
    for i in range(num_models - 1):
        if use_same_base:
            base_model = generated_models[0]
        else:
            base_model = random.choice(generated_models)
        if retrain_idxs:
            new_model = _adjust_model_randomly(architecture_name, base_model, distribution, retrain_idxs[i])
        else:
            new_model = _adjust_model_randomly(architecture_name, base_model, distribution)
        generated_models.append(new_model)

    return generated_models


if __name__ == '__main__':
    snaps = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, [5, 7, 9])
    print('test')
