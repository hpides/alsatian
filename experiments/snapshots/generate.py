# % - start with models_set = {pretrained_model}
# % - while have to generate more models
# % 	- randomly pick model from model_set
# % 	- draw n from distribution (and round to integer)
# % 	- generate new snapshot by
# % 		- copy first (num_layers - n) layers
# % 		- adjust rest of layers
# % 	- add new snapshots to model_set
import random
from enum import Enum

import torch.nn

from custom.models.init_models import initialize_model
from global_utils.model_names import RESNET_18
from global_utils.model_operations import split_model_in_two


class RetrainDistribution(Enum):
    RANDOM = 1
    TOP_LAYERS = 2
    TWENTY_FIVE_PERCENT = 3
    FIFTY_PERCENT = 4


def _adjust_model_randomly(architecture_name: str, base_model: torch.nn.Sequential,
                           distribution: RetrainDistribution) -> torch.nn.Module:
    # draw the number of retrained layers form a distribution
    # TODO
    num_retrained = 3

    keep_layers = len(base_model) - num_retrained
    _, (_, second_layer_names) = split_model_in_two(base_model, keep_layers, include_layer_names=True)
    copied_weights = base_model.state_dict()
    for parm_key in second_layer_names:
        del copied_weights[parm_key]

    new_model = initialize_model(architecture_name, pretrained=False, features_only=True)
    new_model.load_state_dict(copied_weights, strict=False)

    return new_model


def generate_snapshots(architecture_name: str, num_models: int, distribution: RetrainDistribution) -> [torch.nn.Module]:
    # always start with a model pretrained on Imagenet
    pre_trained = initialize_model(architecture_name, pretrained=True, features_only=True)
    generated_models = [pre_trained]
    for _ in range(num_models - 1):
        base_model = random.choice(generated_models)
        new_model = _adjust_model_randomly(architecture_name, base_model, distribution)
        generated_models.append(new_model)

    return generated_models


if __name__ == '__main__':
    pre_trained = initialize_model(RESNET_18, pretrained=True, features_only=True)
    _adjust_model_randomly(RESNET_18, pre_trained, None)
