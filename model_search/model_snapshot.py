import os

import torch

from custom.models.init_models import initialize_model
from global_utils.hash import state_dict_hash
from global_utils.ids import random_short_id
from global_utils.model_names import RESNET_18


class LayerState:
    """
    Represents the state of a single model layer by the path to the layers state dict and a hash of that state dict
    """

    def __init__(self, state_dict_path: str, state_dict_hash: str):
        self.state_dict_path: str = state_dict_path
        self.state_dict_hash: str = state_dict_hash

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._to_dict())

    def _to_dict(self):
        return {
            "state_dict_path": self.state_dict_path,
            "state_dict_hash": self.state_dict_hash
        }


class ModelSnapshot:
    """Simplest form of representing a model"""

    def __init__(self, architecture_name: str, state_dict_path: str):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        """
        self.architecture_name: str = architecture_name
        self.state_dict_path: str = state_dict_path
        self._id = f'{architecture_name}-{random_short_id()}'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._to_dict())

    def _to_dict(self):
        return {
            "architecture_name": self.architecture_name,
            "state_dict_path": self.state_dict_path
        }


class RichModelSnapshot(ModelSnapshot):
    """
    Representing a model snapshot together with some additional infromation
    """

    def __init__(self, architecture_name: str, state_dict_path: str, state_dict_hash: str, layer_states: [LayerState]):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        :param state_dict_hash: a hash of the model's state dict
        :param layer_states: a list of layer states represented as a list of LayerState objects
        """
        super().__init__(architecture_name, state_dict_path)
        self.state_dict_hash: str = state_dict_hash
        self.layer_states: [LayerState] = layer_states

    def _to_dict(self):
        base_dict = super()._to_dict()
        base_dict["state_dict_hash"] = self.state_dict_hash
        base_dict["layer_states"] = str(self.layer_states)
        return base_dict


def to_rich_model_snapshot(snapshot: ModelSnapshot) -> RichModelSnapshot:
    model = initialize_model(snapshot.architecture_name, sequential_model=True, features_only=True)
    state_dict = torch.load(snapshot.state_dict_path)
    model.load_state_dict(state_dict)
    save_path, filename = os.path.split(snapshot.state_dict_path)
    file_name_prefix = filename.replace('.pt', '')
    layer_states = generate_model_layers(model, save_path, name_prefix=file_name_prefix)

    rich_model_snapshot = RichModelSnapshot(
        snapshot.architecture_name,
        snapshot.state_dict_path,
        state_dict_hash=state_dict_hash(state_dict),
        layer_states=layer_states
    )
    return rich_model_snapshot


def generate_model_layers(model, save_path, name_prefix=None):
    layers = []

    model_state = model.state_dict()
    state_dict_keys = list(model_state.keys())

    for layer_i, layer in enumerate(model):
        layer_state = layer.state_dict()
        layer_state_keys = list(layer_state.keys())
        for k in layer_state_keys:
            _key = state_dict_keys.pop(0)
            layer_state[_key] = layer_state.pop(k)

        if name_prefix:
            state_dict_path = os.path.join(save_path, f'{name_prefix}-l-{layer_i}.pt')
        else:
            state_dict_path = os.path.join(save_path, f'l-{layer_i}.pt')

        if not os.path.exists(state_dict_path):
            torch.save(layer_state, state_dict_path)
        layers.append(
            LayerState(state_dict_path, state_dict_hash(layer_state))
        )

    return layers


if __name__ == '__main__':
    model_name = RESNET_18
    state_dict_path = f'/mount-ssd/snapshot-dir/{model_name}.pt'

    if not os.path.exists(state_dict_path):
        model = initialize_model(model_name, sequential_model=True, features_only=True)
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_path)

    snapshot = ModelSnapshot(
        architecture_name=model_name,
        state_dict_path=state_dict_path
    )

    rich_snapshot = to_rich_model_snapshot(snapshot)
    print('test')
