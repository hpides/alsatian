import torch

from custom.models.init_models import initialize_model
from global_utils.hash import state_dict_hash
from global_utils.ids import random_short_id

ID = "id"

STATE_DICT_HASH = "state_dict_hash"

ARCHITECTURE_ID = "architecture_id"

SAVE_PATH = "save_path"


def generate_snapshot_id(architecture_name, state_dict_hash):
    return f'{architecture_name}-{state_dict_hash}-{random_short_id()}'


class ModelSnapshot:
    """Simplest form of representing a model"""

    def __init__(self, architecture_name: str, state_dict_path: str, sdict_hash: str = None, id: str = None):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        """
        self.architecture_id: str = architecture_name
        self.state_dict_path: str = state_dict_path

        if sdict_hash is None:
            state_dict = torch.load(state_dict_path)
            self.state_dict_hash = state_dict_hash(state_dict)
        else:
            self.state_dict_hash: str = sdict_hash

        # a model is defined by its architecture and the parameters
        if id is None:
            self.id = generate_snapshot_id(architecture_name, self.state_dict_hash)
        else:
            self.id = id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        if isinstance(other, ModelSnapshot):
            return self.architecture_id == other.architecture_id and self.state_dict_hash == other.state_dict_hash
        return False

    def to_dict(self):
        return {
            SAVE_PATH: self.state_dict_path,
            ARCHITECTURE_ID: self.architecture_id,
            STATE_DICT_HASH: self.state_dict_hash,
            ID: self.id
        }

    def init_model_from_snapshot(self):
        model = initialize_model(self.architecture_id, sequential_model=True, features_only=True)
        state_dict = torch.load(self.state_dict_path)
        model.load_state_dict(state_dict)
        return model


def model_snapshot_from_dict(snapshot_dict) -> ModelSnapshot:
    save_path = snapshot_dict[SAVE_PATH]
    architecture_id = snapshot_dict[ARCHITECTURE_ID]
    state_dict_hash = snapshot_dict[STATE_DICT_HASH]
    id = snapshot_dict[ID]

    return ModelSnapshot(architecture_id, save_path, state_dict_hash, id)
