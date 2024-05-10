from global_utils.ids import random_short_id
from model_search.model_snapshots.base_snapshot import ModelSnapshot

ARCHITECTURE_HASH = "architecture_hash"
STATE_DICT_HASH = "state_dict_hash"
STATE_DICT_PATH = "state_dict_path"
ROOT = "root"


class LayerState:

    def __init__(self, state_dict_path: str, pickled_layer_path: str, state_dict_hash: str, architecture_hash: str,
                 is_leaf: bool = False):
        """
        Represents the state of a single model layer by
        :param state_dict_path: that path to the state dict for that layer
        :param pickled_layer_path: path to a pickled version of the layer
        :param state_dict_hash: a hash value for the layer parameters
        :param architecture_hash: a hash value for the architecture
        """
        self.state_dict_path: str = state_dict_path
        self.pickled_layer_path: str = pickled_layer_path
        self.state_dict_hash: str = state_dict_hash
        self.architecture_hash: str = architecture_hash
        self.id = f'{architecture_hash}-{state_dict_hash}-{random_short_id()}'
        self.is_leaf = is_leaf

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._to_dict())

    def __eq__(self, other):
        if isinstance(other, LayerState):
            return self.architecture_hash == other.architecture_hash and self.state_dict_hash == other.state_dict_hash
        return False

    def _to_dict(self):
        return {
            STATE_DICT_PATH: self.state_dict_path,
            STATE_DICT_HASH: self.state_dict_hash,
            ARCHITECTURE_HASH: self.architecture_hash
        }

    @property
    def is_root_layer(self) -> bool:
        return self.id == ROOT


def generate_root_layer():
    layer_state = LayerState("", "", "", "")
    layer_state.id = ROOT
    return layer_state


class RichModelSnapshot(ModelSnapshot):
    """
    Representing a model snapshot together with some additional information
    """

    def __init__(self, architecture_name: str, state_dict_path: str, state_dict_hash: str, id: str,
                 layer_states: [LayerState]):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        :param state_dict_hash: a hash of the model's state dict
        :param layer_states: a list of layer states represented as a list of LayerState objects
        """

        super().__init__(architecture_name, state_dict_path, state_dict_hash, id)
        self.layer_states: [LayerState] = layer_states

    def _to_dict(self):
        base_dict = super()._to_dict()
        base_dict["state_dict_hash"] = self.state_dict_hash
        base_dict["layer_states"] = str(self.layer_states)
        return base_dict
