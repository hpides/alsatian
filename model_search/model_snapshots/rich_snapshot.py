from global_utils.ids import random_short_id
from global_utils.json_operations import list_to_dict
from model_search.model_snapshots.base_snapshot import ModelSnapshot, SAVE_PATH, ARCHITECTURE_ID

LEAF = "is_leaf"
ID = "id"
LAYER_STATES = "layer_states"
ARCHITECTURE_HASH = "architecture_hash"
STATE_DICT_HASH = "state_dict_hash"
STATE_DICT_PATH = "state_dict_path"
ROOT = "root"
PICKLED_LAYER_PATH = "pickled_layer_path"


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
        self.output_size = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        if isinstance(other, LayerState):
            return self.architecture_hash == other.architecture_hash and self.state_dict_hash == other.state_dict_hash
        return False

    def to_dict(self):
        return {
            STATE_DICT_PATH: self.state_dict_path,
            PICKLED_LAYER_PATH: self.pickled_layer_path,
            STATE_DICT_HASH: self.state_dict_hash,
            ARCHITECTURE_HASH: self.architecture_hash,
            ID: self.id,
            LEAF: self.is_leaf
        }

    @property
    def is_root_layer(self) -> bool:
        return self.id == ROOT

    def add_output_size(self, output_size):
        self.output_size = output_size


def generate_root_layer():
    layer_state = LayerState("", "", "", "")
    layer_state.id = ROOT
    return layer_state


def layer_state_from_dict(_dict) -> LayerState:
    layer_state = LayerState(
        _dict[STATE_DICT_PATH],
        _dict[PICKLED_LAYER_PATH],
        _dict[STATE_DICT_HASH],
        _dict[ARCHITECTURE_HASH],
        _dict[LEAF]
    )
    layer_state.id = _dict[ID]
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

    def to_dict(self):
        result = super().to_dict()
        result[LAYER_STATES] = list_to_dict(self.layer_states)
        return result


def rich_model_snapshot_from_dict(_dict) -> RichModelSnapshot:
    save_path = _dict[SAVE_PATH]
    architecture_id = _dict[ARCHITECTURE_ID]
    state_dict_hash = _dict[STATE_DICT_HASH]
    id = _dict[ID]
    layer_states = []
    for l_state in _dict[LAYER_STATES]:
        layer_states.append(layer_state_from_dict(l_state))

    return RichModelSnapshot(architecture_id, save_path, state_dict_hash, id, layer_states)
