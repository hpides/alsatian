class ModelSnapshot:
    """Simples form of representing a model"""

    def __init__(self, architecture_name: str, state_dict_path: str):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        """
        self.architecture_name: str = architecture_name
        self.state_dict_path: str = state_dict_path

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._to_dict())

    def _to_dict(self):
        return {
            "architecture_name": self.architecture_name,
            "state_dict_path": self.state_dict_path
        }


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
