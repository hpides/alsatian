STATE_DICT_PATH = "state_dict_path"

ARCHITECTURE_ID = "architecture_id"


class ModelSnapshot:
    """Simplest form of representing a model"""

    def __init__(self, architecture_name: str, state_dict_path: str):
        """
        :param architecture_name: the name of the model architecture that can be used to initialize a Pytorch model
         following a specific architecture, can also be an abstract name like a hash
        :param state_dict_path: the path to a state_dict that holds all model parameters
        """
        self.architecture_id: str = architecture_name
        self.state_dict_path: str = state_dict_path
        # a model is defined by its architecture and the parameters
        self._id = f'{self.architecture_id}-{self.state_dict_path}'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._to_dict())

    def __eq__(self, other):
        if isinstance(other, ModelSnapshot):
            return self._id == other._id
        return False

    def _to_dict(self):
        return {
            ARCHITECTURE_ID: self.architecture_id,
            STATE_DICT_PATH: self.state_dict_path
        }
