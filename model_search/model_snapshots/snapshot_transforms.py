import os

import torch

from custom.models.init_models import initialize_model
from global_utils.hash import state_dict_hash, architecture_hash
from global_utils.model_names import RESNET_18
from model_search.model_snapshots.base_snapshot import ModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState


def to_rich_model_snapshot(snapshot: ModelSnapshot) -> RichModelSnapshot:
    model = initialize_model(snapshot.architecture_id, sequential_model=True, features_only=True)
    state_dict = torch.load(snapshot.state_dict_path)
    model.load_state_dict(state_dict)
    save_path, filename = os.path.split(snapshot.state_dict_path)
    file_name_prefix = filename.replace('.pt', '')
    layer_states = generate_model_layers(model, save_path, name_prefix=file_name_prefix)

    rich_model_snapshot = RichModelSnapshot(
        snapshot.architecture_id,
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
            os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
            torch.save(layer_state, state_dict_path)

        layers.append(
            LayerState(state_dict_path, state_dict_hash(layer_state), architecture_hash(layer))
        )

    return layers


if __name__ == '__main__':
    model_name = RESNET_18
    state_dict_path = '/Users/nils/uni/programming/model-search-paper/tmp_dir/res18.pt'

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
