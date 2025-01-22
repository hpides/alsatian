import json
import os

import torch
import torch.nn

from custom.models.init_models import initialize_model
from global_utils.hash import state_dict_hash, architecture_hash
from global_utils.json_operations import dict_to_dict
from global_utils.state_dict_comparison import _compare_state_dicts_key_wise
from model_search.caching_service import CachingService
from model_search.model_snapshots.base_snapshot import ModelSnapshot
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState, rich_model_snapshot_from_dict, \
    layer_state_from_dict

LAYERS = 'layers'

MODELS = 'models'

SAVE_PATH = 'save_path'


def model_store_from_dict(_dict):
    model_store = ModelStore(_dict[SAVE_PATH])

    models = {}
    for k, v in _dict[MODELS].items():
        models[k] = rich_model_snapshot_from_dict(v)

    layers = {}
    for k, v in _dict[LAYERS].items():
        layers[k] = layer_state_from_dict(v)

    model_store.models = models
    model_store.layers = layers

    return model_store


def match_keys(target_sd_format, current_sd):
    if not next(iter(current_sd)).startswith("0.0"):
        # we only need to adjust of the current sd starts like this
        return current_sd

    def remove_duplicates(input_list):
        seen = set()
        result = []
        for item in input_list:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def extract_prefixes(_keys, prefix_length):
        prefixes = []
        for k in _keys:
            split = k.split('.')
            prefix = ".".join(split[:prefix_length])
            prefixes.append(prefix)

        return prefixes

    # Example dictionary with the given keys
    reference_keys = list(target_sd_format.keys())
    current_keys = list(current_sd.keys())

    reference_keys_prefixes = remove_duplicates(extract_prefixes(reference_keys, 1))
    current_keys_prefixes = remove_duplicates(extract_prefixes(current_keys, 2))

    prefix_mapping = {}
    for i, r_pre in enumerate(reference_keys_prefixes):
        prefix_mapping[current_keys_prefixes[i]] = r_pre

    for k in list(current_sd.keys()):
        current_components = k.split(".")
        replace = current_components[:2]
        keep = current_components[2:]

        old_prefix = ".".join(replace)

        if old_prefix in prefix_mapping:
            new_prefix = prefix_mapping[old_prefix]

            new_key = ".".join([new_prefix] + keep)

            current_sd[new_key] = current_sd[k]

        del current_sd[k]

    for k in list(current_sd.keys()):
        if "running" in k or "tracked" in k:
            current_sd[k] = target_sd_format[k]

    return current_sd


class ModelStore:

    def __init__(self, save_path: str, caching_service=None):
        self.save_path = save_path
        self.models = {}
        self.layers = {}
        self.caching_service: CachingService = caching_service

    @property
    def model_caching_active(self):
        return self.caching_service is not None

    def activate_caching(self, caching_service: CachingService):
        self.caching_service = caching_service

    def to_dict(self):
        result = {}
        result[SAVE_PATH] = self.save_path
        result[MODELS] = dict_to_dict(self.models)
        result[LAYERS] = dict_to_dict(self.layers)
        return result

    def add_snapshot(self, model_snapshot: ModelSnapshot):
        if not isinstance(model_snapshot, RichModelSnapshot):
            rich_model_snapshot = self._gen_rich_model_snapshot(model_snapshot)
        else:
            rich_model_snapshot = model_snapshot

        self.models[rich_model_snapshot.id] = rich_model_snapshot
        self._index_layers(rich_model_snapshot)

    def merge_model_store(self, model_store):
        rich_snapshots = list(model_store.models.values())
        for snapshot in rich_snapshots:
            self.add_snapshot(snapshot)


    def add_output_sizes_to_rich_snapshots(self, info_json, default_size=None):
        with open(info_json, 'r') as file:
            output_info = json.load(file)

            for model_snap in self.models.values():
                for layer_state in model_snap.layer_states:
                    if (model_snap.architecture_id in output_info
                            and layer_state.architecture_hash in output_info[model_snap.architecture_id]):
                        output_size = output_info[model_snap.architecture_id][layer_state.architecture_hash]
                        layer_state.add_output_size(output_size)
                    else:
                        layer_state.add_output_size(default_size)
    def _index_layers(self, rich_snapshot: RichModelSnapshot):
        for layer_state in rich_snapshot.layer_states:
            self.layers[layer_state.id] = layer_state

    def get_snapshot(self, snapshot_id: str) -> RichModelSnapshot:
        return self.models[snapshot_id]

    def get_model(self, snapshot_id: str) -> torch.nn.Module:
        snapshot: ModelSnapshot = self.get_snapshot(snapshot_id)
        return snapshot.init_model_from_snapshot()

    def get_composed_model(self, layer_state_ids: [str]) -> torch.nn.Module:
        # returns a sequential model, that is a sequential model chained of the layer states given
        print(len(layer_state_ids))
        layers = []
        for layer_id in layer_state_ids:
            layer = self._init_layer(layer_id)
            layers.append(layer)

        return torch.nn.Sequential(*(list(layers)))

    def get_multi_model_snapshot(self, model_ids: [str]) -> MultiModelSnapshot:
        pass

    def _gen_rich_model_snapshot(self, snapshot: ModelSnapshot) -> RichModelSnapshot:
        model = initialize_model(snapshot.architecture_id, sequential_model=True, features_only=True, pretrained=True)
        state_dict = torch.load(snapshot.state_dict_path)
        for k, v in state_dict.items():
            state_dict[k] = v.to("cpu")
        adjusted_state_dict = match_keys(model.state_dict(), state_dict)

        # print how many values in state ict are matching
        _compare_state_dicts_key_wise(model.state_dict(), state_dict)

        model.load_state_dict(adjusted_state_dict)
        _, filename = os.path.split(snapshot.state_dict_path)
        file_name_prefix = filename.replace('.pt', '')
        layer_states = self._gen_model_layers(model, self.save_path, name_prefix=file_name_prefix)

        dict_hash = state_dict_hash(adjusted_state_dict)
        architecture_name = snapshot.architecture_id
        rich_model_snapshot = RichModelSnapshot(
            architecture_name=architecture_name,
            state_dict_path=snapshot.state_dict_path,
            state_dict_hash=dict_hash,
            id=snapshot.id,
            layer_states=layer_states
        )
        return rich_model_snapshot

    def _gen_model_layers(self, model, save_path, name_prefix=None):
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
                base_path = os.path.join(save_path, f'{name_prefix}-l-{layer_i}')
                state_dict_path = f'{base_path}.pt'
                pickled_layer_path = f'{base_path}.pkl'
            else:
                base_path = os.path.join(save_path, f'l-{layer_i}')
                state_dict_path = f'{base_path}.pt'
                pickled_layer_path = f'{base_path}.pkl'

            if not os.path.exists(state_dict_path):
                os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
                torch.save(layer_state, state_dict_path)
                torch.save(layer, pickled_layer_path)

            layers.append(
                LayerState(state_dict_path, pickled_layer_path, state_dict_hash(layer_state), architecture_hash(layer))
            )

        # mark the last layer as a leaf
        layers[-1].is_leaf = True

        return layers

    def _init_layer(self, _id) -> torch.nn.Module:
        # NOTE: loading pickled layer and not state dicts is not a very nice way of doing it, but ok for now
        layer_state: LayerState = self.layers[_id]
        layer_path = layer_state.pickled_layer_path
        layer_name = os.path.basename(layer_path)
        if self.model_caching_active and self.caching_service.id_exists(layer_name):
            # if cached on SSD, load from there
            loaded_layer = self.caching_service.get_item(layer_name)
        else:
            # if not cached, load form external and cache on SSD
            loaded_layer = torch.load(layer_path)
            if self.model_caching_active:
                self.caching_service.cache_persistent(layer_name, loaded_layer, is_guaranteed_cpu_data=True)

        return loaded_layer
