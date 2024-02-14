import torch

from custom.models.split_indices import SPLIT_INDEXES


def _in_class_list(child, split_classes):
    for cls in split_classes:
        if isinstance(child, cls):
            return True


def transform_to_sequential(model, include_seq=False, split_classes=None):
    layers = list_of_layers(model, include_seq=include_seq, split_classes=split_classes)
    seq_model = torch.nn.Sequential(*(list(layers)))
    return seq_model


def split_model(model, index, include_layer_names=False):
    assert index <= len(model), "split index larger than available layers"
    layers = list_of_layers(model)
    entire_model = transform_to_sequential(model)
    first_part = torch.nn.Sequential(*(list(layers[:index])))
    second_part = torch.nn.Sequential(*(list(layers[index:])))

    entire_model_state = entire_model.state_dict()
    layer_names = list(entire_model_state.keys())
    name_split_index = len(first_part.state_dict().keys())

    if include_layer_names:
        return (first_part, layer_names[:name_split_index]), (second_part, layer_names[name_split_index:])
    else:
        return first_part, second_part


def list_of_layers(model: torch.nn.Sequential, include_seq=False, split_classes=None):
    result = []
    children = list(model.children())
    for child in children:
        if split_classes and _in_class_list(child, split_classes):
            result += list_of_layers(child, include_seq)
        elif isinstance(child, torch.nn.Sequential) or (include_seq and len(list(child.children())) > 0):
            result += list_of_layers(child, include_seq)
        else:
            result += [child]
    return result


def get_split_index(split_index, model, model_name):
    try:
        split_index = int(split_index)
    except:
        return None

    if split_index >= 0:
        # interpret split-level as percentage number
        num_layers = len(model)
        split_index = int(num_layers * (1 - split_index))
    elif split_index == -1 or split_index == -2:
        # look up split index
        split_index = SPLIT_INDEXES[model_name][-1 * split_index]
    else:
        raise NotImplementedError(f"Split index of {split_index} currently not supported.")
    return split_index
