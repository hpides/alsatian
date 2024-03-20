import torch
from torch import nn

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


def get_split_index(split_index, model_name):
    try:
        split_index = int(split_index)
    except ValueError:
        return None

    num_layers = len(SPLIT_INDEXES[model_name])
    if split_index >= 0:
        # interpret split-level as percentage number, length of available split points
        # example: assume we have 10 layers and get split_index 50, we want to split the model 50/50 -> split index 5
        # example: assume we have 10 layers and get split_index 75, we want to split the model 25/75 -> split index 0.75*10 -> 7.5 -> rounded to 8
        split_index = int(num_layers * (split_index / 100))
    elif split_index < 0:
        # we want to split the last abs(split_index) layers
        # example: split index -2, model has 10 layers -> split 8/2 -> split index 8
        split_index = SPLIT_INDEXES[model_name][-1 * split_index]
        if split_index < 0:
            raise ValueError("split not possible; negative split index abs too high")
    return split_index


def merge_models(base_model: torch.nn.Sequential, to_merge: torch.nn.Sequential, _index):
    base_model_one, head_one = split_model(base_model, _index)
    base_model_two, head_two = split_model(to_merge, _index)

    class MergedHeadModel(nn.Module):
        def __init__(self, head_one, head_two):
            super(MergedHeadModel, self).__init__()
            self.head_one = head_one
            self.head_two = head_two

        def forward(self, x):
            x1 = self.head_one(x)
            x2 = self.head_two(x)
            x = torch.cat((x1, x2), dim=1)
            return x

    merged_model = nn.Sequential(
        *list(base_model_one.children()),
        MergedHeadModel(head_one, head_two)
    )
    return merged_model
