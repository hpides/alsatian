import os

import torch

from custom.models.init_models import initialize_model
from global_utils.hash import state_dict_hash, architecture_hash
from global_utils.model_names import RESNET_18
from model_search.model_snapshots.base_snapshot import ModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState




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
