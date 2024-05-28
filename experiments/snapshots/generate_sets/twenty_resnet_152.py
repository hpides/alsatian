import os

from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import read_json_to_dict, write_json_to_file
from global_utils.model_names import RESNET_152
from model_search.model_management.model_store import model_store_from_dict, ModelStore


def twenty_resnet_152_snapshots(save_path):
    # retrain indexes [2, 5, 4, 3, 1, 1, 0, 4, 3, 3, 0, 6, 4, 1, 1, 1, 2, 3, 2]
    dummy_model_store_path = os.path.join(save_path, 'model_store.json')
    if os.path.exists(dummy_model_store_path):
        model_store_dict = read_json_to_dict(dummy_model_store_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())
    else:
        # generate some dummy snapshots
        model_snapshots = generate_snapshots(RESNET_152, 20, RetrainDistribution.TOP_LAYERS,
                                             save_path=save_path, use_same_base=True)

        # add the snapshots to a model store
        model_store = ModelStore(save_path)
        for snapshot in model_snapshots:
            model_store.add_snapshot(snapshot)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, dummy_model_store_path)

    return model_snapshots, model_store
