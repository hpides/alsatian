import os.path

from custom.models.init_models import initialize_model
from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import write_json_to_file, read_json_to_dict
from global_utils.model_names import RESNET_18
from model_search.model_management.model_store import ModelStore, model_store_from_dict


def dummy_snap_and_mstore(save_path):
    dummy_model_store_path = '/mount-fs/tmp-dir/dummy-model-store.json'
    if os.path.exists(dummy_model_store_path):
        model_store_dict = read_json_to_dict(dummy_model_store_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())
    else:
        # generate some dummy snapshots
        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [5, 7, 9]
        split_idxs = [len(pre_trained_model) - i for i in retrain_idxs]
        model_snapshots = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, save_path=save_path,
                                             retrain_idxs=retrain_idxs, use_same_base=True)

        # add the snapshots to a model store
        model_store = ModelStore(save_path)
        for snapshot in model_snapshots:
            model_store.add_snapshot(snapshot)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, dummy_model_store_path)

    return model_snapshots, model_store
