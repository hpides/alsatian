import os

from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import read_json_to_dict, write_json_to_file
from global_utils.model_names import VISION_MODEL_CHOICES
from model_search.model_management.model_store import model_store_from_dict, ModelStore


def def_generate_snapshot_set(architecture_name, num_models, distribution: RetrainDistribution, base_path):
    save_path = os.path.join(base_path, architecture_name, str(distribution).replace("RetrainDistribution.", ""),
                             str(num_models))
    dummy_model_store_path = os.path.join(save_path, 'model_store.json')
    if os.path.exists(dummy_model_store_path):
        model_store_dict = read_json_to_dict(dummy_model_store_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())
    else:
        # make sure save path exists
        os.makedirs(save_path)
        # generate some dummy snapshots
        model_snapshots = generate_snapshots(architecture_name, num_models, RetrainDistribution.TOP_LAYERS,
                                             save_path=save_path)

        # add the snapshots to a model store
        model_store = ModelStore(save_path)
        for snapshot in model_snapshots:
            model_store.add_snapshot(snapshot)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, dummy_model_store_path)

    return model_snapshots, model_store


if __name__ == '__main__':
    base_path = '/mount-fs/snapshot-sets'
    distributions = [RetrainDistribution.TOP_LAYERS, RetrainDistribution.TWENTY_FIVE_PERCENT,
                     RetrainDistribution.FIFTY_PERCENT]

    model_list = VISION_MODEL_CHOICES

    for dist in distributions:
        for model in model_list:
            print('generating', dist, model)
            def_generate_snapshot_set(model, 50, dist, base_path)
