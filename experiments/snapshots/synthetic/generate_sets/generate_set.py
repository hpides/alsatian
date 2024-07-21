import os

from experiments.snapshots.synthetic.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import read_json_to_dict, write_json_to_file
from global_utils.model_names import VISION_MODEL_CHOICES, RESNETS, TRANSFORMER_MODELS, MOBILE_V2, RESNET_34, RESNET_18, \
    EFF_NET_V2_S, RESNET_50, RESNET_101, VIT_B_16, VIT_B_32, RESNET_152, VIT_L_32, VIT_L_16, EFF_NET_V2_L, \
    CONVOLUTION_MODELS
from model_search.model_management.model_store import model_store_from_dict, ModelStore


def distribute_into_buckets(total, num_buckets):
    # Initialize the buckets with the base value
    base_value = total // num_buckets
    buckets = [base_value] * num_buckets

    # Distribute the remainder
    remainder = total % num_buckets
    for i in range(remainder):
        buckets[i] += 1

    return buckets


def _higher_path(save_path: str, start):
    for i in range(start + 1, 100):
        higher_path = save_path.replace(f'/{start}', f'/{i}')
        if os.path.exists(higher_path):
            return higher_path


def generate_snapshot_set(architecture_name, num_models, distribution: RetrainDistribution, base_path,
                          reuse_allowed=False):
    save_path = _save_path(architecture_name, base_path, distribution, num_models)
    dummy_model_store_path = os.path.join(save_path, 'model_store.json')
    if os.path.exists(dummy_model_store_path):
        model_store_dict = read_json_to_dict(dummy_model_store_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())
    elif reuse_allowed and _higher_path(dummy_model_store_path, num_models) is not None:
        dummy_model_store_path = _higher_path(dummy_model_store_path, num_models)
        model_store_dict = read_json_to_dict(dummy_model_store_path)
        model_store = model_store_from_dict(model_store_dict)

        model_snapshots = list(model_store.models.values())
    else:
        # make sure save path exists
        os.makedirs(save_path)
        # generate some dummy snapshots
        model_snapshots = generate_snapshots(architecture_name, num_models, distribution, save_path=save_path)

        # add the snapshots to a model store
        model_store = ModelStore(save_path)
        for snapshot in model_snapshots:
            model_store.add_snapshot(snapshot)

        # save model store to dict for reuse across executions
        model_store_dict = model_store.to_dict()
        write_json_to_file(model_store_dict, dummy_model_store_path)

    return model_snapshots, model_store


def get_architecture_models(base_path, distribution, num_models, architecture_names):
    model_set = architecture_names
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_all_models(base_path, distribution, num_models):
    model_set = VISION_MODEL_CHOICES
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_resnet_models(base_path, distribution, num_models):
    model_set = RESNETS
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_transformer_models(base_path, distribution, num_models):
    model_set = TRANSFORMER_MODELS
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_small_models(base_path, distribution, num_models):
    model_set = [MOBILE_V2, RESNET_34, RESNET_18, EFF_NET_V2_S]
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_medium_models(base_path, distribution, num_models):
    model_set = [RESNET_34, RESNET_50, RESNET_101, EFF_NET_V2_S, VIT_B_16, VIT_B_32]
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def get_large_models(base_path, distribution, num_models):
    model_set = [RESNET_152, EFF_NET_V2_L, VIT_L_16, VIT_L_32]
    return _compose_pregenerates_set(base_path, distribution, model_set, num_models)


def _compose_pregenerates_set(base_path, distribution, model_set, num_models):
    model_counts = distribute_into_buckets(num_models, len(model_set))
    assert max(model_counts) <= 50, "currently 50 is the maximum number of models per architecture"
    agg_snapshots = []
    model_store = ModelStore("")
    for architecture_name, num_models in zip(model_set, model_counts):
        model_snapshots, model_store = generate_snapshot_set(architecture_name, num_models, distribution, base_path,
                                                   reuse_allowed=True)
        model_snapshots = list(model_store.models.values())[:num_models]
        agg_snapshots += model_snapshots
        for snap in model_snapshots:
            model_store.add_snapshot(snap)
    return agg_snapshots, model_store


def _save_path(architecture_name, base_path, distribution, num_models):
    save_path = os.path.join(base_path, architecture_name, str(distribution).replace("RetrainDistribution.", ""),
                             str(num_models))
    return save_path


if __name__ == '__main__':
    distributions = [
        RetrainDistribution.FIFTY_PERCENT,
        RetrainDistribution.TOP_LAYERS,
        RetrainDistribution.TWENTY_FIVE_PERCENT,
    ]
    base_path = '/mount-fs/snapshot-sets'

    model_list = CONVOLUTION_MODELS + [VIT_B_16, VIT_L_32]

    for dist in distributions:
        for model in model_list:
            print('generating', dist, model)
            generate_snapshot_set(model, 50, dist, base_path)
