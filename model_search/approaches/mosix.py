import os

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.models.init_models import initialize_model
from global_utils.global_constants import TRAIN
from global_utils.model_names import RESNET_18
from model_search.approaches.shift import get_data_ranges, prune_snapshots
from model_search.caching_service import CachingService
from model_search.execution.engine.mosix_execution_engine import MosixExecutionEngine
from model_search.execution.planning.baseline_planner import TEST
from model_search.execution.planning.mosix_planner import MosixExecutionPlanner, MosixPlannerConfig
from model_search.model_snapshot import RichModelSnapshot, generate_model_layers


def _dummy_layer_states():
    model = initialize_model(RESNET_18, features_only=True, sequential_model=True)

    layers = generate_model_layers(model, f'/mount-ssd/snapshot-dir/{RESNET_18}')

    return layers


def _generate_dummy_snapshots():
    _model_snapshots = []
    model_name = RESNET_18
    state_dict_path = f'/mount-ssd/snapshot-dir/{model_name}.pt'
    if not os.path.exists(state_dict_path):
        model = initialize_model(model_name, sequential_model=True, features_only=True)
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_path)

    for i in range(4):
        snap = RichModelSnapshot(
            architecture_name=RESNET_18,
            state_dict_path=state_dict_path,
            state_dict_hash=f"HASH-{i}",
            layer_states=_dummy_layer_states()
        )
        _model_snapshots.append(snap)

    return _model_snapshots


if __name__ == '__main__':

    _model_snapshots = _generate_dummy_snapshots()

    model_snapshots = {}
    for snap in _model_snapshots:
        model_snapshots[snap._id] = snap

    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }

    caching_path = '/mount-ssd/cache-dir'
    tensor_caching_service = CachingService(caching_path)
    model_caching_service = CachingService(caching_path)
    planner_config = MosixPlannerConfig(12, 128)
    planner = MosixExecutionPlanner(planner_config)
    exec_engine = MosixExecutionEngine(tensor_caching_service, model_caching_service)

    train_data = CustomImageFolder(dataset_paths[TRAIN])
    ranges = get_data_ranges(len(list(model_snapshots.values())), len(train_data))

    first_iteration = True
    for _range in ranges:
        plan = planner.generate_execution_plan(list(model_snapshots.values()), dataset_paths, _range, first_iteration)
        exec_engine.execute_plan(plan)
        prune_snapshots(model_snapshots, planner, plan)
        first_iteration = False

    print('done')
