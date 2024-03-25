import os

import torch

from custom.models.init_models import initialize_model
from global_utils.global_constants import TRAIN
from global_utils.model_names import RESNET_18
from model_search.caching_service import TensorCachingService
from model_search.execution.engine.baseline_execution_engine import BaselineExecutionEngine
from model_search.execution.planning.baseline_planner import TEST, BaselineExecutionPlanner
from model_search.model_snapshot import ModelSnapshot

if __name__ == '__main__':
    model_name = RESNET_18
    state_dict_path = f'/mount-ssd/snapshot-dir/{model_name}.pt'

    if not os.path.exists(state_dict_path):
        model = initialize_model(model_name, sequential_model=True, features_only=True)
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_path)

    model_snapshots = [
        ModelSnapshot(
            architecture_name=model_name,
            state_dict_path=state_dict_path
        )
    ]

    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }

    caching_path = '/mount-ssd/cache-dir'

    cachingService = TensorCachingService(caching_path)
    planner = BaselineExecutionPlanner()
    exec_engine = BaselineExecutionEngine(cachingService)

    plan = planner.generate_execution_plan(model_snapshots, dataset_paths)
    exec_engine.execute_plan(plan)
    print('done')
