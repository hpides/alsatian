from abc import ABC, abstractmethod
from enum import Enum

from model_search.execution.planning.data_information import DataInfo
from model_search.model_snapshot import RichModelSnapshot, ModelSnapshot


class ExecutionStatus(Enum):
    OPEN = 1
    DONE = 2


class ExecutionStep(ABC):

    def __init__(self):
        self.execution_status: ExecutionStatus = ExecutionStatus.OPEN
        self.execution_result = None


class ExtractFeaturesStep(ExecutionStep):
    def __init__(self, model_snapshot: RichModelSnapshot, data_info: DataInfo, cache_indices: [int] = None,
                 start_processing_idx: int = None):
        super().__init__()
        self.model_snapshot: RichModelSnapshot = model_snapshot
        self.input_data_information: DataInfo = data_info
        self.cache_indices: [int] = cache_indices
        self.start_processing_idx: int = start_processing_idx
        # TODO use this to specify where things should be cached to (e.g. not everything will fit on GPU cache)
        self.cache_location = None


class ScoreModelStep(ExecutionStep):
    def __init__(self, model_snapshot: RichModelSnapshot, cached_features_ids: [str]):
        super().__init__()
        self.model_snapshot: RichModelSnapshot = model_snapshot
        self.cache_indices: [int] = cached_features_ids


class ModifyCacheStep(ExecutionStep):

    def __init__(self, cache_evict_ids: [str], move_operations: dict):
        super().__init__()
        self.cache_evict_ids: [str] = cache_evict_ids
        self.move_operations = move_operations


class ExecutionPlan:

    def __init__(self, execution_steps: [ExecutionStep]):
        self.execution_steps: [ExecutionStep] = execution_steps


class ExecutionPlanner(ABC):
    @abstractmethod
    def generate_execution_plan(self, model_snapshots: [ModelSnapshot]) -> ExecutionPlan:
        # here we need to insert all the logic to generate the execution plan
        pass
