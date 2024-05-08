from abc import ABC, abstractmethod
from enum import Enum

from model_search.execution.data_handling.data_information import DatasetInformation
from model_search.model_snapshots.base_snapshot import ModelSnapshot


class ExecutionStatus(Enum):
    OPEN = 1
    DONE = 2


class CacheLocation(Enum):
    SSD = 1
    CPU = 2
    GPU = 3


class ScoringMethod(Enum):
    FC = 1


class ExecutionStep(ABC):

    def __init__(self, _id: str):
        self._id = _id
        self.execution_status: ExecutionStatus = ExecutionStatus.OPEN
        self.execution_result = None
        self.execution_logs = None


class BaselineExtractFeaturesStep(ExecutionStep):
    def __init__(self, _id: str, model_snapshot: ModelSnapshot, data_info: DatasetInformation,
                 feature_cache_prefix: str, cache_locations=CacheLocation.SSD):
        super().__init__(_id)
        self.model_snapshot: ModelSnapshot = model_snapshot
        self.inp_data: DatasetInformation = data_info
        self.feature_cache_prefix = feature_cache_prefix
        self.cache_locations = cache_locations


class ScoreModelStep(ExecutionStep):

    def __init__(self, _id: str, scoring_method: ScoringMethod, test_feature_cache_prefixes: [str],
                 train_feature_cache_prefixes: [str], num_classes: int):
        super().__init__(_id)
        self.scoring_method: ScoringMethod = scoring_method
        self.test_feature_cache_prefixes: [str] = test_feature_cache_prefixes
        self.train_feature_cache_prefixes: [str] = train_feature_cache_prefixes
        self.num_classes: int = num_classes


class ModifyCacheStep(ExecutionStep):

    def __init__(self, _id: str, cache_evict_ids: [str], move_operations: dict):
        super().__init__(_id)
        self.cache_evict_ids: [str] = cache_evict_ids
        self.move_operations = move_operations


class ExecutionPlan:

    def __init__(self, execution_steps: [ExecutionStep]):
        self.execution_steps: [ExecutionStep] = execution_steps


class ExecutionPlanner(ABC):
    @abstractmethod
    def generate_execution_plan(self, model_snapshots: [ModelSnapshot], dataset_paths: dict) -> ExecutionPlan:
        # here we need to insert all the logic to generate the execution plan
        pass
