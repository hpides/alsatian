from abc import ABC
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
                 train_feature_cache_prefixes: [str], num_classes: int, scored_models=None):
        super().__init__(_id)
        self.scoring_method: ScoringMethod = scoring_method
        self.test_feature_cache_prefixes: [str] = test_feature_cache_prefixes
        self.train_feature_cache_prefixes: [str] = train_feature_cache_prefixes
        self.num_classes: int = num_classes
        self.scored_models = scored_models


class ModifyCacheStep(ExecutionStep):

    def __init__(self, _id: str, cache_evict_ids: [str]):
        super().__init__(_id)
        self.cache_evict_ids: [str] = cache_evict_ids
        # TODO in the future also add move operations


class ExecutionPlan:

    def __init__(self, execution_steps: [ExecutionStep]):
        self.execution_steps: [ExecutionStep] = execution_steps
