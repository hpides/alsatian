from abc import ABC, abstractmethod
from enum import Enum

from model_search.execution.data_handling.data_information import DataInfo
from model_search.model_snapshot import RichModelSnapshot, ModelSnapshot


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

    def __init__(self):
        self.execution_status: ExecutionStatus = ExecutionStatus.OPEN
        self.execution_result = None


# class ExtractFeaturesStep(ExecutionStep):
#     def __init__(self, model_snapshot: RichModelSnapshot, data_info: DataInfo, cache_indices: [int] = None,
#                  start_processing_idx: int = None, cache_locations=CacheLocation.SSD):
#         super().__init__()
#         self.model_snapshot: RichModelSnapshot = model_snapshot
#         self.input_data_information: DataInfo = data_info
#         self.cache_indices: [int] = cache_indices
#         self.start_processing_idx: int = start_processing_idx
#         # TODO use this to specify where things should be cached to (e.g. not everything will fit on GPU cache)
#         # set it to dict for fine granular specification
#         self.cache_locations = cache_locations

class BaselineExtractFeaturesStep(ExecutionStep):
    def __init__(self, model_snapshot: RichModelSnapshot, data_info: DataInfo, feature_cache_prefix: str,
                 cache_locations=CacheLocation.SSD):
        super().__init__()
        self.model_snapshot: RichModelSnapshot = model_snapshot
        self.input_data_information: DataInfo = data_info
        self.feature_cache_prefix = feature_cache_prefix
        self.cache_locations = cache_locations


class ScoreModelStep(ExecutionStep):

    def __init__(self, scoring_method: ScoringMethod, test_feature_cache_prefixes: [str],
                 train_feature_cache_prefixes: [str]):
        super().__init__()
        self.scoring_method: ScoringMethod = scoring_method
        self.test_feature_cache_prefixes: [str] = test_feature_cache_prefixes
        self.train_feature_cache_prefixes: [str] = train_feature_cache_prefixes


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
