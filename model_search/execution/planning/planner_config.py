from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.planning.execution_plan import CacheLocation


class PlannerConfig:
    def __init__(self, num_workers: int, batch_size: int, target_classes: int, dataset_class: DatasetClass,
                 dataset_paths: dict, default_cache_location: CacheLocation):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.target_classes = target_classes
        self.dataset_class = dataset_class
        self.dataset_paths = dataset_paths
        self.default_cache_location = default_cache_location
