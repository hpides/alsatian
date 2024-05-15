from model_search.execution.data_handling.data_information import DatasetClass


class BaselinePlannerConfig:
    def __init__(self, num_workers: int, batch_size: int):
        self.num_workers = num_workers
        self.batch_size = batch_size


class AdvancedPlannerConfig(BaselinePlannerConfig):
    def __init__(self, num_workers: int, batch_size: int, dataset_class: DatasetClass, dataset_paths: dict):
        super().__init__(num_workers, batch_size)
        self.dataset_class = dataset_class
        self.dataset_paths = dataset_paths
