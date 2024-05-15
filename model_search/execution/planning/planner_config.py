from model_search.execution.data_handling.data_information import DatasetClass


class PlannerConfig:
    def __init__(self, num_workers: int, batch_size: int, target_classes: int, dataset_class: DatasetClass,
                 dataset_paths: dict):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.target_classes = target_classes
        self.dataset_class = dataset_class
        self.dataset_paths = dataset_paths
