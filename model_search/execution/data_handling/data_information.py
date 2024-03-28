from abc import ABC
from enum import Enum
from typing import Callable, Optional


class DatasetClass(Enum):
    CUSTOM_IMAGE_FOLDER = 1
    CACHED_FEATURES = 2


class DataInfo(ABC):
    def __init__(self, num_workers: int, batch_size: int, transform: Optional[Callable] = None):
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.transform: Optional[Callable] = transform


class DatasetInformation(DataInfo):
    def __init__(self, data_set_class: DatasetClass, dataset_path: str, num_workers: int, batch_size: int,
                 transform: Optional[Callable] = None):
        super().__init__(num_workers, batch_size, transform)
        self.data_set_class: DatasetClass = data_set_class
        self.dataset_path: str = dataset_path


class CachedDatasetInformation(DataInfo):
    def __init__(self, data_prefix: str, num_workers: int, batch_size: int, transform: Optional[Callable] = None):
        super().__init__(num_workers, batch_size, transform)
        self.data_prefix = data_prefix


class CachedData(DataInfo):
    pass
