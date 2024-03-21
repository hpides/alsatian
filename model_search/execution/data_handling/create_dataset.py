import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform
from model_search.execution.data_handling.data_information import DataInfo, DatasetInformation, DatasetClass


def create_dataset_obj(data_info: DataInfo):
    if isinstance(data_info, DatasetInformation):
        if data_info.data_set_class == DatasetClass.CUSTOM_IMAGE_FOLDER:
            data_set = CustomImageFolder(data_info.dataset_path, imagenet_inference_transform)
            data_loader = torch.utils.data.DataLoader(
                data_set, batch_size=data_info.batch_size, shuffle=False, num_workers=data_info.batch_size
            )
            return data_loader

    raise NotImplementedError
