import os.path
import random
import shutil
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from global_utils.deterministic import check_deterministic_env_var_set


class CustomImageFolder(ImageFolder):
    """
    This extends the standard PyTorch ImageFolder by adding the option to artificially size the dataset
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            number_images: int = 0,
            return_samples_only=True,
    ):
        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
        )
        if number_images > 0:
            self.imgs = random.choices(self.imgs, k=number_images)
            self.samples = self.imgs

        if check_deterministic_env_var_set():
            random.seed(42)

        random.shuffle(self.samples)
        self.all_samples = self.samples
        self.paths_only = False
        self.return_samples_only = return_samples_only

    def set_subrange(self, from_index, to_index):
        if to_index > len(self.samples):
            raise ValueError(f"the 'to_index' ({to_index}) exceeds the size of the dataset ({len(self.samples)})")
        self.samples = self.all_samples[from_index:to_index]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        if self.paths_only:
            return path, target

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_samples_only:
            return sample, target
        else:
            return (path, sample), target


def create_sub_dataset(dataset_path: str, size: int):
    random.seed(42) # to guarantee same split for same input dataset
    image_folder = CustomImageFolder(dataset_path)
    new_dataset_base_path = dataset_path + f'-{size}'
    if not os.path.exists(new_dataset_base_path):
        images_paths = [x[0] for x in image_folder.imgs]
        subset_image_paths = random.sample(images_paths, size)
        for img_path in subset_image_paths:
            new_path = img_path.replace(dataset_path, new_dataset_base_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy2(img_path, new_path)

    new_image_folder = CustomImageFolder(new_dataset_base_path)
    assert len(new_image_folder) == size

    return new_dataset_base_path
