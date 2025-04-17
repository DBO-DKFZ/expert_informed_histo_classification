from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

from utils.custom_types import FilePath


class SimpleImageDataset(Dataset):
    def __init__(self, data_dir: FilePath, file_type: str, train: Optional[bool] = None, transform: Optional[v2.Compose] = None, target_transform: Optional[v2.Compose] = None, sort: bool = False):
        """
        Initializes the dataset.
        Queries recursively the data_dir with pathlib/rglob

        :param data_dir: path to the dataset
        :param file_type: the file's type
        :param train: just for compatibility, is not used
        :param transform: image transforms
        :param target_transform: label transforms (is not used, as there are no labels, but needed for compatibility)
        """
        super().__init__()

        images = list(Path(data_dir).rglob(f'*{file_type}'))

        self.images = sorted(images) if sort else images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieves a sample from the dataset at given index, and performs transformations (if applicable).

        :param idx: index of the sample
        :return: transformed image and label
        """
        image = read_image(str(self.images[idx]), mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)

        return image, image


class SimpleImageDatasetV2(SimpleImageDataset):
    def __init__(self, images: list[FilePath], **kwargs):
        """
        Initializes the dataset.
        Passes a list containing the images directly.

        :param images: path to the dataset
        :param transform: image transforms
        :param target_transform: label transforms (is not used, as there are no labels, but needed for compatibility)
        """
        super().__init__('None', 'None', sort=False, **kwargs)

        self.images = images

