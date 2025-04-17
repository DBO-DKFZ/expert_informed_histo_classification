from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision.transforms import v2

from utils.custom_types import FilePath


class HDF5Base(Dataset):
    def __init__(self, hdf5_file_path: FilePath, train: bool, transform: Optional[v2.Compose] = None, target_transform: Optional[v2.Compose] = None):
        """
        Initializes an HDF5 dataset. Baseclass, does not implement the getitem method. The data isn't loaded either (just the labels).

        :param hdf5_file_path: path to the HDF5 dataset. _train.hdf5/_test.hdf5 are added automatically.
        :param train: whether to load training or test data
        :param transform: image transforms
        :param target_transform: label transforms
        """
        hdf5_file_path = Path(hdf5_file_path)
        if train:
            self.hdf5_file = h5py.File(hdf5_file_path.with_name(hdf5_file_path.stem + '_train').with_suffix('.hdf5'), 'r')
        else:
            self.hdf5_file = h5py.File(hdf5_file_path.with_name(hdf5_file_path.stem + '_test').with_suffix('.hdf5'), 'r')

        self.targets = self.hdf5_file['labels']
        self.names = self.hdf5_file['names'] if 'names' in self.hdf5_file.keys() else None
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.targets)

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return self.length

    def __del__(self):
        """
        Closes the file.

        :return:
        """
        self.hdf5_file.close()


class HDF5Classification(HDF5Base):
    def __init__(self, hdf5_file_path: FilePath, **kwargs):
        """
        Initializes a classification like HDF5 dataset.

        :param hdf5_file_path: path to the HDF5 dataset. _train.hdf5/_test.hdf5 are added automatically.
        :param kwargs: passed to the HDF5Base class (train & transforms)
        """
        super().__init__(hdf5_file_path, **kwargs)

        self.images = self.hdf5_file['images']

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieves a sample from the dataset at given index, and performs transformations (if applicable).

        :param idx: index of the sample
        :return: transformed image and label
        """
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class HDF5GraphClassification(HDF5Base):
    def __init__(self, hdf5_file_path: FilePath, **kwargs):
        """
        Initializes a classification like HDF5 dataset.

        :param hdf5_file_path: path to the HDF5 dataset. _train.hdf5/_test.hdf5 are added automatically.
        :param kwargs: passed to the HDF5Base class (train & transforms)
        """
        super().__init__(hdf5_file_path, **kwargs)

        self.features = self.hdf5_file['features']
        self.adjacency = self.hdf5_file['adjacency']
        self.positions = self.hdf5_file['position'] if 'position' in self.hdf5_file.keys() else None

    def __getitem__(self, idx: int) -> (Data, torch.Tensor):
        """
        Retrieves a sample from the dataset at given index, and performs transformations (if applicable). Wraps the data
        into a torch_geometric data object.

        :param idx: index of the sample
        :return: a torch_geometric data object and the label
        """
        # create the Data object including the features and the adjacency matrix (i.e. edge_index)
        x = torch.from_numpy(np.vstack(self.features[idx]).T)
        edge_index = torch.from_numpy(np.vstack(self.adjacency[idx]))
        target = torch.tensor(self.targets[idx])
        pos = torch.from_numpy(np.vstack(self.positions[idx]).T) if self.positions else None
        data = Data(x=x, edge_index=edge_index, pos=pos)
        # transformations
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        # ensure data is valid
        data.validate()

        return data, target

