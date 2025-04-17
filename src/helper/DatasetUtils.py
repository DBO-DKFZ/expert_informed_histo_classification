import os
import sys

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datasets import SimpleImageDataset
from utils.custom_types import FilePath


def calc_norm(data_dir: FilePath, file_type: str, size: tuple[int, int] = (224, 224), num_channels: int = 3,
              batch_size: int = 128, num_workers: int = 0) -> (torch.tensor, torch.tensor):
    """
    Calculates the normalization values (mean/std) for given data.

    :param data_dir: the path to the data directory
    :param file_type: the filetype. Dot (.) is added automatically, so just specify 'jpg', 'png', etc.
    :param size: to which size to resize the images
    :param num_channels: how many channels the input images have (3 for RGB, 1 for greyscale, etc.)
    :param batch_size: the batch size for the dataloader. Impacts performance.
    :param num_workers: the number of CPU-threads for the dataloader. Impacts performance.
    :return: mean, std
    """
    transform = v2.Compose([v2.ToImage(), v2.Resize(size=size),])# v2.ToDtype(torch.float32, scale=True)])

    dataset = SimpleImageDataset(data_dir=data_dir, file_type=file_type, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean, std = torch.zeros(num_channels), torch.zeros(num_channels)
    total_images = 0

    for images, _ in dataloader:
        mean += images.mean([0, 2, 3]) * images.size(0)
        std += images.std([0, 2, 3]) * images.size(0)
        total_images += images.size(0)

    return mean / total_images, std / total_images


def create_hdf5(dataset: torch.utils.data.Dataset, output_file: FilePath, batch_size: int = 128, num_workers: int = 0, dtype: np.dtype = np.uint8) -> None:
    """
    Creates an hdf5 file for specified dataset.
    IMPORTANT: Saves as channels-first!

    :param dataset: the dataset for which the hdf5 is to be created
    :param output_file: where to store the hdf5
    :param batch_size: batch size of the dataloader
    :param num_workers: number of workers of the dataloader
    :param dtype: the dtype of images
    :return:
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'Creating {output_file}...')
    with h5py.File(output_file, 'w') as hdf5_file:
        length = len(dataset)
        size = tuple(dataset[0][0].size())
        image_dataset = hdf5_file.create_dataset('images', shape=(length, ) + size, dtype=dtype)
        label_dataset = hdf5_file.create_dataset('labels', shape=(length,), dtype=np.uint8)

        idx = 0
        for images, labels in dataloader:
            # swap axes of images
            image_dataset[idx:idx+batch_size] = images.numpy()
            label_dataset[idx:idx+batch_size] = labels.numpy()
            idx += batch_size
            # progress
            percent = (idx / length) * 100
            bar = '█' * (idx * 40 // length) + '-' * (40 - idx * 40 // length)  # Progress bar
            sys.stdout.write(f'\r|{bar}| {percent:.2f}% Complete')
            sys.stdout.flush()
    print('...Done')


def convert_hdf5(dataset: DictConfig, output_path: FilePath, size: tuple[int, int] = (224, 224), **kwargs) -> None:
    """
    Converts a whole dataset (incl. train/test) into an hdf5 file.

    :param dataset: the DictConfig specifying the dataset to be converted
    :param output_path: where to store the hdf5 files
    :param size: to which size the dataset is to be resized to
    :param kwargs: other parameters to be passed to the creation of the hdf5 files
    :return:
    """
    transforms = v2.Compose([v2.ToImage(), v2.Resize(size=tuple(size), antialias=True)])

    dataset = hydra.utils.instantiate(dataset)

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, dataset.data_set.__name__)

    for name, train in zip(['train', 'test'],[True, False]):
        create_hdf5(dataset.data_set(dataset.data_dir, train=train, transform=transforms, **dataset.kwargs),
                    output_file=f'{out}_{name}.hdf5',
                    **kwargs)


def combine_hdf5(input_file1: FilePath, input_file2: FilePath, output_file: FilePath) -> None:
    """
    Combines two hdf5 files into one. Order is [file1, file2].

    :param input_file1: path to input file 1
    :param input_file2: path to input file 2
    :param output_file: path to output file
    :return:
    """
    file1 = h5py.File(input_file1, 'r')
    file2 = h5py.File(input_file2, 'r')

    assert file1.keys() == file2.keys(), f'Keys do not match: {file1.keys()} != {file2.keys()}.'

    with h5py.File(output_file, 'w') as hdf5_file:
        for key in file1:
            length_f1 = file1[key].shape[0]
            length = length_f1 + file2[key].shape[0]
            dataset = hdf5_file.create_dataset(key, shape=(length, ) + file1[key].shape[1:], dtype=file1[key].dtype)
            dataset[:length_f1] = file1[key][:]
            dataset[length_f1:] = file2[key][:]

