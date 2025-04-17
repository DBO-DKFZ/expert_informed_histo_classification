from inspect import signature
from typing import Type, Optional

import lightning as L
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as geom_Dataloader
from torchvision.transforms import v2

from utils.ImbalancedDatasetSampler import ImbalancedDatasetSampler
from utils.custom_types import FilePath
from utils.split import *


class LitDataModule(L.LightningDataModule):
    def __init__(self, data_set: Type[Dataset], data_dir: FilePath | list[FilePath], transform_train: Optional[v2.Compose], transform_test: Optional[v2.Compose],
                 split: str, transform_train_gpu: Optional[v2.Compose] = None, transform_test_gpu: Optional[v2.Compose] = None,
                 imbalanced_sampler: bool = False, batch_size: int = 128, num_workers: int = 0, graph_data: bool = None,
                 kwargs_dataset: DictConfig = DictConfig({}), kwargs_split: DictConfig = DictConfig({}), kwargs_imbalanced_sampler: DictConfig = DictConfig({})) -> None:
        """
        Wraps the dataset into a LightningDataModule.

        :param data_set: the torch.utils.data.dataset to be used
        :param data_dir: where the data is stored/should be downloaded
        :param transform_train: the transforms to apply to the training data (i.e. including augmentation)
        :param transform_test: the transforms to apply to test. Requires at least ToDtype(). Should also include Normalize().
        :param split: the split mode. See utils/split.py for more information.
        :param imbalanced_sampler: whether to use the ImbalancedDataSampler
        :param batch_size: the DataLoader's batch size
        :param num_workers: the DataLoader's number of workers
        :param graph_data: if the dataset is a graph dataset, which requires the Dataloader of torch_geometric
        :param kwargs_dataset: kwargs for the dataset
        :param kwargs_split: kwargs for the split
        :param kwargs_imbalanced_sampler: kwargs for the ImbalancedDataSampler
        """
        super().__init__()
        self.save_hyperparameters(ignore=['transform_train', 'transform_train_gpu', 'transform_test', 'transform_test_gpu'])
        self.transform_train = transform_train
        self.transform_train_gpu = transform_train_gpu
        self.transform_test = transform_test
        self.transform_test_gpu = transform_test_gpu

        self.DataLoader = geom_Dataloader if graph_data else DataLoader

        self.train, self.val, self.test, self.predict = None, None, None, None
        self.sampler = None

    def prepare_data(self) -> None:
        """
        Prepares the data. Currently only downloads the data (if possible), but can also include tokenizing, etc.
        Is called only within a single process (before setup()), and blocks all other workers until finished.

        :return:
        """
        # download
        if 'download' in signature(self.hparams.data_set.__init__).parameters:
            self.hparams.data_set(self.hparams.data_dir, train=True, download=True)
            self.hparams.data_set(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """
        Is executed on every GPU. Creates the datasets to be assigned for train/val/test/predict, including splitting.

        :param stage: the stage (i.e. fit, test, predict)
        :return:
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            full_train = self.hparams.data_set(self.hparams.data_dir, train=True, transform=self.transform_train, **self.hparams.kwargs_dataset)
            full_val = self.hparams.data_set(self.hparams.data_dir, train=True, transform=self.transform_test, **self.hparams.kwargs_dataset)
            if 'stratify' in self.hparams.split:
                stratify = full_train.targets if self.hparams.split.stratify else None
            else:
                stratify = None
            train_idx, val_idx = custom_split(len(full_train), mode=self.hparams.split.mode, stratify=stratify, **self.hparams.kwargs_split)

            self.train = torch.utils.data.Subset(full_train, train_idx)
            self.val = torch.utils.data.Subset(full_val, val_idx)

            if self.hparams.imbalanced_sampler:
                self.sampler = ImbalancedDatasetSampler(dataset=self.train, **self.hparams.kwargs_imbalanced_sampler)

        # Assign test dataset for use in dataloaders
        if stage == 'test':
            self.test = self.hparams.data_set(self.hparams.data_dir, train=False, transform=self.transform_test, **self.hparams.kwargs_dataset)

        if stage == 'predict':
            self.predict = self.hparams.data_set(self.hparams.data_dir, train=False, transform=self.transform_test, **self.hparams.kwargs_dataset)

    def train_dataloader(self) -> DataLoader:
        # If no sampler is present, shuffle data
        if not self.hparams.imbalanced_sampler:
            return self.DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        else:
            return self.DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, sampler=self.sampler)

    def val_dataloader(self) -> DataLoader:
        return self.DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return self.DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return self.DataLoader(self.predict, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """
        Allows batched, gpu-accelerated augmentations.

        :param batch: a batch of data that needs to be altered or augmented
        :param dataloader_idx: the index of the dataloader to which the batch belongs
        :return: the augmented batch (if applicable)
        """
        x, y = batch
        if self.trainer.training:
            if self.transform_train_gpu:
                x = self.transform_train_gpu(x)
        elif self.transform_test_gpu:
            x = self.transform_test_gpu(x)

        return x, y
