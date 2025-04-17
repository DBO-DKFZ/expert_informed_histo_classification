from typing import Dict

import hydra
import lightning as L
import ray.train.lightning as rtl
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig

from lit_datamodule import LitDataModule
from lit_models import LitModule


def init_datamodule(config: DictConfig) -> L.LightningDataModule:
    """
    Initializes the datamodule.

    :param config: the config containing all information
    :return: the lightning datamodule
    """
    dataset = hydra.utils.instantiate(config.dataset)

    datamodule = LitDataModule(data_set=dataset.data_set,
                               data_dir=dataset.data_dir,
                               transform_train=dataset.transform_train if 'transform_train' in dataset else print('No training transformation specified.'),
                               transform_test=dataset.transform_test if 'transform_test' in dataset else print('No test transformation specified'),
                               transform_train_gpu=dataset.transform_train_gpu if 'transform_train_gpu' in dataset else None,
                               transform_test_gpu=dataset.transform_test_gpu if 'transform_test_gpu' in dataset else None,
                               split=config.split,
                               imbalanced_sampler=config.imbalanced_sampler.use,
                               batch_size=config.batch_size,
                               num_workers=config.resources.cpu_worker,
                               graph_data=dataset.graph_data if 'graph_data' in dataset else None,
                               kwargs_dataset=dataset.kwargs if 'kwargs' in dataset else {},
                               kwargs_split=config.split.kwargs if 'kwargs' in config.split else {},
                               kwargs_imbalanced_sampler=config.imbalanced_sampler.kwargs if 'kwargs' in config.imbalanced_sampler else {})

    return datamodule


def init_model(config: DictConfig) -> L.LightningModule:
    """
    Initializes the model.

    :param config: the config containing all information
    :return: the lightning module
    """
    model = LitModule(model=hydra.utils.instantiate(config.model),
                      optimizer=config.optimizer,
                      loss_module=hydra.utils.instantiate(config.loss),
                      metrics=config.metrics if 'metrics' in config else print('No additional metrics specified.'),
                      lr_scheduler=config.lr_scheduler if 'lr_scheduler' in config else print('No lr scheduler specified.'))

    return model


def get_callbacks(callbacks_config: DictConfig, default_callbacks: Dict[str, L.pytorch.Callback | rtl.RayTrainReportCallback]) -> list[L.pytorch.Callback | rtl.RayTrainReportCallback]:
    """
    Instantiates the callbacks, and adds default callbacks (if not set manually).

    :param callbacks_config: the config for the callbacks
    :param default_callbacks: the default callbacks to add
    :return: a list of all callbacks
    """
    # instantiate callbacks
    callbacks = []
    if callbacks_config:
        for x in callbacks_config:
            callbacks.append(hydra.utils.instantiate(callbacks_config[x]))

    # add default callbacks
    if default_callbacks:
        for x in default_callbacks:
            callbacks.append(default_callbacks[x]) if x not in callbacks_config else print(f'{x} manually set')

    return callbacks
