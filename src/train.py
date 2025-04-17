from functools import partial
from typing import Dict

import hydra
import lightning as L
import ray.train
import ray.train.lightning as rtl
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig, OmegaConf, open_dict
from ray.air import RunConfig
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from utils.utils import init_model, init_datamodule, get_callbacks


def r_train(cfg: DictConfig) -> None:
    """
    Trains with rays TorchTrainer, wraps ray_train into the ecosystem.

    :param cfg: the config containing all information
    :return:
    """
    ray_trainer = TorchTrainer(train_loop_per_worker=partial(ray_train, hyperparameters={}, config=cfg),
                               scaling_config=ScalingConfig(num_workers=cfg.resources.num_workers,
                                                            use_gpu=cfg.resources.gpu_worker,
                                                            resources_per_worker={'CPU': cfg.resources.cpu_worker,
                                                                                  'GPU': cfg.resources.gpu_worker}),
                               run_config=RunConfig(**{key: hydra.utils.instantiate(value) if isinstance(value, DictConfig) else value for key, value in cfg.run_config.items()}) if 'run_config' in cfg else None,
                               resume_from_checkpoint=cfg.checkpoint if 'checkpoint' in cfg else None)

    ray_trainer.fit()


def ray_train(hyperparameters: Dict, config: DictConfig) -> None:
    """
    The training loop for the tuning experiment (using a lightning trainer and ray train).

    :param hyperparameters: the tuned hyperparameters
    :param config: all other parameters
    :return: 
    """
    # update config with hyperparameters
    with open_dict(config):
        for x in hyperparameters:
            OmegaConf.update(config, x, hyperparameters[x])

    L.seed_everything(42)
    # create datamodule and model
    datamodule = init_datamodule(config)
    model = init_model(config)

    # initialize trainer callbacks
    default_callbacks = {'LearningRateMonitor': LearningRateMonitor('epoch'),
                         'RayTrainReportCallback': rtl.RayTrainReportCallback()}

    callbacks = get_callbacks(callbacks_config=config.callbacks, default_callbacks=default_callbacks)

    # create the trainer...
    trainer = L.Trainer(devices='auto',
                        accelerator='auto',
                        accumulate_grad_batches=config.accumulate_grad_batches if 'accumulate_grad_batches' in config else 1,
                        callbacks=callbacks,
                        strategy=rtl.RayDDPStrategy(find_unused_parameters=True),
                        plugins=[rtl.RayLightningEnvironment()],
                        enable_progress_bar=False,
                        **config.trainer_params)

    # ...make it compatible with ray train...
    trainer = rtl.prepare_trainer(trainer)
    # ...and train (from checkpoint, if available)
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:   # download to local dir if remote (done by only one worker)
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_dir)
    else:
        trainer.fit(model=model, datamodule=datamodule)
