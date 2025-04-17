from typing import Optional

import hydra
import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection


class LitModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, loss_module: torch.nn.modules.loss._Loss,
                 optimizer: DictConfig, metrics: Optional[DictConfig] = None, lr_scheduler: Optional[DictConfig] = None) -> None:
        """
        Wraps the torch model into a LightningModule.

        :param model: the (base) torch model.
        :param loss_module: the loss to be used.
        :param optimizer: a DictConfig containing all information about the optimizer (which optimizer, hparams, etc.).
        :param metrics: an optional DictConfig containing a list of all metrics to be used.
        :param lr_scheduler: an optional DictConfig containing all information about the lr scheduler.
        """
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        metrics = MetricCollection(OmegaConf.to_container(hydra.utils.instantiate(metrics))) if metrics else None
        self.train_metrics = metrics.clone(prefix='train_') if metrics else None
        self.val_metrics = metrics.clone(prefix='val_') if metrics else None
        self.test_metrics = metrics.clone(prefix='test_') if metrics else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        A single training step, that computes the forward pass, calculates the loss, and updates the metrics.

        :param batch: the current batch of data.
        :param batch_idx: the index of the current batch.
        :return: the loss.
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)

        # metrics & logs
        batch_size = len(y)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        if self.train_metrics:
            if y.ndim > 1:
                y = y.argmax(dim=-1)
            self.train_metrics(preds, y)
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, 'val', self.val_metrics)

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, 'test', self.test_metrics)

    def _shared_eval(self, batch, batch_idx: int, prefix: str, metric: MetricCollection | None) -> None:
        """
        A shared eval function used by the validation and test steps. Calculates the forward pass, loss and updates the
        corresponding metrics.

        :param batch: the current batch of data.
        :param batch_idx: the index of the current batch.
        :param prefix: the prefix to be used for logging (e.g. 'val', 'test').
        :param metric: the metrics to be computed/updated.
        :return:
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)

        # logs
        batch_size = len(y)
        self.log(f'{prefix}_loss', loss, prog_bar=True, batch_size=batch_size)
        if metric:
            if y.ndim > 1:
                y = y.argmax(dim=-1)
            metric(preds, y)
            self.log_dict(metric, prog_bar=True, batch_size=batch_size)

    def predict_step(self, batch, batch_idx: int):
        x, y = batch
        preds = self.forward(x)

        return preds, y
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Sets up and configures optimizer and learning rate scheduler (if one specified). For the lr scheduler, by
        default the validation loss is used as a monitor.

        :return: the optimizer, or a dict containing the optimizer, lr_scheduler, and the monitor.
        """
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.model.parameters())
        if self.lr_scheduler is None:
            return optimizer
        else:
            lr_scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

