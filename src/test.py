from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer
from torchmetrics import BootStrapper, MetricCollection
from torchmetrics.functional import confusion_matrix

from lit_models import LitModule
from utils.utils import init_datamodule, init_model


def test(config: DictConfig) -> None:
    """
    Predicts the output for a test set using the specified model checkpoint. Then, specified metrics (including CIs) are
    computed. Saves predictions and metrics at the location of the model checkpoint.

    :param config: the config containing all information
    :return:
    """
    model = LitModule.load_from_checkpoint(config.checkpoint)
    datamodule = init_datamodule(config)
    trainer = L.Trainer(devices=1)

    predictions = trainer.predict(model=model, datamodule=datamodule)
    preds, y = torch.cat([x[0] for x in predictions]), torch.cat([x[1] for x in predictions])
    # apply temperature scaling
    if config.temp_scaling.use:
        preds /= hydra.utils.instantiate(config.temp_scaling.value) if isinstance(config.temp_scaling.value, BaseContainer) else config.temp_scaling.value
    
    # calculate all metrics with 95% CIs (bootstrapped)
    if 'metrics' in config:
        metrics = MetricCollection(OmegaConf.to_container(hydra.utils.instantiate(config.metrics)))
        for x in metrics:
            metrics[x] = BootStrapper(metrics[x], num_bootstraps=1000, quantile=torch.tensor([0.025, 0.975]), raw=True, sampling_strategy='multinomial')
        metrics(preds, y)
        out = metrics.compute()

        # save metrics
        metrics_df = pd.DataFrame()
        for x in metrics:
            metrics_df[x] = out.pop(f'{x}_raw')

        metrics_df.to_csv(Path(config.checkpoint).parent / f'{Path(config.checkpoint).stem}_metrics_raw.csv')
        with open(Path(config.checkpoint).parent / f'{Path(config.checkpoint).stem}_metrics.txt', 'w') as f:
            f.write(str(out))
        print(out)

    # save predictions
    pd.DataFrame(torch.cat([preds, y.unsqueeze(1)], dim=1), index=[name.decode() for name in names] if names else None).to_csv(Path(config.checkpoint).parent / f'{Path(config.checkpoint).stem}_predictions.csv')

