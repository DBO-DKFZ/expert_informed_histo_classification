import hydra
import lightning as L
from omegaconf import DictConfig
from ray import tune

from helper.DatasetUtils import convert_hdf5
from test import test
from train import r_train
from tune import tune_hyper

# Function for setting the seed
L.seed_everything(42, workers=True)


@hydra.main(version_base=None, config_path='../cfg', config_name='config')
def main(cfg: DictConfig) -> None | tune.ResultGrid:
    """
    Depending on the cfg.job, selects the function to execute.

    :param cfg: the configuration file. Is implicitly passed by hydra (see config path in the decorator)
    :return: If tuning: results of the tuning experiment, otherwise: None
    """
    match cfg.job:
        case 'train':
            r_train(cfg)
        case 'tune':
            return tune_hyper(cfg)
        case 'test':
            test(cfg)
        case 'convert_hdf5':
            convert_hdf5(cfg.dataset, **cfg.convert_hdf5)
        case _:
            raise KeyError(f'Unknown mode: {cfg.task}')


if __name__ == '__main__':
    results = main()
