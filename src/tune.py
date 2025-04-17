from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ray import tune
from ray.air import RunConfig
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from train import ray_train


def tune_hyper(cfg: DictConfig) -> tune.ResultGrid:
    """
    Prepares everything for a tuning experiment, including setting up the search space, parallel ray trainer for torch,
    and the tuner. The tunable hyperparameters are to be listed in the section "hyperparameter_tuning" within the config,
    and are instantiated with hydra.utils.instantiate.

    :param cfg: the config containing all information
    :return: the result of the tuning experiment
    """
    # setting up search space including constants, will be passed to training function
    search_space = {}

    for x in cfg.hyperparameter_tuning:
        parameter = hydra.utils.instantiate(cfg.hyperparameter_tuning[x])
        if type(parameter) == DictConfig:
            # find and replace all 'target_' keys with '_target_'
            result = replace_key(parameter, 'target_')
            while result:
                parameter = result
                with open_dict(cfg):
                    result = replace_key(parameter, 'target_')
            parameter = OmegaConf.to_container(parameter)
        search_space.update({x: parameter})

    #ray.init(num_cpus=cfg.resources.cpu_worker*cfg.resources.num_workers,
    #         num_gpus=cfg.resources.gpu_worker*cfg.resources.num_workers)

    # Trainer for parallel PyTorch training. Note: train_loop_config will be passed by tuner
    ray_trainer = TorchTrainer(train_loop_per_worker=tune.with_parameters(ray_train, config=cfg),
                               scaling_config=ScalingConfig(num_workers=cfg.resources.num_workers,
                                                            use_gpu=cfg.resources.gpu_worker,
                                                            resources_per_worker={'CPU': cfg.resources.cpu_worker,
                                                                                  'GPU': cfg.resources.gpu_worker}),
                               run_config=RunConfig(**{key: hydra.utils.instantiate(value) if isinstance(value, DictConfig) else value for key, value in cfg.run_config.items()}) if 'run_config' in cfg else None,
                               # datasets=, (with ray.train.get_dataset_shard() in worker)
                               resume_from_checkpoint=cfg.checkpoint if 'checkpoint' in cfg else None)

    tuner = tune.Tuner(trainable=ray_trainer,
                       param_space={'train_loop_config': search_space},
                       tune_config=tune.TuneConfig(search_alg=hydra.utils.instantiate(cfg.tune_config.search_alg),
                                                   scheduler=hydra.utils.instantiate(cfg.scheduler),
                                                   **cfg.tune_config.kwargs))

    return tuner.fit()


# Function to find a specific key in a nested DictConfig
def replace_key(config: DictConfig, key_to_find: str, prefix: str = "") -> Optional[DictConfig]:
    """
    Recursively search for a key and replaces it with _target_. Stops on the first occurrence.

    :param config: the config to search
    :param key_to_find: the key to find
    :param prefix: to search nested configs
    :return: the config with the key changed, or None if the key is not found
    """
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key == key_to_find:
            config._target_ = value
            del config[key]
            return config
        elif isinstance(value, DictConfig):
            found = replace_key(value, key_to_find, full_key)  # Recursively search nested configs
            if found:
                config[key] = found
                return config
    return None
