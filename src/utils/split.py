from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold


def custom_split(idx_len: int, mode: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates indices for splitting the data into train and validation set depending on the chosen mode.

    :param idx_len: the length of the data to be split
    :param mode: how to split. Available are:
            'KFold':    k-fold cross validation
            'single':   a single split
            'fix':      a fix split (depending on train_len)
    :param kwargs: additional arguments to be passed to the respective function. Differs depending on the mode:
            'KFold':    n_splits: number of splits, k: current iteration, shuffle: whether to shuffle the data (default True)
            'single':   train_frac: the fracture of data used for training set, shuffle: whether to shuffle the data (default True)
            'fix':      train_len
    :return: train_index, val_index
    """
    match mode:
        case 'KFold':
            return kfold(idx_len, **kwargs)
        case 'single':
            return train_test_split(np.arange(idx_len), **kwargs)
        case 'fix':
            return np.arange(kwargs['train_len']), np.arange(kwargs['train_len'], idx_len)
        case _:
            raise ValueError(f'"{mode}" does not match any available mode.')


def kfold(idx_len: int, n_splits: int, k: int, shuffle: bool = True, stratify: Optional[torch.Tensor] = None) -> tuple[np.ndarray, np.ndarray]:
    assert k < n_splits, f'Error in KFold: current fold ({k}) exceeds total folds ({n_splits}).'
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        all_splits = [_ for _ in skf.split(np.arange(idx_len), stratify)]
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        all_splits = [_ for _ in kf.split(np.arange(idx_len))]
    return all_splits[k]

