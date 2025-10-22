import ast
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc, accuracy, f1_score, precision, specificity, kl_divergence

RANDOM_STATE = 42


def ci(y_true: pd.DataFrame, y_pred: pd.DataFrame, patient_ids: pd.DataFrame, function: Callable, class_wise: bool = False, n_bootstrap: int = 1000) -> str:
    """
    Calculates the 95% CI for given predictions based on the metric provided via function. Bootstraps/samples based on
    patients, and then gathers respective lesions.

    :param y_true: true labels, index needs to be consistent with y_pred and patient_ids
    :param y_pred: predictions, index needs to be consistent with y_true and patient_ids
    :param patient_ids: the patient ids, index needs to be consistent with y_true and y_pred
    :param function: the function to calculate the metric. It needs to handle the input in the form of a dataframe (with columns y_true, y_pred)
    :param class_wise: whether the function calculates class-wise values
    :param n_bootstrap: the number of bootstraps
    :return: the CI as a string in the form of '.3f-.3f', or '[.3f-.3f, .3f-.3f, ...]' for class-wise results
    """
    # merge all data into one dataframe (for sampling)
    tmp = pd.merge(pd.merge(y_true, y_pred, left_index=True, right_index=True), patient_ids, left_index=True, right_index=True)

    # get unique patients
    unique_patients = pd.Series(patient_ids.unique())

    # bootstrap and accumulate all metric results
    result = []
    for i in range(n_bootstrap):
        sampled_patients = unique_patients.sample(n=len(unique_patients), replace=True, random_state=RANDOM_STATE+i)

        # count how many patients are sampled
        counts = sampled_patients.value_counts()

        merged = tmp.merge(counts, left_on='patientId', right_index=True)

        # repeat rows depending on counts
        sampled_df = merged.loc[merged.index.repeat(merged['count'])].drop(columns='count')

        metric = function(sampled_df.drop(columns='patientId'))

        result.append(metric)

    # convert to pd.Series
    result = pd.Series(result)

    # calculate 95% CIs
    if class_wise:
        ret = '['
        for idx in range(len(result[0])):
            out = result.apply(lambda x: x[idx]).quantile([0.025, 0.975])
            ret += f'{out.iloc[0]:.3f}-{out.iloc[1]:.3f}, '

        ret += ']'
    else:
        out = result.quantile([0.025, 0.975])
        ret = f'{out.iloc[0]:.3f}-{out.iloc[1]:.3f}'

    return ret


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the (macro) Brier score (mean of mse per class).

    :param probs: predicted probabilities
    :param targets: target labels
    :return: the Brier score
    """
    # ensure probs are valid
    if probs.dim() != 2:
        raise ValueError("`probs` must be of shape (N, C)")
    N, C = probs.shape

    # if 1 dim vector (i.e. hard-labels), transform to one hot
    if targets.dim() == 1:
        targets_oh = torch.nn.functional.one_hot(targets.to(int), num_classes=C).float()
    elif targets.shape == probs.shape:
        targets_oh = targets.float()
    else:
        raise ValueError("`targets` must be shape (N,) or (N, C)")

    # Brier score: mse per class, mean of that
    brier = torch.nn.functional.mse_loss(probs, targets_oh, reduction="none").mean(dim=0).mean()
    return brier



function_clswise = lambda x: auroc(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average=None, task='multiclass')
functions = {'AUROC': lambda x: auroc(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
             'Acc.': lambda x: accuracy(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
             'F1': lambda x: f1_score(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
             'Precision': lambda x: precision(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
             'Specificity': lambda x: specificity(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
             'Brier score (majority-vote based)': lambda x: brier_score(torch.nn.functional.softmax(torch.tensor(np.stack(x.iloc[:, 1].tolist())), dim=1), torch.tensor(x.iloc[:, 0].values, dtype=torch.int)),
             }
functions_soft = {'Brier score (soft-label based)': lambda x: brier_score(torch.nn.functional.softmax(torch.tensor(np.stack(x.iloc[:, 1].tolist())), dim=1), torch.tensor(np.stack(x.iloc[:, 0].tolist()))),
                  'KL Divergence': lambda x: kl_divergence(torch.nn.functional.softmax(torch.tensor(np.stack(x.iloc[:, 1].tolist())), dim=1)+1e-12, torch.tensor(np.stack(x.iloc[:, 0].tolist()))+1e-12),
                  }


filenames = Path('./predictions').rglob('*.csv')


for file in filenames:
    print(f'Current file: {file}')
    print('=============================')
    df = pd.read_csv(file, index_col=0)
    df['patientId'] = df.apply(lambda x: x.name.split('-')[0], axis=1)

    # convert predictions to single column
    preds = df[['0', '1', '2']].apply(lambda x: x.tolist(), axis=1)
    preds.name = 'preds'

    patient_ids = df['patientId']

    # calculate each metric + CI that is in functions
    print('Majority-votes as targets')
    for key, func in functions.items():
        print(f"{key}: {func(pd.merge(df['3'], preds, left_index=True, right_index=True))}, "
              f"95% CI {ci(y_true=df['3'], y_pred=preds, patient_ids=patient_ids, function=func)}")

    # calculate class-wise function
    print(f"AUROC class-wise: {function_clswise(pd.merge(df['3'], preds, left_index=True, right_index=True))}, "
          f"95% CI {ci(y_true=df['3'], y_pred=preds, patient_ids=patient_ids, function=function_clswise, class_wise=True)}")

    # calculate functions on soft-labels (if existing)
    if 'soft_labels' in df.columns:
        print('Soft-labels as targets')
        df['soft_labels'] = df['soft_labels'].apply(ast.literal_eval)
        for key, func in functions_soft.items():
            print(f"{key}: {func(pd.merge(df['soft_labels'], preds, left_index=True, right_index=True))}, "
                  f"95% CI {ci(y_true=df['soft_labels'], y_pred=preds, patient_ids=patient_ids, function=func)}")

    print('=============================')
    print('=============================')

