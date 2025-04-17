from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from scipy.optimize import minimize


RANDOM_STATE = 42

def temperature_nll_loss(temperature: float | torch.Tensor, logits: torch.Tensor, target: torch.Tensor):
    """
    Calculates the nll loss based on the temperature value(s) provided.

    :param temperature: the temperature value(s), either one for all, or class-wise.
    :param logits: the logits to scale, with shape [num_samples, num_classes]
    :param target: the true labels, with shape [num_samples].
    :return:
    """
    temperature = torch.tensor(temperature, requires_grad=False)
    scaled_logits = logits / temperature
    log_probs = F.log_softmax(scaled_logits, dim=1)

    return F.nll_loss(log_probs, target=target)


def tune_temperature(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Tunes the temperature per class by minimizing the nll loss.

    :param logits: the logits of the model to calibrate, with shape: [num_samples, num_classes].
    :param targets: the true labels of the predictions.
    :return: the optimized temperatures, with shape: [num_classes]
    """
    num_classes = logits.size(1)

    # Initial temperatures for each class
    initial_temps = torch.ones(num_classes)

    result = minimize(lambda temp: temperature_nll_loss(temp, logits, targets),
                      x0=initial_temps.numpy(),
                      bounds=[(0.1, 100)] * num_classes,
                      method='L-BFGS-B')

    return result.x


def compute_ece_with_confidence(probs: torch.Tensor, targets: torch.Tensor, patient_ids: pd.DataFrame, num_bins: int = 10, n_bootstrap: int = 1000):
    """
    Compute ECE and its confidence intervals using bootstrapping.

    :param probs: Predicted probabilities (after softmax).
    :param targets: Ground-truth labels.
    :param patient_ids: Patient Ids
    :param num_bins: Number of bins to use for calibration.
    :param n_bootstrap: number of bootstraps
    :return: ECE, lower CI, upper CI, all ECE values
    """
    total_samples, num_classes = probs.size()
    task = 'multiclass' if num_classes > 1 else 'binary'

    unique_patients = pd.Series(patient_ids.unique())
    patient_ids = pd.DataFrame(patient_ids, columns=['patientId']).reset_index(drop=True)

    ece_values = []

    for i in range(n_bootstrap):
        # Sample patients with replacement
        sampled_patients = unique_patients.sample(n=len(unique_patients), replace=True, random_state=RANDOM_STATE + i)

        # count how many patients are sampled
        counts = sampled_patients.value_counts()

        merged = patient_ids.merge(counts, left_on='patientId', right_index=True)

        # repeat rows depending on counts
        sampled_df = merged.loc[merged.index.repeat(merged['count'])].drop(columns='count')

        # Collect indices of all samples corresponding to sampled patients
        sampled_indices = sampled_df.index.values

        # Get the subset
        probs_sample = probs[sampled_indices]
        targets_sample = targets[sampled_indices]

        # Compute ECE for this bootstrap sample
        ece_sample = torchmetrics.functional.calibration_error(probs_sample, targets_sample, task=task, num_classes=num_classes, n_bins=num_bins)
        ece_values.append(ece_sample)


    # Stack to tensor
    ece_values = torch.stack(ece_values)

    # Compute quantiles
    lower_q, upper_q = torch.quantile(ece_values, torch.tensor((0.025, 0.975)))

    # return ece_mean, lower_ci, upper_ci, ece_values
    ece_true = torchmetrics.functional.calibration_error(probs, targets, task=task, num_classes=num_classes, n_bins=num_bins)
    return ece_true, lower_q.item(), upper_q.item(), ece_values



mode = {'Majority': Path('./predictions/validation/majority/gcn_asap_predictions.csv'), 'Softlabel': Path('./predictions/validation/softlabels/transformer_predictions.csv')}
# Read in val predictions
for approach, path in mode.items():
    print(f'{approach}')
    print('=============================')
    
    # read in predictions
    out = pd.read_csv(path, index_col=0)

    # Convert to tensors
    preds = torch.tensor(out.iloc[:, 0:-1].values)
    labels = torch.tensor(out.iloc[:, -1].values, dtype=torch.long)
    patients = out.apply(lambda x: x.name.split('-')[0], axis=1)

    # Tune temperatures
    optimized_temps = tune_temperature(preds, labels)
    print(f'Optimized temperatures: {optimized_temps}')

    # Apply temperature scaling
    scaled_logits = preds / torch.tensor(optimized_temps)
    scaled_probs = F.softmax(scaled_logits, dim=1)

    # Compute ECE before scaling
    original_probs = F.softmax(preds, dim=1)
    ece_before, ci_lower_before, ci_upper_before, ece_values_before = compute_ece_with_confidence(original_probs, labels, patients)
    print(f"ECE before temperature scaling: {ece_before:.4f} (95% CI: [{ci_lower_before:.4f}, {ci_upper_before:.4f}])")

    # Compute ECE after scaling
    ece_after, ci_lower_after, ci_upper_after, ece_values_after = compute_ece_with_confidence(scaled_probs, labels, patients)
    print(f"ECE after temperature scaling: {ece_after:.4f} (95% CI: [{ci_lower_after:.4f}, {ci_upper_after:.4f}])")

    # Compute difference CI
    ece_diff = ece_values_after - ece_values_before
    ece_diff_ci = ece_diff.quantile(torch.tensor([0.025, 0.975]))
    print(f'ECE difference 95% CI: [{ece_diff_ci[0]:.4f}, {ece_diff_ci[1]:.4f}]')

    # Create box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot([ece_values_before, ece_values_after], tick_labels=["Before Scaling", "After Scaling"])
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'./plots/ECETemperatureScaling{approach}.pdf')
    plt.savefig(f'./plots/ECETemperatureScaling{approach}.png')

    print('=============================')
    print('=============================')
