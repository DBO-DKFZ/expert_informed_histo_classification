import argparse
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from matplotlib.patches import Patch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, confusion_matrix


@dataclass
class PlotData:
    title: str
    data: pd.DataFrame
    labels: list[str] = None


def _plot_roc(data: PlotData, ax: plt.Axes, use_softmax=False) -> plt.Axes:
    """
    Plots a ROC curve using scikit-plot. Adds the AUROC value to the legend for each class as well as the macro-averaged
    one. Additionally, allows to apply a softmax function (if passing logits).

    :param data: the data object.
    :param ax: the axis object where to plot the data.
    :param use_softmax: whether to apply a softmax to the predictions.
    :return: the updated axis.
    """
    labels, preds = data.data.iloc[:, -1], data.data.iloc[:, :-1]
    # plots the roc curve with a macro averaged one
    skplt.metrics.plot_roc(labels, preds, title="", plot_micro=False, ax=ax)
    if use_softmax:
        preds = softmax(preds, axis=1)

    # calculate metrics and add to legend
    metrics = roc_auc_score(labels, preds, multi_class="ovr", average=None)

    legend = [f'{data.labels[x]} (area={metrics[x]:.3f})' for x in range(len(metrics))]
    legend.append(f'Average macro-AUROC (area={roc_auc_score(labels, preds, multi_class="ovr", average="macro"):.3f})')

    ax.legend(legend, loc='lower right')
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(data.title, loc="left")

    return ax


def _plot_triangular(data: PlotData, ax: plt.Axes) -> plt.Axes:
    """
    Plots a triangular plot. Expects the data to have 4 columns: three predictions (as logits) and one true label (last
    column).

    :param data: the data object.
    :param ax: the axis object where to plot the data.
    :return: the updated axis.
    """
    # Ensure the data has the correct format
    if data.data.shape[1] != 4:
        raise ValueError("The data must have exactly 4 columns: three predictions and one true label.")

    # Extract the logits and labels
    logits = data.data.iloc[:, :3].values
    true_labels = data.data.iloc[:, 3].values

    # Normalize the logits to probabilities
    probabilities = softmax(logits, axis=1)

    # Compute the triangular coordinates
    x_coords = probabilities[:, 0] + 0.5 * probabilities[:, 1]
    y_coords = np.sqrt(3) / 2 * probabilities[:, 1]

    # Add a triangular boundary
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], "k-")

    # Add labels to the triangle corners
    ax.text(1, -0.04, data.labels[0], ha="center")
    ax.text(0.5, np.sqrt(3)/2 +0.01, data.labels[1], ha="center")
    ax.text(0, -0.04, data.labels[2], ha="center")

    # Scatter the points
    ax.scatter(x_coords, y_coords, c=true_labels, edgecolor="k", s=50)

    # Add a categorical legend
    unique_labels = np.unique(true_labels)
    legend_elements = [Patch(facecolor=plt.cm.viridis(label / max(unique_labels)), edgecolor="k", label=f"{name}") for label, name in enumerate(data.labels)]
    ax.legend(handles=legend_elements, title="True Label")

    # Customize plot
    ax.set_title(data.title, loc='left')
    ax.axis("off")

    return ax


def _plot_confusion(data: PlotData, ax: plt.Axes) -> plt.Axes:
    """
    Plots a confusion matrix. Adds percentages below the actual number (per "true" class predicted correctly). Colours
    represent the percentages (0-100%).

    :param data: the data object.
    :param ax: the axis object where to plot the data.
    :return: the updated axis.
    """
    # Either apply argmax if predictions are passed, or assume the column already contains the prediction (might not hold for binary case!)
    if data.data.shape[1] > 2:
        predictions = data.data.iloc[:, :-1].idxmax(axis=1).astype(int)
    else:
        predictions = data.data.iloc[:, 0]

    cm_raw = confusion_matrix(data.data.iloc[:, -1], predictions)
    ConfusionMatrixDisplay.from_predictions(data.data.iloc[:, -1], predictions, normalize='true', display_labels=data.labels, ax=ax, cmap=plt.cm.Blues, include_values=False, colorbar=False)

    cm_norm = confusion_matrix(data.data.iloc[:, -1], predictions, normalize='true')

    # Force color normalization to 0-1
    ax.images[0].set_clim(0, 1)

    for y in range(cm_norm.shape[0]):
        for x in range(cm_norm.shape[1]):
            ax.text(x, y, f'\n{cm_raw[y, x]}\n{cm_norm[y, x]:.2%}', ha='center', va='center', color="#052360" if cm_norm[y, x] < 0.5 * cm_norm.max() else "white")

    # Modify the color bar to display 0-100% instead of 0-1
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)  # Get the color bar
    # cbar.set_label('Percentage (%)')  # Label the color bar
    cbar.set_ticks(np.linspace(0, 1, 6))  # Set tick positions (e.g., 0, 0.2, ..., 1)
    cbar.set_ticklabels([f'{int(tick * 100)}%' for tick in np.linspace(0, 1, 6)])  # Convert to percentages

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(data.title, loc='left')

    return ax


def _match_numbering(numbering: Optional[str], length: int) -> list[str]:
    """
    Returns a sequence depending on the numbering chosen. Creates an array of length `length`, filled with values
    depending on the numbering system chosen. When the length exceeds 26, "arabic" is used as a fallback option
    (if a valid numbering is chosen, otherwise returns a list of empty strings).
,
    :param numbering: supported are alphabetical (a,b,c,...), Alphabetical (A,B,C,...), arabic (1,2,3,...),
                      roman (i, ii, iii, ...), and Roman (I, II, III, ...). If anything else is passed, returns a list
                      of empty strings.
    :param length: the size of the sequence.
    :return: a list indices.
    """
    roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv',
                      'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi']
    roman_numerals_upper = [r.upper() for r in roman_numerals]

    match numbering:
        case 'alphabetical' if length <= 26:
            index = [chr(ord('a') + x) for x in range(length)]
        case 'Alphabetical' if length <= 26:
            index = [chr(ord('A') + x) for x in range(length)]
        case 'roman' if length <= 26:
            index = roman_numerals[:length]
        case 'Roman' if length <= 26:
            index = roman_numerals_upper[:length]
        case 'arabic' | 'alphabetical' | 'Alphabetical' | 'roman' | 'Roman':
            index = [str(x + 1) for x in range(length)]
        case _:
            index = ["" for _ in range(length)]

    return index


def _get_index_array(numbering: Optional[str], length: int, sub_length: int = 1) -> list[str]:
    """
    Creates an array of length `length`, filled with values depending on the numbering system chosen. When the length 
    exceeds 26, "arabic" is used as a fallback option (if a valid numbering is chosen). Allows for mixed numbering as 
    well, divided by a dash ('-') (e.g. arabic-alphabetical would produce 1a, 1b, ...).

    :param numbering: supported are alphabetical (a,b,c,...), Alphabetical (A,B,C,...), arabic (1,2,3,...),
                      roman (i, ii, iii, ...), and Roman (I, II, III, ...). If anything else is passed, returns a list
                      of empty strings. For mixed formats, use a dash ('-'), e.g. arabic-alphabetical.
    :param length: the size of the primary sequence.
    :param sub_length: the size of the secondary sequence (for mixed formats).
    :return: a list of formatted indices.
    """
    # Determine primary sequence
    primary = _match_numbering(numbering.split("-")[0], length)

    # Determine secondary sequence (for mixed numbering)
    if "-" in numbering:
        secondary = _match_numbering(numbering.split("-")[1], sub_length)
        # Generate combinations
        index = [f"{p}{s}" for p, s in product(primary, secondary)]
    else:
        index = primary

    return index

def _multi_plot(data_2d: list[list[PlotData]], func: Callable = None, scaling: Optional[tuple[float | int, float | int]] = None, **kwargs) -> plt.Figure:
    """
    Plots a 2D grid of func plots. Defers from the data_2d the number of rows and columns to be plotted.

    :param data_2d: a list of list containing the data to plot (corresponds to rows & cols).
    :param func: which function to invoke to plot the data (e.g. _plot_roc, _plot_confusion).
    :param scaling: the size of the figure. If nothing is provided, uses (6, 4.5).
    :param kwargs: kwargs to be passed to the func (besides data & ax).
    :return: the created figure.
    """
    if scaling is None:
        scaling = (6, 4.5)
    rows = len(data_2d)
    cols = len(data_2d[0])
    figure, axes = plt.subplots(rows, cols, figsize=(scaling[0]*cols, scaling[1]*rows))
    axes = axes.reshape(rows, cols)     # ensure axes is 2d

    for row_nr, data_1d in enumerate(data_2d):
        for col_nr, data in enumerate(data_1d):
            func(data=data, ax=axes[row_nr, col_nr], **kwargs)

    figure.tight_layout()

    return figure


def plot_data(rows: list[str], cols: list[str], files: list[list[str | Path]], class_labels: list[str] = None,
              numbering: Optional[str] = None, func: Callable = None, scaling: Optional[tuple[float | int, float | int]] = None, **kwargs) -> plt.Figure:
    """
    Plots data in a 2D grid structure.

    :param rows: the row titles. Will output as "`row` - `col`".
    :param cols: the colum titles. Will output as "`row` - `col`".
    :param files: a 2D list containing the paths to the files to be plotted.
    :param class_labels: labels for the different classes.
    :param numbering: which numbering should be used. They are prefixed before the tiles in bold. Supported are
                      - alphabetical (a,b,c,...), Alphabetical (A,B,C,...),
                      - arabic (1,2,3,...),
                      - roman (i, ii, iii, ...), and Roman (I, II, III, ...).
                      Allows mixed formats (divided by a dash '-'): e.g. arabic-alphabetical (1a, 1b, ...).
                      If anything else is passed, no numbering is used.
    :param func: which plot function should be used (e.g. _plot_roc, _plot_confusion).
    :param scaling: the size of the figure.
    :param kwargs: kwargs to be passed to the func.
    :return: the created figure.
    """
    # create data in 2d structure
    data_2d = []
    # handle indexing
    length = len(rows)*len(cols)
    i = 0
    index = _get_index_array(numbering, length, len(cols))

    for row_idx, row in enumerate(rows):
        data_1d = []
        for col_idx, col in enumerate(cols):
            data_1d.append(PlotData(title=rf'$\bf{{{index[i]}}}$ {row} - {col}', data=pd.read_csv(files[row_idx][col_idx], index_col=0), labels=class_labels))
            i += 1
        data_2d.append(data_1d)

    # plot data
    return _multi_plot(data_2d, func=func, scaling=scaling, **kwargs)



def main(plot: str, input_path: Path, output_path: Path, numbering: str = 'alphabetical'):

    match plot:
        case 'auroc':
            rows = ['Histopath. classifier (majority votes)', 'Histopath. classifier (soft-labels)']
            cols = ['holdout test dataset', 'external test dataset']
            files = [[input_path / 'majority_holdout_temp_predictions.csv', input_path / 'majority_extern_temp_predictions.csv'],
                     [input_path / 'softlabels_holdout_predictions.csv', input_path / 'softlabels_extern_predictions.csv']]

            figure = plot_data(rows, cols, files,
                               class_labels=['Invasive Melanoma', 'Non-invasive melanoma', 'Nevus'],
                               numbering=numbering,
                               func=_plot_roc,
                               scaling=(6, 4.5),
                               use_softmax=True)

            figure.savefig(output_path / 'AUROCHisto.pdf')
            figure.savefig(output_path / 'AUROCHisto.png')

        case 'confusion':
            figure = plot_data(['Histopathological classifier (majority votes)'],
                               ['holdout test dataset', 'external test dataset'],
                               [[input_path / 'majority_holdout_temp_predictions.csv', input_path / 'majority_extern_temp_predictions.csv']],
                               class_labels=['IM', 'NIM', 'Nevus'],
                               numbering=numbering,
                               scaling=(7,5.25), func=_plot_confusion)

            figure.savefig(output_path / 'ConfusionHistoMajority.pdf')
            figure.savefig(output_path / 'ConfusionHistoMajority.png')

            figure = plot_data(['Histopathological classifier (soft-labels)'],
                               ['holdout test dataset', 'external test dataset'],
                               [[input_path / 'softlabels_holdout_predictions.csv', input_path / 'softlabels_extern_predictions.csv']],
                               class_labels=['IM', 'NIM', 'Nevus'],
                               numbering=numbering,
                               scaling=(7,5.25), func=_plot_confusion)

            figure.savefig(output_path / 'ConfusionHistoSoftLabel.pdf')
            figure.savefig(output_path / 'ConfusionHistoSoftLabel.png')

        case 'triangular':
            # majority vote
            figure = plot_data(['Histopathological classifier (majority votes)'],
                               ['uncalibrated model', 'calibrated model'],
                               [[input_path / 'validation' / 'majority' / 'gcn_asap_predictions.csv', input_path / 'validation' / 'majority' / 'gcn_asap_temp_predictions.csv']],
                               class_labels=['Invasive mel.', 'Non-invasive mel.', 'Nevus'],
                               numbering=numbering,
                               func=_plot_triangular,
                               scaling=(6, 6))
            figure.savefig(output_path / 'PredictionsTemperatureScalingMajority.pdf')
            figure.savefig(output_path / 'PredictionsTemperatureScalingMajority.png')
            # soft label
            figure = plot_data(['Histopathological classifier (soft-labels)'],
                               ['uncalibrated model', 'calibrated model'],
                               [[input_path / 'validation' / 'softlabels' / 'transformer_predictions.csv', input_path / 'validation' / 'softlabels' / 'transformer_temp_predictions.csv']],
                               class_labels=['Invasive mel.', 'Non-invasive mel.', 'Nevus'],
                               numbering=numbering,
                               func=_plot_triangular,
                               scaling=(6,6))
            figure.savefig(output_path / 'PredictionsTemperatureScalingSoftLabel.pdf')
            figure.savefig(output_path / 'PredictionsTemperatureScalingSoftLabel.png')
        case _:
            print(f'Unknown plot: {plot}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots figures.")
    parser.add_argument("--plot", required=True, help="What to plot: auroc, confusion, triangular")
    parser.add_argument("--input_path", default="./predictions", help="Path to directory where predictions are stored (default: ./predictions)")
    parser.add_argument("--output_path", default="./plots", help="Path where output will be stored (default: ./plots)")
    parser.add_argument("--numbering", default="alphabetical", help="Numbering style to use (default: alphabetical)")

    args = parser.parse_args()

    main(args.plot, Path(args.input_path), Path(args.output_path), args.numbering)

