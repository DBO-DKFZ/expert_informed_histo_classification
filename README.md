# Expert-informed histopathological classifier

This repository provides code to test, train, and tune graph neural networks for histopathological image analysis, and accompanies the paper:

**Expert-informed melanoma classification with dermoscopic and histopathologic data: Results and recommendations for practice**, *Haggenmüller, S.; Heinlein, L.; Abels, J.; et al.*, Conference/Journal TBD, 2025

This code is developed and maintained by [*Heinlein, L.*](https://github.com/V3ritate).

The following sections include instructions for installation, running a demo, and preparing your own dataset for model training, tuning, or evaluation.

The specific configurations for the model trained on one-hot encoded majority votes [`models/HistoMajority.ckpt`](./models/HistoMajority.ckpt) and for the model trained on reviewer uncertainties as soft labels [`models/HistoMajority.ckpt`](./models/HistoSoftLabel.ckpt) are available in [`cfg/config.yaml`](./cfg/config.yaml) and [`cfg/config_soft.yaml`](./cfg/config_soft.yaml), respectively.

## Pipeline overview

![Histopathological Classifier](./images/HistopathologicalClassifier.svg)

## System requirements

The code was tested on an Ubuntu 22.04.5 as well as an Ubuntu 24.04.2 system with an NVIDIA GeForce RTX 2080 TI GPU, using Python 3.11. The complete list of required Python packages, including their version can be found in the [`requirements.txt`](./requirements.txt) file and has been omitted here due to its extensiveness.

## Installation guide

**Note:** an active internet connection is required.

First, if Python 3.11 and `venv` are not already installed, you can install them via `apt`:

```bash
sudo apt install python3.11 python3.11-venv
```

Navigate to the downloaded source code. Then, run the following commands to create a new environment with `venv`, activate it, and install all required packages:

```bash
# Create a new environment '.venv'
python3.11 -m venv .venv
# Activate the virtual environment
. .venv/bin/activate
# Install required packages
pip install -r requirements.txt
```

Installation typically takes only a few minutes, depending on your system and internet connection.

### Troubleshooting

If you're unable to install Python 3.11 via `apt`, you may need to add the Deadsnakes PPA first:

```bash
sudo apt update
sudo apt install software-properties-common
sudo -E add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

## Demo

This project contains five slides from TCGA-SKCM:

- TCGA-BF-A1PU-01Z-00-DX1
- TCGA-BF-A1PV-01Z-00-DX1
- TCGA-BF-A1PZ-01Z-00-DX1
- TCGA-BF-A3DJ-01Z-00-DX1
- TCGA-BF-A3DL-01Z-00-DX1

Due to the original data size, they have already been preprocessed and formatted correctly. They are available in [`data/tcga-demo_test.hdf5`](./data/tcga-demo_test.hdf5).

Before running the demo, modify the [`cfg/config.yaml`](./cfg/config.yaml) file as needed. You can control the number of CPU cores used via `resources.cpu_worker` (set to 4 by default). To run the demo, execute:

```bash
python src/main.py job=test
```

It should output the following (after some debug & logging information):

```bash
{'AUROC_mean': tensor(0.), 'AUROC_std': tensor(0.), 'AUROC_quantile': tensor([0., 0.]), 'Accuracy_mean': tensor(0.3388), 'Accuracy_std': tensor(0.2162), 'Accuracy_quantile': tensor([0.1000, 1.0000]), 'F1Score_mean': tensor(0.3952), 'F1Score_std': tensor(0.1872), 'F1Score_quantile': tensor([0.1667, 1.0000])}
```

Additionally, files are created in the directory containing the model checkpoint (per default: [`models/`](./models/), using the checkpoint's stem as a prefix). These contain all metrics, including confidence intervals, as well as all bootstrapped values used to calculate them. Lightning (default: `lightning_logs`) and Hydra (default: `outputs`) create logging files as well, but these can be deactivated via the [`cfg/config.yaml`](./cfg/config.yaml) file.

Depending on the hardware used, this process should only take a few minutes.

## Instructions for use

### Recreating the plots

To recreate the plots, run the script [`src/utils/plots.py`](./src/utils/plots.py) (set the `--plot` argument to `auroc`, `confusion`, or `triangular`, depending on which plot to generate):

```bash
python -m src.utils.plots --plot auroc
```

Input and output paths, as well as the numbering style, can be adjusted using the `--input_path`, `--output_path` and `--numbering` arguments.
The generated plots (in both `.png` and `.pdf` formats) are also available in the [plots/](./plots/) directory.

### Recreating the measurements

To recreate the measurements, including 95% confidence intervals, run the script [`src/utils/statistics.py`](./src/utils/statistics.py):

```bash
python -m src.utils.statistics
```

### Recreating temperature scaling

To recreate the temperature values, Expected Calibration Errors, including confidence intervals, and associated plots, run the script [`src/utils/TemperatureScaling.py`](./src/utils/TemperatureScaling.py):

```bash
python -m src.utils.TemperatureScaling
```

### General instructions/running on different datasets

To switch to the soft label configuration, change `config_name='config_soft'` in line 15 in [`src/main.py`](./src/main.py#L15).

To create a new dataset from scratch, you need to tile the WSIs. This project uses non-overlapping tiles of size 224x224 with 20x magnification (higher magnification may cause memory constraints). You can use a tiling script such as: [Tile WSI Script](https://github.com/vkola-lab/tmi2022/blob/main/src/tile_WSI.py)

Then, adjust lines 140 and subsequent in [`src/helper/GraphUtils.py`](./src/helper/GraphUtils.py#L140) as needed, and run the script:

```bash
cd src
python -m helper.GraphUtils
```

This also requires a backbone that produces 512-dimensional feature vectors. In this project, the ResNet18 from [this repository](https://github.com/ozanciga/self-supervised-histopathology) was used. Generally, any feature extractor with the same input and output dimensionality should work.

Finally, you can also train (`task=train`), or tune (`task=tune`) a new model after making the necessary adjustments in the [`cfg/config.yaml`](./cfg/config.yaml) file, such as setting `dataset.data_dir`. For details on tuning parameters, refer to the `hyperparamter_tuning` section within the [`cfg/config.yaml`](./cfg/config.yaml).

**Note:** The scripts expect the following naming convention: 

- `_train.hdf5` for training sets
- `_test.hdf5` for test sets.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

