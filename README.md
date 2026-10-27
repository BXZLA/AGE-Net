# AGE-Net

A PyTorch model for predicting bulk modulus (K) and shear modulus (G) from crystal structures.

## Dependency

Python 3.9, PyTorch 2.8.0, NumPy 1.26.4, pandas 2.3.3 and pymatgen 2024.8.9.

## Installation

Run from the repository root:

```sh
conda env create -f environment.yml
conda activate agenet-paper
```

For an existing Python 3.9 environment, use `pip install -e .`.

## Prediction

Check the data and model weights, then predict the supplied test set:

```sh
python -m agenet.check
python -m agenet.predict --device cpu
```

The output in `outputs/predictions.csv` includes log10 predictions and moduli in GPa.

For new structures, provide a CSV with a `material_id` column and matching `<material_id>.cif` files:

```sh
python -m agenet.predict --input-csv materials.csv --cif-dir cifs --output outputs/predictions.csv
```

## Training

```sh
python -m agenet.train --property K
python -m agenet.train --property G
```

K uses scalar angles; G uses Fourier angle features (F = 5). Training settings are in [config.py](agenet/config.py). Each run saves its weights and logs under `outputs/`. Use a new `--run-name` for repeated runs. Add `--report-tests` to save test predictions after model selection.

## Data

The Materials Project data include 10,225 CIF structures with VRH modulus labels in GPa.

| File | Samples | Use |
|---|---:|---|
| `non_sse_train.csv` | 9,459 | Non-Li training |
| `finetune_train_core.csv` | 536 | Li-containing training |
| `val_SSE.csv` | 115 | Validation and model selection |
| `test_B.csv` | 115 | Test |

CSV files and structures are in `data/splits/`. The Li-containing splits were rebalanced using seed 20260512 and 50,000 candidate partitions; previous split boundaries were not retained. The supplied K/G weights in `checkpoints/` use seed 42.

On Windows CPU, predictions for 14 test structures differ from the stored reference values beyond rounding (maximum log10 differences: K 0.009738; G 0.005159). Two checked cases trace to the choice of equally distant neighbors at the 12-neighbor cutoff.

## Acknowledgement

The crystal-graph convolution and data loading code are adapted from [CGCNN](https://github.com/txie-93/cgcnn). Its original MIT license is retained in [LICENSE_CGCNN](LICENSE_CGCNN).
