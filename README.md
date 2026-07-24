# SHy

Implementation of **Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction (SHy)**.

This repository contains code only. MIMIC raw files, preprocessed arrays, checkpoints, training logs, and result images are intentionally excluded from Git.

## Project Structure

```text
SHy/
+-- shy/                 # Python package and training/evaluation code
+-- notebooks/           # MIMIC-III and MIMIC-IV preprocessing notebooks
+-- requirements.txt
+-- .gitignore
+-- README.md
```

Runtime directories are created or populated locally and are ignored by Git:

- `data/`
- `saved_models/`
- `training_logs/`

## Requirements

- Python 3.9.13
- PyTorch 1.13.1
- torch_scatter 2.1.0
- torch_sparse 0.6.16
- torch_geometric 2.2.0
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

`torch_scatter`, `torch_sparse`, and `torch_geometric` must match your PyTorch and CUDA/CPU environment.

## Data Preparation

Create the local data directories:

```bash
mkdir -p data/RAW/MIMIC_III data/RAW/MIMIC_IV data/MIMIC_III data/MIMIC_IV/binary_train_x_slices data/MIMIC_IV/binary_test_x_slices
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data\RAW\MIMIC_III, data\RAW\MIMIC_IV, data\MIMIC_III, data\MIMIC_IV\binary_train_x_slices, data\MIMIC_IV\binary_test_x_slices
```

MIMIC-III raw files expected in `data/RAW/MIMIC_III/`:

- `ADMISSIONS.csv`
- `DIAGNOSES_ICD.csv`
- `D_ICD_DIAGNOSES.csv`
- `icd9.txt`

MIMIC-IV raw files expected in `data/RAW/MIMIC_IV/`:

- `admissions.csv`
- `diagnoses_icd.csv`
- `d_icd_diagnoses.csv`
- `icd9.txt`
- `dump_list_icd9.pkl`

Run preprocessing:

- `notebooks/iii_preprocessing.ipynb`
- `notebooks/iv_preprocessing.ipynb`

## Training

Run from the project root.

MIMIC-III:

```bash
python -u -m shy.main --dataset_name MIMIC_III --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```

MIMIC-IV:

```bash
python -u -m shy.main --dataset_name MIMIC_IV --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```

Checkpoints are saved under `saved_models/`. Metrics, losses, and plots are saved under `training_logs/`.

## Result Aggregation

After training runs finish:

```bash
python -m shy.aggregate_results
```

The script reads run folders under `training_logs/` and prints grouped summary statistics.

## Notes

- MIMIC data is not included because of size and access restrictions.
- Generated `.pkl`, `.npy`, `.pth`, log, and plot files should stay local.
- Linux filesystems are case-sensitive. Keep MIMIC-III filenames exactly as shown above.
