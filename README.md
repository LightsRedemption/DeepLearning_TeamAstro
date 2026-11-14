# Sagittarius Stream Stellar Labels from Gaia XP

This project transfers stellar labels from APOGEE DR17 to Gaia DR3 BP/RP XP spectra and applies the selected model to candidate members of the Sagittarius stellar stream.

The predicted labels are:

- effective temperature (`TEFF`)
- surface gravity (`LOGG`)
- metallicity (`FE_H`)
- nitrogen-to-oxygen abundance (`N_O = N_FE - O_FE`)
- aluminum-to-iron abundance (`AL_FE`)

## Current results

### APOGEE-Gaia training data

- 733,901 APOGEE DR17 catalogue rows inspected
- 334,486 sources retained after quality, Gaia-ID, XP-availability, and target-warning filtering
- 48,650 XP sources targeted in the completed batch download
- 44,650 unique XP sources downloaded and accepted
- 4,000 sources unavailable after retries
- 108 normalized XP features per source: 54 BP and 54 RP coefficients
- 0 preprocessing rejections among downloaded rows

### Fixed model partitions

| Partition | Sources |
|---|---:|
| Training | 31,255 |
| Validation | 6,697 |
| Test | 6,698 |
| Total | 44,650 |

The partitions use random seed 42 and are saved under `data/splits/`.

### Models compared

1. Ridge regression, regularized linear baseline
2. Extra Trees, nonlinear tree ensemble with 300 trees
3. Baseline MLP, hidden layers `(128, 128)`
4. Deeper MLP, hidden layers `(256, 128, 64)`

The baseline MLP was selected using validation performance. Its best checkpoint occurred at epoch 113 with standardized validation loss 0.218281.

### Selected-model test results

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| `TEFF` | 76.323 K | 115.708 K | 0.9601 |
| `LOGG` | 0.1721 | 0.2518 | 0.9402 |
| `FE_H` | 0.0838 | 0.1281 | 0.8629 |
| `N_O` | 0.0978 | 0.1600 | 0.4787 |
| `AL_FE` | 0.0552 | 0.0837 | 0.6324 |

### Sagittarius application results

- 29,886 XP rows received predictions
- 27,106 predictions matched the final Sagittarius DR3 XP catalogue
- 2,780 predictions were retained separately for provenance auditing
- 319 matched rows were flagged outside at least one training-label range
- 26,787 matched, in-domain rows remain in the conservative science catalogue

The primary science-ready output is:

```text
results/sgr_science_catalog_in_training_domain.csv
```

## Repository layout

```text
.
├── README.md
├── references.bib
├── report_ps6.pdf
├── requirements-ps6.txt
├── src/
│   ├── build_training_set.ipynb
│   ├── reading_catalog_ps4.ipynb
│   ├── combine_training_xp.py
│   ├── create_splits.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_sagittarius.py
├── tests/
│   └── test_preprocessing.py
├── data/
│   ├── splits/
│   ├── apogee_xp_batch_progress.csv
│   └── apogee_xp_download_summary.json
└── results/
    ├── baseline_mlp/
    ├── figures/
    ├── model_comparison.csv
    ├── model_selection_summary.json
    ├── sgr_predictions_with_catalog.csv
    └── sgr_science_catalog_in_training_domain.csv
```

Large FITS batches, combined FITS products, and `data/model_ready_splits.npz` are generated locally and excluded from normal Git tracking.

## Environment setup

The project was run with Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ps6.txt
```

Alternatively, with Conda:

```bash
conda create -n sgr-xp python=3.11 -y
conda activate sgr-xp
python -m pip install -r requirements-ps6.txt
```

For notebook use in VS Code:

```bash
python -m ipykernel install --user --name sgr-xp --display-name "sgr-xp"
```

## Data construction

### Sagittarius catalogue and XP products

Open and run:

```text
src/reading_catalog_ps4.ipynb
```

The notebook obtains the Vasiliev Sagittarius catalogue, maps Gaia DR2 identifiers to Gaia DR3, separates unique and ambiguous matches, retains unique DR3 sources with XP continuous spectra, and downloads XP products in restart-safe batches.

### APOGEE training metadata and XP download

Open and run:

```text
src/build_training_set.ipynb
```

The notebook:

1. loads APOGEE DR17 `allStarLite-dr17-synspec_rev1.fits`;
2. computes `N_O = N_FE - O_FE`;
3. applies the quality and target-warning policy;
4. verifies Gaia source IDs;
5. requires XP continuous availability;
6. constructs the stratified download selections; and
7. downloads XP products in restart-safe batches.

Do not rerun completed download cells unless missing batches are intentionally being retried.

## Combining downloaded APOGEE XP batches

From the repository root:

```bash
python src/combine_training_xp.py --batch-dir data/apogee_xp_batches_stratified_core_50k --pattern 'apogee_xp_*.fits' --requested data/apogee_training_stratified_100k.csv --output data/apogee_dr17_xp_training_combined.fits --source-col source_id --bp-col bp_coefficients --rp-col rp_coefficients
```

The completed run produced 44,650 unique rows with valid 55-element BP and RP coefficient arrays.

## Creating fixed splits

The metadata file must contain only sources present in the combined XP data. The completed local file was:

```text
data/apogee_training_available_44650.csv
```

Create the deterministic source-ID splits with:

```bash
python src/create_splits.py --input data/apogee_training_available_44650.csv --output-dir data/splits --seed 42
```

Expected counts:

```text
train: 31,255
validation: 6,697
test: 6,698
```

## Model-ready data

`src/preprocessing.py` validates each 55+55 coefficient pair and transforms it into 108 features by dividing coefficients 1 through 54 by coefficient 0 independently for BP and RP. The resulting arrays and five targets are stored locally in:

```text
data/model_ready_splits.npz
```

Expected NPZ keys are:

```text
X_train
y_train
train_source_ids
X_validation
y_validation
validation_source_ids
X_test
y_test
test_source_ids
target_names
```

## Training models

All commands below assume `data/model_ready_splits.npz` exists.

### Ridge

```bash
python src/train_model.py --model ridge --data data/model_ready_splits.npz --output-root results --ridge-alpha 1.0 --seed 42
```

### Extra Trees

```bash
python src/train_model.py --model extra_trees --data data/model_ready_splits.npz --output-root results --extra-trees-estimators 300 --extra-trees-max-features sqrt --extra-trees-min-samples-leaf 2 --extra-trees-jobs -1 --seed 42
```

### Baseline MLP

On Apple Silicon with MPS:

```bash
python src/train_model.py --model baseline_mlp --data data/model_ready_splits.npz --output-root results --max-epochs 200 --batch-size 256 --checkpoint-epochs 10,25,50,100,200 --device mps --seed 42
```

On a CUDA system, replace `--device mps` with `--device cuda`. For automatic device selection, use `--device auto`.

### Deeper MLP

```bash
python src/train_model.py --model deeper_mlp --data data/model_ready_splits.npz --output-root results --max-epochs 200 --batch-size 256 --checkpoint-epochs 10,25,50,100,200 --device mps --seed 42
```

Both MLPs use Adam, standardized training features and targets, validation-based early stopping, scheduled checkpoints, and a separately saved best-validation checkpoint.

## Sagittarius inference

The selected baseline MLP requires:

```text
results/baseline_mlp/best_model.pt
results/baseline_mlp/feature_scaler.joblib
results/baseline_mlp/target_scaler.joblib
```

Run inference with:

```bash
python src/predict_sagittarius.py --input data/vasiliev_sgr_xp_combined.fits --model results/baseline_mlp/best_model.pt --feature-scaler results/baseline_mlp/feature_scaler.joblib --target-scaler results/baseline_mlp/target_scaler.joblib --output results/sgr_preliminary_predictions.csv --device mps
```

The prediction output contains:

```text
source_id
pred_TEFF
pred_LOGG
pred_FE_H
pred_N_O
pred_AL_FE
```

Predictions outside the training-label ranges are retained with quality flags but excluded from the conservative science catalogue.

## Tests

Run:

```bash
python -m pytest -q
```

Current result:

```text
11 passed
```

## Report

The PS6 report is:

```text
report_ps6.pdf
```

The report includes the dataset construction, preprocessing, model comparison, held-out MAE/RMSE/R2 results, selected baseline MLP, Sagittarius inference accounting, and limitations.

## Important generated files

### Compact files retained in Git

```text
data/splits/*.csv
results/model_comparison.csv
results/model_selection_summary.json
results/baseline_mlp/best_model.pt
results/baseline_mlp/feature_scaler.joblib
results/baseline_mlp/target_scaler.joblib
results/figures/*.png
results/sgr_predictions_with_catalog.csv
results/sgr_science_catalog_in_training_domain.csv
```

### Large local files not normally committed

```text
data/apogee_xp_batches_stratified_core_50k/
data/apogee_dr17_xp_training_combined.fits
data/model_ready_splits.npz
results_local_test/
```

## Common issues

### Gaia DataLink HTTP 503 or closed connection

Gaia DataLink can be temporarily unavailable. Keep completed batch files and rerun only the restart-safe download section. Existing nonempty batches are detected and skipped.

### No XP files found

The notebooks run from `src/`, while the data directory is at the repository root. From a notebook in `src/`, use `../data/...`. From the repository root, use `data/...`.

### Missing Python packages

Activate the intended environment and reinstall:

```bash
source .venv/bin/activate
python -m pip install -r requirements-ps6.txt
```

### MPS unavailable

Verify Apple GPU support:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())
PY
```

Use `--device cpu` if MPS is unavailable.

## Data sources

- VizieR catalogue `J/MNRAS/501/2279`
- APOGEE DR17 `allStarLite-dr17-synspec_rev1.fits`
- Gaia DR3 `gaia_source`
- Gaia DR3 `dr2_neighbourhood`
- Gaia DR3 XP continuous products through Gaia DataLink

## Authors

Giovanny Rosales and Hunter Morgan
