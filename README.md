# Sagittarius Stream Stellar Labels from Gaia XP

This project prepares Gaia DR3 BP/RP XP spectra for two related datasets:

1. **Sagittarius application sample:** stars from the Vasiliev et al. Sagittarius catalogue (`J/MNRAS/501/2279`).
2. **APOGEE training sample:** APOGEE DR17 stars with stellar labels and verified Gaia DR3 XP-continuous availability.

The eventual model will predict:

- effective temperature (`TEFF`)
- surface gravity (`LOGG`)
- metallicity (`FE_H`)
- nitrogen-to-oxygen abundance (`N_O = N_FE - O_FE`)
- aluminum-to-iron abundance (`AL_FE`)

## Current status

### Sagittarius sample

- 55,192 catalogue sources
- 42,386 sources with exactly one Gaia DR3 candidate
- 3,628 sources with multiple Gaia DR3 candidates
- 27,069 unambiguous sources with XP continuous spectra
- 25,850 XP sources downloaded
- 1,219 XP sources pending
- 95.50% download completion

### APOGEE training metadata

- 733,901 APOGEE DR17 rows loaded
- 448,728 valid unique Gaia source IDs
- 448,728 IDs verified in Gaia DR3
- 405,160 sources with XP continuous spectra available
- 334,486 sources retained after target-related warning filtering
- 0 duplicate final source IDs

The APOGEE XP coefficients have not yet been downloaded. A test request reached Gaia DataLink but returned HTTP 503 Service Unavailable. The metadata and XP-availability results remain valid, and the download is designed to resume when the service is available.

## Repository layout

```text
.
├── README.md
├── references.bib
├── report_ps5.pdf
├── src/
│   ├── reading_catalog_ps4.ipynb
│   └── build_training_set.ipynb
└── data/
    ├── data.txt
    ├── apogee_training_cutflow.csv
    ├── apogee_training_summary.json
    └── apogee_training_xp_source_ids.csv
```

Large downloaded FITS files, Gaia query batches, and XP batch directories are generated locally and should not be committed to normal Git history.

## Requirements

The notebooks are designed for Python 3.11 and use:

- `numpy`
- `pandas`
- `matplotlib`
- `astropy`
- `astroquery`
- `tqdm`
- `ipywidgets`
- `ipykernel`

VS Code also requires the Microsoft **Python** and **Jupyter** extensions.

## Environment setup

Clone the repository and open it in VS Code:

```bash
git clone https://github.com/LightsRedemption/DeepLearning_TeamAstro
cd DeepLearning_TeamAstro
code .
```

In VS Code, open **Terminal > New Terminal** and run:

```bash
conda create -n sgr-xp python=3.11 -y
conda activate sgr-xp
python -m pip install numpy pandas matplotlib astropy astroquery tqdm ipywidgets ipykernel
python -m ipykernel install --user --name sgr-xp --display-name "sgr-xp"
```

Open a notebook in VS Code, select **Kernel > Select Another Kernel > Python Environments**, and choose `sgr-xp`.

To confirm the selected environment, run:

```python
import sys
print(sys.executable)
```

The printed path should contain `sgr-xp`.

## Running the Sagittarius notebook

Open:

```text
src/reading_catalog_ps4.ipynb
```

Select the `sgr-xp` kernel, then run the cells from top to bottom. The notebook:

1. obtains the 55,192-row Sagittarius catalogue from VizieR;
2. extracts Gaia DR2 source IDs;
3. joins the IDs through `gaiadr3.dr2_neighbourhood` and `gaiadr3.gaia_source`;
4. separates unique and ambiguous DR2-to-DR3 candidates;
5. retains unique DR3 sources with `has_xp_continuous = True`;
6. downloads XP continuous products in restart-safe batches; and
7. saves summaries, missing IDs, crossmatch tables, and combined products.

To resume an interrupted XP download, rerun the setup and crossmatch sections before rerunning the download section. Existing completed batch files will be detected and skipped.

If Gaia DataLink returns HTTP 503, rerun the download section later. Existing completed XP files are scanned and skipped.

## Building the APOGEE DR17 training metadata

Open:

```text
src/build_training_set.ipynb
```

Keep this setting disabled for the metadata-only run:

```python
DOWNLOAD_XP = False
```

Select the `sgr-xp` kernel and run the notebook from top to bottom. Keep `DOWNLOAD_XP = False` for the metadata-only workflow. The saved Gaia query batches are reused on later runs, so successfully completed 40,000-ID queries do not need to be submitted again.

The notebook performs the following steps:

1. downloads or reuses `allStarLite-dr17-synspec_rev1.fits`;
2. verifies the required APOGEE columns;
3. calculates `N_O = N_FE - O_FE`;
4. rejects `STAR_BAD` sources;
5. requires `SNR >= 70`;
6. requires all five targets to be finite;
7. applies `3500 <= TEFF <= 7000` and `0 <= LOGG <= 5`;
8. requires valid unique Gaia source IDs;
9. verifies those IDs in Gaia DR3 using restart-safe 40,000-ID query batches;
10. retains sources with XP continuous spectra available; and
11. removes `TEFF_WARN`, `LOGG_WARN`, `M_H_WARN`, and `N_M_WARN` sources.

`VMICRO_WARN` is recorded but is not used as an exclusion because microturbulence is not a prediction target.

### Verified training metadata results

```text
APOGEE rows loaded:                 733,901
Valid unique Gaia source IDs:       448,728
IDs verified in Gaia DR3:           448,728
IDs not returned:                         0
XP-continuous sources:              405,160
Final warning-filtered sample:      334,486
Duplicate final source IDs:               0
```

### Generated audit files

```text
data/apogee_training_cutflow.csv
data/apogee_training_summary.json
data/apogee_training_xp_source_ids.csv
data/apogee_gaia_unverified_ids.csv
```

The notebook also generates the larger local file:

```text
data/apogee_dr17_gaia_dr3_xp_training_metadata_quality.csv
```

This 66 MB metadata file is generated locally and excluded from Git tracking. It can be regenerated by running the notebook.

## Downloading the APOGEE XP coefficients

The XP download is the final major external data-acquisition step before feature preprocessing and model training.

First, run the 10-source test at the end of `build_training_set.ipynb`. Only continue if the test succeeds and the returned product structure has been inspected.

Then change:

```python
DOWNLOAD_XP = False
```

to:

```python
DOWNLOAD_XP = True
```

The downloader uses the 334,486-source quality-filtered list:

```python
training_meta_quality["source_id"]
```

Downloaded batches are written to:

```text
data/apogee_xp_batches/
```

Failures are recorded in:

```text
data/apogee_xp_download_failures.csv
```

Do not enable the full download while Gaia DataLink is returning HTTP 503. Existing nonempty batch files are skipped when the section is rerun.

## Remaining work before model training

After the APOGEE XP download succeeds:

1. combine all XP FITS batches;
2. verify requested, downloaded, missing, unexpected, and duplicate source IDs;
3. require exactly 55 BP and 55 RP coefficients per source;
4. divide each coefficient group by its first coefficient;
5. remove each normalized first coefficient;
6. concatenate 54 BP and 54 RP values into 108 input features;
7. preserve source-ID, feature, and label alignment;
8. create reproducible 70%/15%/15% train, validation, and test splits using seed 42; and
9. fit feature and target scalers using only the training partition.

## Common issues

### Gaia query returns exactly 50,000 rows

A single uploaded-table query may be capped at 50,000 rows. The training notebook avoids this by querying Gaia in 40,000-ID batches, saving each completed batch, and combining the results.

### HTTP 503 from Gaia DataLink

HTTP 503 indicates temporary maintenance or capacity problems. Keep `DOWNLOAD_XP = False`, retain completed files, and retry later. The metadata and XP-availability query do not need to be repeated if their saved batch files are intact.

### `ModuleNotFoundError`

Activate the environment and reinstall the dependencies:

```bash
conda activate sgr-xp
python -m pip install numpy pandas matplotlib astropy astroquery tqdm ipywidgets ipykernel
```

### Corrupted XP batch

Delete only the unreadable batch and rerun the corresponding download section. Do not delete the entire batch directory, because completed files are used for restart-safe progress.

## Data sources

- VizieR catalogue `J/MNRAS/501/2279`
- APOGEE DR17 `allStarLite-dr17-synspec_rev1.fits`
- Gaia DR3 `gaia_source`
- Gaia DR3 `dr2_neighbourhood`
- Gaia DR3 XP continuous products through Gaia DataLink

## Authors

Giovanny Rosales and Hunter Morgan
