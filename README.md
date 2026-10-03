# Sagittarius Gaia XP Catalog Downloader

This notebook builds a Gaia DR3 XP-continuous-spectra sample for stars in the Sagittarius (Sgr) catalog available through VizieR as `J/MNRAS/501/2279`.

The workflow:

1. Downloads the Sgr catalog from VizieR.
2. Extracts the Gaia DR2 source IDs stored in the catalog.
3. Cross-matches Gaia DR2 source IDs to Gaia DR3 using the Gaia Archive `dr2_neighbourhood` table.
4. Separates unambiguous and ambiguous DR2 → DR3 matches.
5. Keeps unambiguous DR3 sources with available XP continuous spectra.
6. Downloads the Gaia DR3 XP continuous coefficients in restart-safe batches.
7. Saves each successful XP batch as a FITS file.

The notebook is designed to tolerate temporary Gaia Archive failures by retrying requests and automatically splitting failed batches into smaller groups.

---

## Requirements

A recent Python 3 installation is recommended. The notebook uses:

- `numpy`
- `pandas`
- `matplotlib`
- `astropy`
- `astroquery`
- Jupyter Notebook or JupyterLab

The standard-library packages `pathlib`, `time`, and `gc` are also used and do not need to be installed separately.

### Install with pip

```bash
python -m pip install numpy pandas matplotlib astropy astroquery jupyterlab
```

### Optional: create a conda environment

```bash
conda create -n sgr-xp python=3.11 numpy pandas matplotlib astropy astroquery jupyterlab
conda activate sgr-xp
```

No local input catalog is required because the Sgr catalog is downloaded directly from VizieR. An internet connection is required for both the VizieR and Gaia Archive queries.

---

## Repository layout

A simple repository can look like:

```text
.
├── reading_catalog.ipynb
├── README.md
└── Vasiliev_Sgr_XP_batches/   # created automatically when XP downloads begin
```

During execution the notebook also creates:

```text
vasiliev_sgr.fits
```

which is a local FITS copy of the VizieR Sgr catalog.

---

## Running the notebook

Clone the repository and enter it:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

Activate the environment containing the required packages, then start JupyterLab:

```bash
jupyter lab
```

Open `reading_catalog.ipynb` and run the cells from top to bottom.

You can also launch the notebook directly with:

```bash
jupyter lab reading_catalog.ipynb
```

If your notebook has a different filename, replace `reading_catalog.ipynb` with that filename.

---

## What each section does

### 1. Download the Sagittarius catalog

The notebook queries VizieR with:

```python
catalogs = viz.get_catalogs("J/MNRAS/501/2279")
```

and loads the first returned table as the Sgr catalog.

The current catalog contains 55,192 rows in the notebook run included with this project.

The notebook also inspects which stars have metallicity measurements and which of those measurements are identified with the APOGEE reference flag used in the catalog.

### 2. Extract Gaia DR2 source IDs

The `SimbadName` column contains values such as:

```text
Gaia DR2 2422822873087610624
```

The notebook extracts the numeric Gaia DR2 identifier and converts it to a 64-bit integer before uploading the IDs to the Gaia Archive.

### 3. Cross-match Gaia DR2 to Gaia DR3

The notebook uploads the DR2 IDs to the Gaia Archive and joins them against:

```text
gaiadr3.dr2_neighbourhood
```

and:

```text
gaiadr3.gaia_source
```

The query returns the DR3 source ID along with useful information including:

- DR2–DR3 angular distance
- magnitude difference
- sky position
- Gaia G, BP, and RP photometry
- BP-RP color
- RUWE
- XP continuous availability
- XP sampled availability

### 4. Remove ambiguous DR2 → DR3 matches

Some Gaia DR2 stars can have multiple DR3 candidate matches. The notebook separates these from stars that have exactly one DR3 match.

The ambiguous matches are inspected with histograms of angular separation and magnitude difference, while the XP download sample uses only the unambiguous matches.

For the run saved in the notebook, the unambiguous sample contains 50,074 DR3 matches.

### 5. Select sources with XP continuous spectra

Only sources satisfying:

```python
has_xp_continuous == True
```

are sent to the XP downloader.

In the current notebook run, this gives 32,468 sources, or about 64.84% of the unambiguous DR3 sample.

### 6. Download Gaia XP coefficients

The downloader uses:

```python
Gaia.load_data(
    ids=...,
    data_release="Gaia DR3",
    retrieval_type="XP_CONTINUOUS",
    data_structure="RAW",
    format="votable"
)
```

The returned VOTables are converted to Astropy tables and saved as FITS files.

---

## Download settings

The main settings are defined near the XP download section:

```python
OUTDIR = Path("Vasiliev_Sgr_XP_batches")
BATCH_SIZE = 250
MIN_BATCH_SIZE = 10
MAX_RETRIES = 3
PAUSE_SECONDS = 2
```

### `OUTDIR`

Directory in which downloaded XP FITS files are stored.

### `BATCH_SIZE`

Number of stars requested in each top-level Gaia request.

The notebook currently uses:

```python
BATCH_SIZE = 250
```

A larger value may be faster when the Gaia Archive is responsive, but smaller batches are generally more robust.

### `MIN_BATCH_SIZE`

If a request repeatedly fails, the code recursively splits the request into smaller groups. It stops splitting once a failed group reaches this size.

### `MAX_RETRIES`

Number of times a request is attempted before the code splits the group into smaller requests.

### `PAUSE_SECONDS`

Delay between successful requests. This reduces the rate at which requests are sent to the Gaia Archive.

---

## Restarting or resuming a download

The downloader is restart-safe.

Before beginning the main download loop, it scans:

```text
Vasiliev_Sgr_XP_batches/xp_*.fits
```

Each existing FITS file is opened and its `source_id` column is collected. Those stars are then removed from the list of targets that still need to be downloaded.

Therefore, if the notebook stops because of:

- a lost internet connection,
- Gaia Archive downtime,
- a Jupyter kernel restart,
- a computer restart, or
- a manually stopped run,

rerun the notebook cells required to recreate `xp_ids`, then rerun the XP download section. Existing stars will be skipped automatically.

You should see output similar to:

```text
Existing batch files: 42
Already downloaded : 10,500
Remaining          : 21,968
```

Do **not** delete `Vasiliev_Sgr_XP_batches/` if you want the notebook to resume from previous progress.

---

## Retry and failure handling

Each XP request is attempted up to `MAX_RETRIES` times.

Between failed attempts, the notebook waits progressively longer before trying again. With the default settings the waits are approximately 10 seconds and then 20 seconds before the final attempt.

If all attempts fail and the group contains more than `MIN_BATCH_SIZE` sources, the group is split in half and both halves are attempted recursively.

For example:

```text
250 -> 125 + 125
```

If a group is already at or below `MIN_BATCH_SIZE` and still cannot be downloaded, those source IDs are added to the in-memory `failed_ids` list.

You can inspect them after the run with:

```python
len(failed_ids)
failed_ids[:20]
```

Note that `failed_ids` is not currently written to disk automatically.

---

## Output files

XP downloads are written to:

```text
Vasiliev_Sgr_XP_batches/
```

Files use names of the form:

```text
xp_<first_source_id>_<last_source_id>_<number_of_rows>.fits
```

For example:

```text
xp_19182943147267968_6736970526664168832_250.fits
```

Each file contains the Gaia DR3 XP continuous coefficients returned for that successful request.

The exact number of rows in a file may be smaller than the number requested. If Gaia omits any requested source IDs, the notebook identifies the missing IDs and recursively attempts to download them again.

---

## Reading the downloaded XP files

A single batch can be opened with Astropy:

```python
from astropy.table import Table

xp = Table.read(
    "Vasiliev_Sgr_XP_batches/xp_19182943147267968_6736970526664168832_250.fits"
)

print(xp.colnames)
print(xp[:5])
```

To find all downloaded batches:

```python
from pathlib import Path

files = sorted(Path("Vasiliev_Sgr_XP_batches").glob("xp_*.fits"))
print(f"Found {len(files)} XP batch files")
```

---

## Recommended `.gitignore`

The Gaia XP files can become large and normally should not be committed directly to GitHub.

Add the following to `.gitignore` if you want the repository to contain the code but not the downloaded data:

```gitignore
# Downloaded Gaia data
Vasiliev_Sgr_XP_batches/
vasiliev_sgr.fits

# Jupyter temporary files
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
```

If you intentionally want to distribute the downloaded catalog products, consider storing them in a dedicated data release rather than normal Git history.

---

## Common issues

### Gaia request fails or times out

This can happen when the Gaia Archive is busy. The notebook already retries failed requests and reduces their size automatically.

If failures are frequent, try lowering:

```python
BATCH_SIZE = 100
```

or increasing:

```python
PAUSE_SECONDS = 5
```

Then rerun the download section. Previously downloaded stars will be skipped.

### `ModuleNotFoundError`

Install the required dependencies:

```bash
python -m pip install numpy pandas matplotlib astropy astroquery jupyterlab
```

### FITS warnings involving `[Fe/H]`

Astropy may display FITS compatibility warnings for the original catalog column named `[Fe/H]` or for long FITS metadata keywords. These warnings are generated while writing the VizieR catalog and do not necessarily indicate that the file was not written successfully.

### A saved XP FITS file cannot be read

`get_already_downloaded()` prints a warning and ignores unreadable files. The associated sources may therefore be requested again when the notebook resumes.

If a batch file is corrupted, remove that individual file and rerun the download section.

---

## Notes

- The notebook uses public VizieR and Gaia Archive services, so network availability can affect execution.
- The counts listed above reflect the run stored in the current notebook and may change if the upstream catalogs or services change.
- Only unambiguous DR2 → DR3 matches are included in the XP download sample.
- The notebook requests **raw Gaia DR3 XP continuous coefficients**, not a wavelength-sampled spectrum.

---

## Data sources

This project accesses:

- VizieR catalog `J/MNRAS/501/2279`
- Gaia DR3 `gaia_source`
- Gaia DR3 `dr2_neighbourhood`
- Gaia DR3 XP continuous spectra through the Gaia DataLink service

When using these data in scientific work, please cite the original VizieR catalog publication and the appropriate Gaia DR3 papers/data-release documentation.

---

## Author

Giovanny Rosales

