"""
Downloads the Elliptic Data Set from Kaggle via kagglehub and places the
three raw CSVs where data/loader.py expects them (data/raw/).

This exists as its own script (rather than a one-off shell command) so it
can be the first step of Render's build command -- a fresh deploy has no
local copy of the ~700MB raw data, so this has to run before anything else
in the pipeline. Idempotent: skips the download if the files are already
present (e.g. running locally after already downloading once).

--------------------------------------------------------------------------------
Authentication
--------------------------------------------------------------------------------
kagglehub resolves credentials from (in order): an in-process token, then
KAGGLE_USERNAME + KAGGLE_KEY environment variables, then a kaggle.json file
under ~/.kaggle/ (or $KAGGLE_CONFIG_DIR), then Colab secrets. The environment
variable path is what makes this work in a headless build environment like
Render with no browser and no pre-existing credentials file -- generate an
API token at kaggle.com/settings (Account > API > Create New Token) and set
KAGGLE_USERNAME/KAGGLE_KEY as Render secrets (see README Deployment section).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from data.loader import CLASSES_CSV, EDGELIST_CSV, FEATURES_CSV, RAW_DIR

DATASET_HANDLE = "ellipticco/elliptic-data-set"


def download_raw_data() -> None:
    if FEATURES_CSV.exists() and EDGELIST_CSV.exists() and CLASSES_CSV.exists():
        print(f"Raw data already present at {RAW_DIR}, skipping download.")
        return

    print(f"Downloading {DATASET_HANDLE} via kagglehub...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    # kagglehub extracts into a nested folder; find the actual CSVs wherever
    # they landed rather than assuming a fixed relative path.
    source_dir = next(downloaded_path.rglob("elliptic_txs_features.csv")).parent

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for csv_name in ("elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv"):
        shutil.copy(source_dir / csv_name, RAW_DIR / csv_name)

    print(f"Copied 3 CSVs to {RAW_DIR}")


if __name__ == "__main__":
    download_raw_data()
