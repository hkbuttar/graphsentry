"""
Loaders for the Elliptic Bitcoin transaction dataset.

Source: Kaggle "Elliptic Data Set" (ellipticco/elliptic-data-set), also described in
Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
Networks for Financial Forensics" (2019).

Three raw files, expected under data/raw/:
  - elliptic_txs_features.csv  : 203,769 rows x 167 cols, no header.
                                  col 0 = txId, col 1 = time step (1..49),
                                  cols 2..166 = 165 additional features (166 total
                                  features excluding txId, per the paper's convention
                                  of calling the time step "feature 1").
  - elliptic_txs_edgelist.csv  : 234,355 directed edges, header txId1,txId2.
  - elliptic_txs_classes.csv   : 203,769 rows, header txId,class.
                                  class in {"1" (illicit), "2" (licit), "unknown"}.

The graph is a single snapshot covering 49 discrete time steps. Each transaction
(node) belongs to exactly one time step, and edges only ever connect transactions
in the same or adjacent time steps (a transaction can only spend outputs that
already existed). This is what makes a temporal train/test split (early steps to
train, late steps to test) meaningful instead of arbitrary -- see graph/README
notes in graph_builder.py for how this is used downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"

FEATURES_CSV = RAW_DIR / "elliptic_txs_features.csv"
EDGELIST_CSV = RAW_DIR / "elliptic_txs_edgelist.csv"
CLASSES_CSV = RAW_DIR / "elliptic_txs_classes.csv"

NUM_TIME_STEPS = 49
NUM_RAW_FEATURES = 165  # excluding txId and the time-step column
# Class label mapping used throughout the project: 1 = illicit, 0 = licit, -1 = unknown.
CLASS_MAP = {"1": 1, "2": 0, "unknown": -1}


@dataclass
class EllipticDataset:
    """In-memory holder for the three raw tables plus a merged node table."""

    features: pd.DataFrame  # indexed by txId, columns: time_step, feat_1..feat_165
    edges: pd.DataFrame  # columns: txId1, txId2
    classes: pd.DataFrame  # indexed by txId, column: label (1/0/-1)
    nodes: pd.DataFrame  # merged: time_step, feat_1..feat_165, label


def _load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Elliptic Data Set from Kaggle "
            "(ellipticco/elliptic-data-set) and place the three CSVs under data/raw/."
        )
    col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, NUM_RAW_FEATURES + 1)]
    df = pd.read_csv(path, header=None, names=col_names)
    return df.set_index("txId")


def _load_edges(path: Path = EDGELIST_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


def _load_classes(path: Path = CLASSES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    df["label"] = df["class"].astype(str).map(CLASS_MAP)
    return df.set_index("txId")[["label"]]


def load_elliptic(use_cache: bool = True) -> EllipticDataset:
    """Load the three raw CSVs and build the merged node table.

    If a cached parquet snapshot exists under data/cache/, use it instead of
    re-parsing the ~690MB features CSV (parsing that file from scratch takes
    noticeably longer than reading a parquet file back).
    """
    cache_path = CACHE_DIR / "nodes.parquet"
    if use_cache and cache_path.exists():
        nodes = pd.read_parquet(cache_path)
        features = nodes.drop(columns=["label"])
        classes = nodes[["label"]]
        edges = _load_edges()
        return EllipticDataset(features=features, edges=edges, classes=classes, nodes=nodes)

    features = _load_features()
    edges = _load_edges()
    classes = _load_classes()

    nodes = features.join(classes, how="left")
    nodes["label"] = nodes["label"].fillna(-1).astype(int)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        nodes.to_parquet(cache_path)

    return EllipticDataset(features=features, edges=edges, classes=classes, nodes=nodes)


def sanity_check(ds: EllipticDataset) -> dict:
    """Confirm the dataset matches the documented Elliptic shape and class balance."""
    n_nodes = len(ds.nodes)
    n_edges = len(ds.edges)
    label_counts = ds.nodes["label"].value_counts().to_dict()
    n_illicit = label_counts.get(1, 0)
    n_licit = label_counts.get(0, 0)
    n_unknown = label_counts.get(-1, 0)
    n_labeled = n_illicit + n_licit

    time_steps = sorted(ds.nodes["time_step"].unique())

    report = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_time_steps": len(time_steps),
        "time_step_range": (min(time_steps), max(time_steps)),
        "n_illicit": n_illicit,
        "n_licit": n_licit,
        "n_unknown": n_unknown,
        "pct_unknown": round(100 * n_unknown / n_nodes, 1),
        "pct_illicit_of_labeled": round(100 * n_illicit / n_labeled, 1) if n_labeled else None,
    }
    return report


if __name__ == "__main__":
    ds = load_elliptic()
    report = sanity_check(ds)
    print("Elliptic dataset sanity check")
    print("-" * 40)
    for k, v in report.items():
        print(f"{k:>28}: {v}")
