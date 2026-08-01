"""
Feature engineering: merges the 166 raw Elliptic features with the structural
features from Step 3 (pagerank, degree, clustering, betweenness, community),
handles unknown-labeled nodes, and produces the temporal train/test split.

--------------------------------------------------------------------------------
Unknown-labeled nodes: kept in the full table, dropped for supervised training
--------------------------------------------------------------------------------
77% of nodes have label "unknown" (not licit, not illicit -- just never
classified by Elliptic's team). There's no ground truth to train or evaluate
against for these, so:
  - `build_feature_table()` keeps ALL 203,769 nodes, unknowns included.
  - `get_labeled_subset()` drops them before Steps 5/6 (XGBoost, GraphSAGE)
    train, because you cannot compute precision/recall/F1 against a label
    that doesn't exist. This is a real loss of data -- 77% of the graph is
    invisible to the supervised models -- called out again in the README.
  - `features/pseudo_label.py` is a separate, self-contained attempt to
    recover some signal from these nodes via self-training. It does NOT feed
    back into Steps 5/6 -- see that module's docstring for why keeping it
    separate matters.

--------------------------------------------------------------------------------
Temporal train/test split: time steps 1-34 train, 35-49 test
--------------------------------------------------------------------------------
This is the split used in the original Elliptic paper (Weber et al. 2019) and
most published work on this dataset, so results here are comparable to that
literature. Steps 1-34 (~69% of the timeline) train, steps 35-49 (~31%) test.

Checked empirically (not assumed): the illicit rate is NOT stable across the
split. Among labeled nodes, train is 11.6% illicit; test is only 6.5% illicit.
That's a real distribution shift baked into this dataset -- the true label
rate declines over the tracked period. It means:
  - Test is a harder, rarer-positive regime than train -- don't expect
    train-time and test-time metrics to land in the same range.
  - A random split would have masked this shift entirely by mixing time steps
    together, which is exactly the kind of leakage a temporal split is meant
    to catch instead of hide.
This is reported plainly here and again in the Results section, rather than
tuned around.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.loader import load_elliptic
from graph.analytics import build_structural_feature_table

CACHE_DIR = Path(__file__).parent / "cache"
FULL_TABLE_CACHE_PATH = CACHE_DIR / "full_features.parquet"

# Matches the Elliptic paper's convention so results are comparable to
# published baselines: steps 1-34 train, 35-49 test.
DEFAULT_TRAIN_MAX_STEP = 34


def build_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """Merge raw Elliptic features with structural features into one table.

    Every node (label unknown or not) is kept here -- see module docstring.
    `community` is cast to a pandas 'category' dtype rather than left as a
    plain int: community IDs are arbitrary labels (community 42 is not "more"
    of anything than community 7), so treating them as an ordered number would
    let a model invent a meaningless ordering. XGBoost's native categorical
    support (Step 5) reads this dtype directly.
    """
    if use_cache and FULL_TABLE_CACHE_PATH.exists():
        return pd.read_parquet(FULL_TABLE_CACHE_PATH)

    ds = load_elliptic()
    structural = build_structural_feature_table()

    raw = ds.nodes.drop(columns=["label"])
    table = raw.join(structural).join(ds.nodes[["label"]])
    table["community"] = table["community"].astype("category")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_parquet(FULL_TABLE_CACHE_PATH)

    return table


def get_labeled_subset(table: pd.DataFrame) -> pd.DataFrame:
    """Drop unknown-labeled nodes, for the supervised classification task."""
    return table[table["label"] >= 0].copy()


def temporal_train_test_split(
    table: pd.DataFrame, train_max_step: int = DEFAULT_TRAIN_MAX_STEP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split LABELED nodes by time step: early steps train, later steps test.

    Deliberately not a random shuffle -- see module docstring. Caller is
    expected to have already run get_labeled_subset(); this function doesn't
    do it implicitly so it's obvious at the call site whether unknowns are in
    play.
    """
    train = table[table["time_step"] <= train_max_step].copy()
    test = table[table["time_step"] > train_max_step].copy()
    return train, test


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print class balance for each side of the split -- makes the illicit-rate
    shift described in the module docstring visible every time this runs."""
    for name, df in [("train", train), ("test", test)]:
        n_illicit = (df["label"] == 1).sum()
        n_licit = (df["label"] == 0).sum()
        n = len(df)
        steps = (df["time_step"].min(), df["time_step"].max())
        print(f"{name:>5}: n={n}, illicit={n_illicit} ({100*n_illicit/n:.1f}%), "
              f"licit={n_licit}, time_steps={steps}")


if __name__ == "__main__":
    full_table = build_feature_table()
    print(f"Full feature table: {full_table.shape[0]} rows, {full_table.shape[1]} columns")
    print(f"Columns: {list(full_table.columns)[:5]} ... "
          f"{list(full_table.columns)[-6:]}")

    labeled = get_labeled_subset(full_table)
    n_dropped = len(full_table) - len(labeled)
    print(f"\nDropped {n_dropped} unknown-labeled nodes "
          f"({100*n_dropped/len(full_table):.1f}%) for the supervised task; "
          f"{len(labeled)} remain.")

    train_df, test_df = temporal_train_test_split(labeled)
    print(f"\nTemporal split (train: steps 1-{DEFAULT_TRAIN_MAX_STEP}, "
          f"test: steps {DEFAULT_TRAIN_MAX_STEP+1}-49):")
    split_summary(train_df, test_df)
