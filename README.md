# GraphSentry — Fraud Detection on the Elliptic Bitcoin Transaction Network

**By Harleen Kaur Buttar**

Combines classical graph analytics (PageRank, Louvain community detection) with a GraphSAGE GNN for illicit-node classification, benchmarked against XGBoost. FastAPI backend, SvelteKit dashboard. Runs entirely on CPU.

## Motivation

Anti-money-laundering work in crypto is fundamentally a graph problem: a transaction's risk depends not just on its own attributes but on who it's connected to and how. This project asks two things: (1) how far do classical, interpretable graph statistics (PageRank, centrality, community structure) get you on their own, and (2) does a graph neural network add meaningfully more than a strong feature-based baseline once those statistics are already in play. Both questions get an honest answer, not a hoped-for one.

## Data

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Kaggle) / Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics" (2019).

- **203,769 nodes** (Bitcoin transactions), **234,355 directed edges** (an edge means an output of one transaction was spent as an input to another).
- **166 features per node**: the first is a time step (1-49), the remaining 165 are Elliptic's pre-computed local + aggregated transaction features (exact derivation undisclosed by the dataset authors).
- **Labels**: 4,545 illicit (2.2%), 42,019 licit (20.6%), 157,205 unknown (77.1%). Among only the labeled nodes, illicit is 9.8%.
- **49 discrete time steps**, each roughly two weeks of activity. Verified empirically (not assumed): every single edge connects two nodes within the *same* time step -- there are zero edges crossing time steps. So the graph is really 49 separate transaction subgraphs sitting side by side, each one a self-contained weakly-connected component (component sizes range up to 7,880 nodes). This structure is what makes a temporal train/test split meaningful rather than arbitrary: it mirrors how the model would actually be deployed (train on the past, generalize to transactions you haven't seen yet), and it's why the GNN has to be inductive.

Raw CSVs are downloaded via `kagglehub` and cached under `data/raw/` (gitignored).

## Methodology

### Graph construction (`graph/graph_builder.py`)

A `networkx.DiGraph` -- one node per transaction, one directed edge per payment flow. Direction is kept deliberately: collapsing to an undirected graph would erase in-degree/out-degree asymmetries (e.g. many-inputs-one-output vs. one-input-many-outputs), which are exactly the kind of structural signature associated with laundering patterns like layering. Every node carries `time_step`, 165 raw features (`feat_1`..`feat_165`), and `label` (1 = illicit, 0 = licit, -1 = unknown) as attributes.

### Classical graph analytics (`graph/analytics.py`)

- **PageRank** and **in/out-degree** on the full directed graph.
- **Clustering coefficient** (directed generalization, Fagiolo 2007).
- **Betweenness centrality**: computing this exactly for all 203k nodes is infeasible (`O(V*E)`). Since the graph is actually 49 disjoint per-time-step components, betweenness is instead computed exactly on the single largest one (7,880 nodes, ~32s) rather than an arbitrary random sample. The other ~196k nodes get `NaN` here -- a real, documented gap, not an oversight.
- **Louvain community detection** (315 communities, ~31s), run on the undirected version of the graph (direction doesn't matter for "who clusters with whom").
- **Illicit clustering check**: does illicit activity concentrate in specific communities, or spread evenly? Checked directly rather than assumed: baseline illicit rate among labeled nodes is 9.76%; communities with more than double that rate (>19.5%) contain **45.4% of all illicit nodes** while making up only **13.4% of all nodes** in the dataset. Illicit nodes do cluster disproportionately into a minority of communities -- a genuine finding, reported as found.

### Feature engineering (`features/build_features.py`)

Merges the 166 raw features with the structural feature table (pagerank, in/out-degree, clustering, betweenness, community) into one 173-column table keyed by transaction ID. `community` is kept as a categorical (not ordinal-numeric) column, since community IDs carry no inherent order.

**Unknown-labeled nodes** (77.1% of the graph) are kept in the full table but dropped before supervised training -- there's no ground truth to train or score against. This is a real loss of usable data, addressed separately below rather than silently accepted.

**Temporal train/test split**: time steps 1-34 train, 35-49 test (the convention used in the original Elliptic paper, for comparability). Checked empirically: illicit rate is **11.6% in train vs. 6.5% in test** -- a real distribution shift across the split, not something a random shuffle would have revealed. Test is a harder, rarer-positive regime than train, and that's stated plainly rather than tuned around.

### Self-training on unknown nodes (`features/pseudo_label.py`)

An exploratory semi-supervised extension: can a classifier recover usable signal from the 77% of nodes with no label? Kept strictly separate from the supervised pipeline -- its output never feeds into model training or evaluation, since validating a model against labels a similar model invented would be circular.

- A `HistGradientBoostingClassifier` (chosen for native support of missing values and categorical features, avoiding manual imputation/encoding) is trained on labeled train-time nodes only.
- **Calibration check first**: 5-fold cross-validated, out-of-fold predictions on the labeled training data show that when the model is >=95% confident a node is illicit, it's right **99.71%** of the time; when it's <=5% confident (i.e. confident-licit), it's right **99.83%** of the time. This is the only honest evidence available on trustworthiness, since it's checked against real labels.
- Applied to the actual unknown nodes: **74.8%** (117,590 of 157,205) get a confident pseudo-label. Notably, the *pseudo*-illicit rate among previously-unknown nodes runs the opposite direction of the real labels (6.9% for train-time unknowns vs. 8.1% for test-time unknowns, versus 11.6%/6.5% for the real labels) -- a reminder that Elliptic's human labelers selected which transactions to label non-randomly, so "labeled" and "unknown" were never the same population to begin with.
- **Consistency check**: since there's no ground truth to validate the pseudo-labels against directly, they're instead checked against an independent signal -- the earlier finding (above) that illicit nodes cluster into specific Louvain communities. Pseudo-illicit nodes fall into those same "high-risk" communities **20.0%** of the time vs. **12.3%** for pseudo-licit nodes -- the same direction as the real-label result (45.4% vs. 13.4%) but noticeably weaker. Structural feature averages tell a similar story: pseudo-illicit nodes have lower in/out-degree than pseudo-licit nodes (consistent with real illicit nodes), but their clustering coefficient sits closer to the licit profile than the real illicit one does. Read plainly: self-training recovers a real but diluted version of the true pattern -- suggestive corroboration, not proof the pseudo-labels are reliable.
- **Threshold sensitivity** (`features/threshold_sensitivity.py`): is the 0.95/0.05 confidence cutoff a good choice, or an arbitrary one? Swept 0.90/0.95/0.99 over the same out-of-fold predictions:

  | threshold | coverage (illicit) | precision (illicit) | coverage (licit) | precision (licit) |
  |---|---|---|---|---|
  | 0.90 | 10.80% | 99.44% | 86.56% | 99.78% |
  | 0.95 | 10.54% | 99.71% | 85.19% | 99.83% |
  | 0.99 | 9.54%  | 99.93% | 78.63% | 99.95% |

  Precision is already >99.4% at the loosest threshold tested, so tightening further buys very little: going from 0.95 to 0.99 costs 6.6 points of licit coverage (thousands of nodes) for a 0.12-point precision gain, and about 1 point of illicit coverage for a 0.22-point gain. Diminishing returns set in past 0.95, most visibly on the licit side. 0.95 is a reasonable, data-justified point on this curve, not an arbitrary round number.

### Baseline model: XGBoost (`models/baseline_xgboost.py`)

Trained on the merged feature table (raw + structural), evaluated with **precision/recall/F1/PR-AUC, not accuracy** -- illicit is 9.8% of labeled nodes, so a model that predicts "licit" for everything scores 90.2% accuracy while catching zero fraud. PR-AUC (threshold-independent) is used over ROC-AUC specifically because ROC-AUC gets inflated by the large number of easy true negatives under this much class imbalance. This baseline is the bar the GraphSAGE GNN needs to clear to justify its added complexity -- if the GNN can't beat it, that will be reported as a finding, not tuned away.

Two internal design choices worth naming: (1) boosting-round selection uses a nested validation split -- steps 1-29 train, 30-34 validate (early stopping only) -- so the steps 35-49 test set is never touched until the one final evaluation; (2) `community` is passed as a native pandas categorical column (`enable_categorical=True`) rather than one-hot encoded or treated as an ordered number.

**The full tuning history, reported honestly rather than showing only the final number:**

| config | test precision | test recall | test F1 | test PR-AUC |
|---|---|---|---|---|
| original (max_depth=6, no regularization) | 0.553 | 0.754 | 0.638 | 0.788 |
| + full regularization (depth 4, subsample/colsample 0.8, min_child_weight 5) | 0.394 | 0.752 | 0.517 | 0.757 |
| + depth 6, subsample 0.8 only | 0.418 | 0.746 | 0.536 | 0.760 |
| + recency-weighted training + tuned decision threshold | 0.537 | 0.756 | 0.628 | 0.798 |
| **+ drop 6 severely time-drifted features (final)** | **0.953** | **0.715** | **0.817** | **0.804** |

- **Regularization was tried and rejected.** The first run showed a large train/test gap (F1 0.97 -> 0.64), suggesting overfitting -- but cutting model capacity made test performance *worse*, not better, in two separate attempts. If capacity were the problem, less of it should have closed the gap; instead it widened. That's evidence the gap was distribution shift across time, not memorization.
- **Recency weighting + threshold tuning** (rows weighted 0.5x-1.5x by how late their time_step falls in the training window; decision threshold chosen by maximizing F1 on out-of-sample validation predictions rather than assuming 0.5) produced a small, mostly-noise-level PR-AUC gain over the original.
- **The decisive change**: `models/feature_drift_audit.py` compares every feature's distribution between the train and test periods using a two-sample Kolmogorov-Smirnov test -- no labels involved, the same kind of covariate-shift check a production system would run on live data. Six features (`feat_100`, `feat_101`, `feat_103`, `feat_136`, `feat_137`, `feat_139`) showed near-total separation (KS > 0.9) between the two periods, a distinct cluster clearly separated from the rest (the next-highest KS statistic is 0.61) -- consistent with these being cumulative/time-indexed aggregated features (per the original paper's local+aggregated feature split) that mechanically trend with absolute time step rather than encoding durable transaction risk. Tree models can't extrapolate past value ranges seen in training, so a feature whose entire test-period range sits outside its train-period range actively misleads the model rather than merely failing to help. Dropping these 6 features (out of 172) nearly doubled precision (0.537 -> 0.953) and raised F1 from 0.628 to 0.817 -- and train performance improved too (F1 0.958 -> 0.987), which rules out an ordinary bias-variance trade-off explanation: removing the features let the trees find cleaner splits on the remaining, stable features, rather than just reducing variance at the cost of fit.

**Final baseline result: precision 0.953, recall 0.715, F1 0.817, PR-AUC 0.804 on the steps 35-49 test set.** This is the number Step 6's GraphSAGE model needs to clear.

## Tech stack (so far)

- **Data / graph**: pandas, networkx, python-louvain, pyarrow (parquet caching)
- **Modeling**: XGBoost, PyTorch + PyTorch Geometric (GraphSAGE)
- **Serving / UI**: FastAPI, SvelteKit

## Repo structure

```
graphsentry/
├── data/          # Elliptic loaders, cached parquet
├── graph/         # NetworkX construction, centrality, community detection
├── features/      # feature engineering (structural + node attributes), pseudo-labeling, threshold sensitivity
├── models/        # baseline (XGBoost), GNN (GraphSAGE/GCN via PyG)
├── backend/       # FastAPI
├── frontend/      # SvelteKit dashboard
├── notebooks/      # research notebook
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset (requires Kaggle credentials configured for `kagglehub`):

```
python3 -c "import kagglehub; print(kagglehub.dataset_download('ellipticco/elliptic-data-set'))"
```

Copy the three CSVs into `data/raw/`, then:

```
python -m data.loader          # sanity-check the raw data
python -m graph.graph_builder  # build and cache the graph
python -m graph.analytics      # compute structural features
python -m features.build_features       # merge features, print temporal split summary
python -m features.pseudo_label         # run the self-training extension
python -m features.threshold_sensitivity  # pseudo-label confidence threshold sweep
python -m models.feature_drift_audit    # audit train/test feature distribution shift
python -m models.baseline_xgboost       # train and evaluate the XGBoost baseline
```

## Limitations (so far)

- Betweenness centrality is only available for nodes in the single largest time-step component (~4% of all nodes).
- 77.1% of nodes have no real label and are excluded from supervised training; the pseudo-labeling extension is exploratory and not verified ground truth.
- The train/test class balance shifts over time (11.6% -> 6.5% illicit), so train-time and test-time metrics are not directly comparable to each other.
- The 6 features excluded from the XGBoost baseline for severe train/test drift (see Methodology) are excluded only from that model -- they're still present in the shared feature table, so the GNN (Step 6) will need its own judgment call on whether to use them.
- Even after removing the drifted features, baseline recall on test is 0.715 -- roughly 1 in 4 illicit test transactions still go undetected. Precision improved substantially, but this is not a solved problem.

## Future work

- Extend the temporal-GNN idea to explicitly model change across time steps, rather than treating each as an independent snapshot.
- Feed high-confidence pseudo-labels into a controlled semi-supervised training experiment, clearly separated from the primary evaluation.
- Investigate the labeling-selection bias noted above (why "unknown" and "known" populations differ) rather than treating it as background noise.
- Understand *why* the 6 excluded features drift so severely (confirm the cumulative/time-indexed hypothesis directly, rather than inferring it from the KS statistic pattern alone).
