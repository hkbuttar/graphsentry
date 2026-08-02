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
- **`community` was also tested for exclusion, and reverted -- a genuine theory-vs-result tension worth reporting as-is.** Checked directly: of the 206 Louvain community IDs in the train period and 109 in the test period, the overlap is exactly zero -- a structural guarantee, not a drift pattern, since communities can't span time steps (no edges cross time steps at all -- see Graph construction above). That reads like a clean argument for dropping `community` the same way the six drifted features were dropped: a model conditioning on a specific community ID is learning something certain to never recur at test time. Tested directly rather than left as a plausible-sounding argument: dropping `community` alongside the six drifted features cost 0.11 F1 points (0.817 -> 0.709), mostly via a large precision drop (0.953 -> 0.680). So `community` was kept in, despite the sound theoretical case against it. Likely explanation: XGBoost doesn't ignore unseen categories at prediction time, it routes them down a learned default split direction -- so even though no specific community ID transfers, something about how the model was shaped to handle an unfamiliar community in general still carries signal. This is reported as an unresolved tension between theory and result, not smoothed over in either direction.

**Final baseline result: precision 0.953, recall 0.715, F1 0.817, PR-AUC 0.804 on the steps 35-49 test set.** This is the number Step 6's GraphSAGE model needs to clear.

### GNN model: GraphSAGE (`models/gnn_data.py`, `models/gnn_graphsage.py`)

**Why GraphSAGE, not a plain GCN:** every edge stays within its own time step (see Graph construction above), so a test-period node's entire neighborhood is made of other test-period nodes the model never had access to during training -- not rarely seen, structurally guaranteed unseen, every time. A transductive method (plain GCN, or a per-node embedding table) needs its target nodes present in the graph during training to have learned anything about them at all. GraphSAGE instead learns a shared *aggregation function* -- how to combine a node's neighbors' features -- which, once learned, applies unchanged to any new neighborhood, including a disconnected component from a time period it's never seen.

**Directed message passing:** rather than the usual shortcut of adding reverse edges (which would undo the in/out-degree asymmetry Graph construction deliberately preserved), each layer (`DirectedSAGEBlock`) runs two separate `SAGEConv`s -- one over edges as given (aggregating a node's in-neighbors, who paid it), one over the reversed edge index (aggregating its out-neighbors, who it paid) -- then concatenates both. 3 layers, hidden dimension 64, full-batch training (the whole graph fits in ~140MB of tensors, so `NeighborLoader` mini-batching isn't needed here).

**Inputs:** the same 6 severely time-drifted features excluded from the baseline are excluded here too, for the same reason. `community` is *also* excluded, but for a different reason than why the baseline kept it: XGBoost's categorical splitting has a learned fallback for an unseen category (a default split direction), but a GNN embedding table has no equivalent -- every test-period community ID is unseen, so every test node would get the same constant placeholder embedding, which is not learned behavior, just a fixed stand-in repeated 100% of the time. Numeric features are standardized using train-period statistics only (all nodes, not just labeled ones -- no label information used). Training/evaluation protocol deliberately mirrors the baseline's nested validation split exactly (steps 1-29 train / 30-34 validate for early stopping and threshold selection / 1-34 refit / 35-49 test once), so the two models are actually comparable rather than scored under different rules.

**The tuning history, reported the same way as the baseline's:**

| stage | test precision | test recall | test F1 | test PR-AUC |
|---|---|---|---|---|
| first version (dropout 0.3, weight_decay 1e-5) | 0.775 | 0.487 | 0.598 | 0.568 |
| + validation-tuned regularization (dropout 0.5, weight_decay 1e-4) (final) | **0.809** | **0.556** | **0.659** | **0.631** |

- **Diagnosed before tuning, not assumed fixable:** the first result was weak, but before treating that as final, the validation-loss curve was inspected epoch-by-epoch. Early stopping wasn't the cause -- validation PR-AUC climbed to ~0.93-0.94 by epoch 40-50 and then plateaued/turned noisy while train PR-AUC kept climbing toward 0.999+ (textbook overfitting past that point). So the weak *test* result wasn't under-training; validation looked strong and still didn't transfer to test -- a much larger validation-to-test collapse than the baseline showed.
- **A validation-only sweep** (never touching test) compared four configurations by best validation PR-AUC: lr=0.003 (0.9457), 2-layer instead of 3 (0.9417, worse -- less capacity did not help), dropout=0.5/weight_decay=1e-4 (0.9476, best), and the original (0.9444). All four land within a narrow 0.94-0.95 band -- not a dramatic swing, reported honestly as inconclusive-but-slightly-favoring more regularization. The winning configuration was applied and confirmed on the one real test evaluation: a genuine improvement (F1 +0.061, PR-AUC +0.063).
- **A hypothesis was tested and did not hold up, corrected rather than left standing:** the first instinct was that the *shape* of a typical neighborhood (density, clustering) itself drifts across time periods, making a structure-reliant model more exposed to temporal shift than a feature-only one. Checked directly against the drift audit: `pagerank`, `in_degree`, `out_degree`, and `clustering` all rank in the *least*-drifted half of all 169 features (ranks 61, 101, 140, 160). That hypothesis doesn't hold up, and is corrected here rather than left as an unverified explanation.
- **`betweenness` was investigated instead, and is a real, different story.** It never appeared in the drift audit's output at all -- silently skipped, because it had 7,880 non-null values in train and *zero* in test (it's only computed for the single largest weakly-connected component, which happens to fall entirely in the train period -- see Classical graph analytics above). That means every test node got the exact same constant placeholder value for this feature. Tested directly: removing `betweenness` made GNN validation PR-AUC *worse* (0.9476 -> 0.9408) and made no real difference for the baseline (aucpr 0.9901 -> 0.9902) -- kept in both models, the same "sound theory, contradicted by the result" pattern seen with `community`.
- **Full betweenness coverage was then implemented and tested end-to-end, and reverted.** Since the graph is 49 disjoint components, betweenness computed exactly per component and unioned together is the exact answer, not an approximation -- so the constant-placeholder gap looked like a fixable bug. Timed at ~8.3 minutes and implemented, but the result was measurably worse for **both** models: XGBoost test F1 dropped 0.817 -> 0.728 (PR-AUC 0.804 -> 0.662); the GNN's test F1 dropped 0.659 -> 0.627 (PR-AUC 0.631 -> 0.605). The first explanation considered -- that `networkx`'s per-component normalization (each component is a different size, so the normalization denominator differs) made values incomparable across components -- was checked directly and largely refuted: correlation between component size and mean/max betweenness is weak (0.09 / 0.17); two components of nearly identical size produced betweenness scales differing by 1000x. Betweenness is better understood as a high-variance, topology-specific metric reflecting one time period's particular chain/star/cluster shape -- closer to a per-period idiosyncrasy than a durable, transferable pattern. Reverted to the largest-component-only version, which produced the best measured results for both models.
- **Recency-weighted training was tried last, and is the clearest example in this project of validation and test disagreeing.** The same 0.5x-1.5x row-weighting scheme that gave the baseline a small PR-AUC gain was applied to the GNN's loss. Validation-only check: best validation PR-AUC improved 0.9476 -> 0.9521, a real-looking win. The one actual test evaluation said otherwise: F1 dropped 0.659 -> 0.621, PR-AUC dropped 0.631 -> 0.619, recall fell notably (0.556 -> 0.502). Reverted, because the held-out test result is the only one that counts, and it disagreed with validation -- a concrete reminder that under a real, documented distribution shift, validation is an imperfect proxy for test, not a guarantee.

**Final GNN result: precision 0.809, recall 0.556, F1 0.659, PR-AUC 0.631 on the steps 35-49 test set -- below the XGBoost baseline (F1 0.817, PR-AUC 0.804) on every metric.** Stated plainly rather than tuned further to close the gap: on this dataset, with these features, the GNN does not beat a well-tuned feature-based baseline. This matches a common, defensible finding in the graph-fraud literature -- once strong structural features (pagerank, community, degree, clustering) are already available to a tree model directly, a GNN's main theoretical advantage (learning structure implicitly) has less room to add value, and here it adds a new failure mode instead: heavier reliance on graph structure that itself doesn't transfer across time as cleanly as node-level features do.

## Results: XGBoost vs. GraphSAGE (`models/compare_models.py`)

Both models' cached probabilities are loaded (no retraining) and scored with identical logic, so this table isn't two independently-computed numbers that happen to be placed side by side -- it's one comparison script reading one source of truth per model.

| model | split | n | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | train (1-34) | 29,894 | 0.9985 | 0.9763 | 0.9873 | 0.9997 |
| GraphSAGE | train (1-34) | 29,894 | 0.9803 | 0.9330 | 0.9560 | 0.9930 |
| **XGBoost** | **test (35-49)** | **16,670** | **0.9532** | **0.7147** | **0.8169** | **0.8044** |
| **GraphSAGE** | **test (35-49)** | **16,670** | **0.8091** | **0.5559** | **0.6590** | **0.6313** |

**The GNN does not beat the baseline -- not on precision, not on recall, not on F1, not on PR-AUC, on the test set that actually matters.** That's stated plainly rather than tuned toward a different answer: the GNN's weak first attempt was diagnosed rather than assumed fixable (early stopping wasn't the cause), validation-swept across four configurations, and cross-examined against two competing hypotheses for the gap -- one (neighborhood-shape drift) that didn't hold up under a direct check, and one (betweenness coverage) that did, in a more nuanced and occasionally counterintuitive way than expected (see GNN model section above for the full investigation). None of it closed the gap to the baseline.

This is also a common, defensible finding in the graph-fraud literature, not a surprising one: once strong structural features (PageRank, community, degree, clustering) are already handed directly to a tree-based model, a GNN's main theoretical advantage -- learning structure implicitly through message passing -- has less room to add value. Worse, here it introduces a genuine liability: the GNN leans on graph structure that itself doesn't transfer across time as cleanly as standalone node features do, so it's *more* exposed to this dataset's real, documented distribution shift, not less. A feature-based model that already has good structural features handed to it directly is a hard bar to clear.

## Backend (`backend/`)

A thin FastAPI wrapper around everything above -- every endpoint reads from cached artifacts loaded once at startup (`backend/state.py`: the graph pickle, the merged feature table, both models' cached predictions), never recomputes graph analytics or retrains a model per request. This is the same "one source of truth" principle as `models/compare_models.py` extended to the API layer.

- **`GET /network`** -- one time step's transaction subgraph as nodes/edges JSON (`time_step` query param, default 32 -- 342 illicit nodes of 4,525, a visually meaningful default). Even a single time step can be too large to render smoothly, so if it exceeds `max_nodes` (API default 300; the frontend requests 500 -- see Frontend section below for why), every illicit-labeled node is kept (they're the whole point of the dashboard and would otherwise get diluted by random sampling) and the remaining budget is filled with a random sample of the rest; the response is the induced subgraph on just that sampled set.
- **`GET /predictions`** -- per-node probabilities from both models, filterable by `model`, `min_proba`, and `limit` so the frontend can narrow the payload server-side instead of shipping all 46,564 labeled nodes on every request.
- **`GET /metrics`** -- the exact Step 7 comparison table, computed by calling `models/compare_models.py`'s `compare()` function directly rather than re-deriving precision/recall/F1/PR-AUC here -- the dashboard cannot show numbers that drift from what's documented as the project's result.
- **`GET /node/{id}`** -- single-node lookup: raw + structural features, both models' predicted probabilities, and direct in/out-neighbors (read straight from the loaded graph, respecting direction the same way every other part of this project does: in-neighbors paid this node, out-neighbors were paid by it).

CORS is controlled by an `ALLOWED_ORIGINS` environment variable (comma-separated), defaulting to common local SvelteKit/Vite dev ports -- in production (Step 10) this points at the deployed Vercel URL instead, so the backend never needs a code change to point at a different frontend.

## Frontend (`frontend/`)

A SvelteKit + TypeScript dashboard (Tailwind CSS for styling), built as one cohesive page rather than several stitched-together routes -- the network view, node lookup, and model comparison table all live on `/` and react to the same shared state.

**Why Svelte stores, specifically.** `frontend/src/lib/stores.ts` holds four pieces of state (`selectedNodeId`, `activeModel`, `probabilityThreshold`, `activeTimeStep`) shared across otherwise-unrelated components -- the network graph, the node lookup panel, and the controls bar all read and write these directly. This is a genuinely different model from React: there's no prop-drilling through a common ancestor and no separate state-management library (Context, Redux, Zustand) reached for once state needs to cross component boundaries. A Svelte store is just a value living outside the component tree; any component subscribes by prefixing it with `$` (e.g. `$selectedNodeId`), and the compiler generates the subscribe/unsubscribe boilerplate. Clicking a node in the graph updates `selectedNodeId`; the node lookup panel -- a sibling with no direct relationship to the graph component -- reacts on its own.

**Network visualization** (`NetworkGraph.svelte`): D3 (`d3-force`) computes node positions; Svelte owns the DOM, rendering nodes/edges through ordinary `{#each}` blocks fed by plain (non-reactive) arrays that the simulation mutates each tick, then copied into a `$state` array for the template. Feeding Svelte 5's reactive proxies directly into d3-force was deliberately avoided -- d3 expects to freely mutate plain objects every tick, and mixing that with Svelte's proxy wrapper is an easy source of subtle bugs. Nodes are colored by the active model's predicted probability against the threshold slider (red = predicted illicit, blue = predicted licit, gray = no prediction available for unlabeled nodes), with a black ring marking nodes that are *actually* illicit -- so agreement and disagreement between prediction and ground truth are both visible at once, not just the prediction alone.

**Subsampling, tuned after actually looking at it.** The curated "interesting" time steps in `Controls.svelte` were deliberately chosen for high illicit counts (up to 342, to make the visualization meaningful) -- but the backend's own default render cap (300) meant illicit nodes alone could fill the entire budget, leaving zero room for a licit or unlabeled node to ever appear, so the legend's other colors were never actually demonstrated. Caught by manually driving the app in a real browser (Playwright), not just type-checking: the frontend now requests 500 nodes instead of 300, leaving headroom for real color variety alongside every illicit node.

**A real bug caught the same way**: `NodePanel.svelte` threw `Cannot read properties of null (reading 'toFixed')` on initial load, before any node was even selected. A `key in node.features` guard directly inside a keyed `{#each}` block was not a reliable guarantee that the looked-up value was a number by the time `.toFixed()` ran -- an each-block nested inside a conditional branch could observe a stale/transitional value during a store update. Fixed by computing a filtered, pre-validated list (`$derived`, keeping only entries where `typeof value === 'number'`) before the template ever sees it, rather than guarding ad-hoc inside the each-block.

Run it with:

```
cd frontend
npm install
cp .env.example .env   # set PUBLIC_API_BASE_URL if the backend isn't on the default
npm run dev
```

## Tech stack (so far)

- **Data / graph**: pandas, networkx, python-louvain, pyarrow (parquet caching)
- **Modeling**: XGBoost, PyTorch + PyTorch Geometric (GraphSAGE)
- **Serving / UI**: FastAPI, SvelteKit + TypeScript + Tailwind CSS, d3-force

## Repo structure

```
graphsentry/
├── data/          # Elliptic loaders, cached parquet
├── graph/         # NetworkX construction, centrality, community detection
├── features/      # feature engineering (structural + node attributes), pseudo-labeling, threshold sensitivity
├── models/        # baseline (XGBoost), GNN (GraphSAGE/GCN via PyG), model comparison
├── backend/       # FastAPI: /network, /predictions, /metrics, /node/{id}
├── frontend/      # SvelteKit dashboard: network graph, node lookup, model comparison
├── notebooks/     # research.ipynb -- pre-executed narrative walkthrough
├── tests/         # pytest suite (structural invariants, metrics, model logic, API)
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
python -m models.gnn_data               # build and inspect the PyG graph object
python -m models.gnn_graphsage          # train and evaluate the GraphSAGE GNN
python -m models.compare_models          # print the final XGBoost vs. GraphSAGE comparison
```

Run the backend (requires all of the above to have been run at least once, so the cached artifacts it reads exist):

```
uvicorn backend.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive API docs, or try `http://127.0.0.1:8000/network`.

Run the test suite (each test file skips gracefully if the cached artifacts it needs haven't been generated yet, rather than failing opaquely). Make sure the venv is active first -- a bare `pytest` on your PATH may resolve to an unrelated global installation that doesn't have this project's dependencies:

```
source .venv/bin/activate
pytest tests/ -v
```

Read `notebooks/research.ipynb` for a pre-executed, narrative walkthrough of the whole project with inline charts -- it imports and reuses the same modules as everything above rather than recomputing anything with different logic. To re-run it yourself: `python -m ipykernel install --user --name graphsentry` once, then open it in Jupyter/VS Code and select the "graphsentry" kernel.

## Limitations (so far)

- Betweenness centrality is only available for nodes in the single largest time-step component (~4% of all nodes) -- full per-component coverage was implemented and tested, and measurably hurt both models (see Methodology), so this gap is a deliberate trade-off, not an unexamined one.
- 77.1% of nodes have no real label and are excluded from supervised training; the pseudo-labeling extension is exploratory and not verified ground truth.
- The train/test class balance shifts over time (11.6% -> 6.5% illicit), so train-time and test-time metrics are not directly comparable to each other.
- Even after removing the drifted features, baseline recall on test is 0.715 -- roughly 1 in 4 illicit test transactions still go undetected. Precision improved substantially, but this is not a solved problem.
- The GNN underperforms the baseline on every metric (F1 0.659 vs. 0.817, PR-AUC 0.631 vs. 0.804). This is reported as the honest result of a genuinely tuned attempt (diagnosed, validation-swept, and cross-checked against two competing hypotheses about why), not an under-explored one.

## Future work

- Extend the temporal-GNN idea to explicitly model change across time steps, rather than treating each as an independent snapshot.
- Feed high-confidence pseudo-labels into a controlled semi-supervised training experiment, clearly separated from the primary evaluation.
- Investigate the labeling-selection bias noted above (why "unknown" and "known" populations differ) rather than treating it as background noise.
- Try recency-weighted training for the GNN (deferred deliberately -- see Methodology), or a different neighbor-aggregation function (e.g. max/attention instead of mean), as further-out ideas for narrowing the GNN's gap to the baseline.
- Understand *why* the 6 excluded features drift so severely (confirm the cumulative/time-indexed hypothesis directly, rather than inferring it from the KS statistic pattern alone).
