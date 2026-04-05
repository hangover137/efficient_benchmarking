# Benchmarking on Tasks That Matter: Dataset Selection for Preserving Model Rankings

Code and experiment notebooks accompanying the paper **"Benchmarking on Tasks That Matter: Dataset Selection for Preserving Model Rankings"**.

This repository studies how to replace a large benchmark with a **small, representative subset of datasets** while preserving the **global ranking of models** induced by the full benchmark.

The codebase implements the evaluation protocol from the paper, including:

- repeated subsampling of dataset pools or model pools,
- multiple dataset-selection strategies,
- rank-preservation metrics,
- confidence-interval aggregation,
- plotting and AUC summaries,
- statistical tests over repeated trials.

---

## What this repository contains

At a high level, the repository has two layers:

1. **Core Python utilities** used by all experiments.
2. **Notebook-driven experiments** that correspond to the main text, appendices, and auxiliary hypothesis checks from the paper.

The main programmatic entry point is:

- `testing_pipeline(...)` in `testing_pipeline_stats.py`

This function runs the repeated benchmarking protocol, evaluates each dataset-selection method over a grid of subset sizes, saves raw and aggregated results, and produces plots.

---

## Repository structure

```text
.
├── testing_pipeline_stats.py
├── get_global_const.py
├── get_metrics.py
├── get_ranks.py
├── plot_results.py
├── data_loader.py
├── data_stats.py
├── testing_pipeline_tsc.ipynb
├── testing_pipeline_recsys.ipynb
├── testing_pipeline_models_all.ipynb
├── testing_pipeline_feat_select.ipynb
├── oracle_exp.ipynb
├── testing_pipeline_pca.ipynb
├── 30_9_reduce_tsc_exp.ipynb
├── extract_all_feats.ipynb
├── extract_SM_feats.ipynb
├── extract_tsfresh.ipynb
├── corr_pics.ipynb
└── distances_hyp.ipynb
```

### Core Python files

- `testing_pipeline_stats.py` — the main experiment runner.
- `get_global_const.py` — loads benchmark score tables and returns `(scores, datasets, models)`.
- `get_ranks.py` — builds fold-wise and mean model rankings.
- `get_metrics.py` — computes rank-preservation metrics such as `MAE`, `Spearman`, `Kendall`, and `NDCG@k`.
- `plot_results.py` — plots metric-vs-subset-size curves and prints AUC summaries.
- `data_loader.py` — utilities for loading published TSC benchmark results and caching datasets.
- `data_stats.py` — small helper utilities for dataset-level tabular statistics.

---

## Paper-to-code map

This section is the most important one if you want to reproduce a specific result from the paper. The most illustrative notebook is `testing_pipeline_tsc.ipynb`

### 1. Main TSC benchmark experiments

**Notebook:** `testing_pipeline_tsc.ipynb`

This is the main notebook for the **Time Series Classification (TSC)** experiments on the full benchmark. It loads the TSC benchmark, constructs dataset representations, prepares `metods_data_list`, and launches `testing_pipeline(...)` for the main selection strategies.

It covers the core TSC comparisons across:

- Random sampling,
- K-Means,
- Greedy farthest-first (cosine / Euclidean),
- A-optimality,
- D-optimality,
- multiple dataset representations such as TSFRESH, Catch22, MiniRocket, Summary, Simple Features, and Landmarking.

Use this notebook for:

- the **main TSC dataset-subset experiments**,
- the **complete TSC method–representation grid**,
- selecting the **best representation per strategy** used in the compact summary tables.

In paper terms, this notebook is the main source for:

- the TSC part of the main benchmark comparison,
- the best-configuration summaries reported in **Table 1**,
- the appendix section **“Complete Method–Representation Evaluation Grid”**, including:
  - **Figure 3 / Table 4** — K-Means in TSC,
  - **Figure 4 / Table 5** — greedy farthest-first in TSC,
  - **Figure 5 / Table 6** — optimal design-based methods in TSC.

---

### 2. Main RecSys benchmark experiments

**Notebook:** `testing_pipeline_recsys.ipynb`

This notebook runs the **Recommender Systems (RecSys)** experiments. It prepares the RecSys feature table, constructs ranking targets, and launches the same experimental pipeline as in the TSC case, but for the RecSys benchmark.

Use this notebook for:

- the **main RecSys dataset-subset experiments**,
- the RecSys portion of the best-strategy comparison in the paper,
- comparing Random / K-Means / farthest-first / A-opt / D-opt on the RecSys benchmark.

This notebook also includes a **RecSys feature-selection variant**, where a smaller set of RecSys features is selected and the experiments are rerun.

In paper terms, this notebook supports:

- the RecSys part of **Table 1**,
- the broader comparison showing that strategy differences in RecSys are much smaller than in TSC.

---

### 3. Model pool variation experiments

**Notebook:** `testing_pipeline_models_all.ipynb`

This notebook corresponds to the experiments where the **dataset pool is fixed** and the **model pool is repeatedly subsampled**.

It runs the pipeline with `models_bench=True`, which changes the evaluation regime from “vary datasets” to “vary models”. The notebook includes both:

- **TSC model-pool subsampling**, and
- **RecSys model-pool subsampling**.

Use this notebook for reproducing the appendix section on robustness to model-pool changes.

In paper terms, this notebook corresponds to:

- **Appendix E: Sensitivity to the Model Pool Composition**,
- **Figure 6**,
- **Table 7**.

---

### 4. Feature–performance diagnostics and feature-selection sensitivity

**Notebook:** `testing_pipeline_feat_select.ipynb`

This notebook first computes feature-level dependence statistics between dataset descriptors and an aggregated performance target. It then uses simple fixed rules to keep only selected features and reruns the benchmark pipeline on the reduced representations.

It is therefore the notebook behind two closely related parts of the paper:

1. **Feature–performance correlation diagnostics**.
2. **Feature-selection sensitivity experiments**.

The notebook computes per-feature measures such as:

- Pearson correlation,
- Spearman correlation,
- mutual information,

and then applies the selection rules described in the paper to build reduced representations for:

- TSFRESH,
- Catch22,
- MiniRocket,
- Summary,
- Landmarking,
- Simple Features.

Use this notebook for:

- the diagnostic analysis of representation informativeness,
- rerunning TSC experiments after feature filtering.

In paper terms, this notebook supports:

- **Figure 10** — feature–performance correlation diagnostics,
- the **feature-selection sensitivity study**, including:
  - **Figure 7 / Table 8** — K-Means with feature selection,
  - **Figure 8 / Table 9** — farthest-first with feature selection,
  - **Figure 9 / Table 10** — optimal-design methods with feature selection.

---

### 5. Synthetic oracle vs. broken representation experiment

**Notebook:** `oracle_exp.ipynb`

This notebook builds a **synthetic benchmark** with latent structure, then compares selection methods under two regimes:

- an **oracle / aligned representation**, and
- a **broken / misaligned representation**.

It is intended to show that geometric selection helps only when the representation space is actually aligned with ranking-relevant structure.

Use this notebook for the synthetic sanity-check experiment.

In paper terms, this notebook corresponds to:

- **Table 11**,
- **Figure 11**.

---

### 6. PCA ablation / dimensionality reduction experiments

**Notebook:** `testing_pipeline_pca.ipynb`

This notebook reruns the TSC experiments with PCA-based dimensionality reduction enabled in the method arguments (for example with `pca_099=True`).

It is useful for:

- checking robustness of the selection methods to compressed representations,
- reducing feature dimensionality before K-Means / farthest-first / A-opt / D-opt,
- exploratory ablations on representation preprocessing.

This notebook is best viewed as an **ablation / robustness notebook**. In the current manuscript snapshot, it is not as clearly tied to a numbered figure/table as the notebooks above, so it is best treated as an auxiliary reproduction notebook.

---

### 7. Downscaled TSC benchmark: 30 datasets × 9 models

**Notebook:** `30_9_reduce_tsc_exp.ipynb`

This notebook is the bridge between the full TSC benchmark and the smaller RecSys-scale regime. It works with a **downscaled TSC setting of 30 datasets and 9 models**, matching the scale comparison explicitly mentioned in the paper.

In the current version of the notebook, the emphasis is on **collecting and aggregating already-saved run outputs** (for example from prefixes like `Random_30_9_F_`, `Cosine_Method_Catch22_30_9_F_`, etc.) and comparing AUC gains relative to Random.

Use this notebook for:

- the **TSC 30×9** downscaled regime,
- scale-comparison sanity checks,
- comparing how much of the TSC advantage remains when the benchmark is shrunk to RecSys size.

In paper terms, this notebook is the natural source for:

- the **TSC 30×9** regime in **Table 2**.

---

### 8. Feature generation / preprocessing notebooks

These notebooks generate the dataset-level representations later consumed by the main experiment notebooks.

#### `extract_all_feats.ipynb`

Creates **series-level** representations from dataset JSON files in `data/datasets_metrics/datasets/` and saves them under `data/datasets_features/`:

- `tsfresh_series_features.csv`
- `catch22_series_features.csv`
- `summary_series_features.csv`
- `minirocket_series_features.csv`

Each row in these files corresponds to a **single time series** inside a dataset and includes at least:

- `dataset`
- `series_id`
- feature columns

#### `extract_SM_feats.ipynb`

Builds **dataset-level probe / meta descriptors** from the series-level representation files and also computes raw landmarking descriptors.

Outputs include:

- `data/datasets_features/apriori_meta_tsfresh_probe.csv`
- `data/datasets_features/apriori_meta_catch22_probe.csv`
- `data/datasets_features/apriori_meta_summary_probe.csv`
- `data/datasets_features/apriori_meta_minirocket_probe.csv`
- `data/datasets_features/landmarking_raw.csv`

These files are later reused in:

- `testing_pipeline_tsc.ipynb`
- `testing_pipeline_feat_select.ipynb`
- `corr_pics.ipynb`
- `distances_hyp.ipynb`

#### `extract_tsfresh.ipynb`

Older / standalone notebook for extracting **full TSFRESH features** directly from dataset JSONs. This is the source of:

- `data/datasets_features/tsfresh_full_features.csv`

which is later used in the correlation-analysis notebooks.

---

### 9. Hypothesis / geometry / correlation notebooks

These notebooks are not the main benchmark runners, but they are useful for checking the geometric hypotheses behind the paper.

#### `distances_hyp.ipynb`

This notebook tests whether the **geometry of the feature space** is aligned with the **geometry of the benchmark ranking space**.

It contains utilities for:

- building pairwise **rank-distance** matrices between datasets,
- building pairwise **feature-distance** matrices between datasets,
- measuring alignment between them via Spearman correlation,
- comparing distance choices such as **cosine**, **euclidean**, and **correlation**,
- checking how PCA changes alignment,
- comparing the global geometry of different representations.

Use this notebook for questions such as:

- Does cosine distance align better with rank geometry than Euclidean?
- Does PCA improve or degrade alignment?
- Which representations have the most ranking-aware geometry?

#### `corr_pics.ipynb`

This notebook focuses on **feature–performance dependence diagnostics**.

It computes and visualizes per-feature relationships with aggregated benchmark targets using:

- Pearson correlation,
- Spearman correlation,
- mutual information,
- histograms of metric distributions,
- heatmaps of selected feature groups.

Use this notebook for plots and sanity checks behind the feature-analysis parts of the paper.

---

## Which notebook should I run?

- **Main TSC experiments**: `testing_pipeline_tsc.ipynb`.
- **Main RecSys experiments**: `testing_pipeline_recsys.ipynb`.
- **Model-pool robustness** study: `testing_pipeline_models_all.ipynb`.
- **Feature-selection sensitivity** study: `testing_pipeline_feat_select.ipynb`.
- **Oracle vs. broken synthetic** experiment: `oracle_exp.ipynb`.
- **PCA ablation**: `testing_pipeline_pca.ipynb`.
- **Downscaled TSC 30×9** regime: `30_9_reduce_tsc_exp.ipynb`.
- **Generate dataset representations from raw dataset JSONs**: `extract_all_feats.ipynb` and `extract_SM_feats.ipynb`.
- **Check geometry / distance hypotheses**: `distances_hyp.ipynb`.
- **Inspect feature–performance correlation plots**:  `corr_pics.ipynb`.

---

## Pipeline schematic

![We study how to select a small subset of datasets that best preserves the global model ranking of a large benchmark. Datasets are embedded via task features, subsets are chosen by multiple strategies (geometric, clustering, design-based), and an evaluation protocol with bootstrap aggregation yields rank-preservation metrics, enabling comparisons across subset sizes and domains (time series classification, recommender systems).](figures\pipeline_overview.png)

```text
Raw benchmark results + dataset JSONs
        ↓
Dataset-level representations / feature tables
        ↓
Dataset selection method
        ↓
Selected subset of datasets
        ↓
Aggregate model ranks on selected subset
        ↓
Compare against full-benchmark ranking
        ↓
Metrics, confidence intervals, plots, AUC, stats tests
```

---

## Experimental protocol implemented in code

The main evaluation protocol implemented by `testing_pipeline(...)` is:

1. Repeatedly sample benchmark perturbations.
   - either by subsampling datasets,
   - or by subsampling models when `models_bench=True`.
2. For each selection method and each subset size `k`:
   - select a subset of datasets,
   - compute the induced model ranking,
   - compare it to the reference full-benchmark ranking.
3. Aggregate results over trials.
4. Save raw trial-level outputs and confidence-interval summaries.
5. Plot curves and compute AUC summaries.

This directly matches the experimental protocol described in the paper.

---

## Canonical data formats used in the code

This section documents the **actual formats expected by the pipeline utilities**, based on how the notebooks and helper functions use them.

### `datasets`

`datasets` is the ordered collection of dataset identifiers.

Typical form:

```python
list[str] | np.ndarray[str]
```

Examples from the notebooks:

```python
scores, datasets, models = get_global_const()
chosen_datasets = sorted(datasets)
chosen_datasets = np.array(chosen_datasets)
```

Important properties:

- each entry is a dataset name / ID,
- order matters,
- rows of every representation matrix passed into `metods_data_list` must be aligned with this order.

### `scores`

`scores` is the raw benchmark-result object.

Canonical form:

```python
dict[str, pandas.DataFrame]
```

Interpretation:

- key = model name,
- value = one DataFrame with benchmark results for that model across datasets and folds.

For the TSC utilities, each DataFrame is expected to look like:

```text
folds: | 0 | 1 | 2 | ...
```

where:

- column `folds:` contains dataset names,
- the remaining columns are per-fold scores for that model.

So conceptually:

```python
scores[model_name].shape == (n_datasets, 1 + n_folds)
```

This is why the notebooks do operations such as:

```python
scores_aggr = {
    model: model_score.set_index("folds:").loc[chosen_datasets, :].reset_index()
    for model, model_score in scores.items()
}
```

If your raw benchmark file is not already in this format, convert it into this dictionary-of-DataFrames structure first.

### `models`

`models` is the ordered list of benchmark model names.

Typical form:

```python
list[str]
```

This is primarily used when:

- subsetting the model pool,
- constructing `model_indx`,
- running the `models_bench=True` regime.

### `ranks`

`ranks` is the precomputed fold-wise rank object returned by:

```python
get_ranks_s(..., return_ranks=True)
```

Canonical form:

```python
dict[str, pandas.DataFrame]
```

Interpretation:

- key = dataset name,
- value = DataFrame with one row per model and one column per fold rank.

Conceptually, each entry looks like:

```text
model | 0 | 1 | 2 | ...
```

where the numeric columns contain **ranks, not raw scores**.

The notebooks use it like this:

```python
ranks[d].drop(columns=['model']).mean(axis=1)
```

which means:

- each dataset-specific DataFrame stores model ranks over folds,
- averaging over fold columns gives mean rank per model for that dataset.

### `ranks_all`

`ranks_all` is the aggregated rank vector returned by:

```python
get_ranks_s(..., return_ranks=False)
```

It is the mean model-rank vector over the currently selected dataset set.

Conceptually:

```python
np.ndarray[float]  # length = n_models
```

### Representation matrices passed to methods

The `data` field inside each `metods_data_list` tuple is usually a numeric matrix aligned row-wise with `datasets`:

```python
np.ndarray[float]  # shape = (n_datasets, n_features)
```

Examples:

- `tsf.values`
- `c22.values`
- `mr.values`
- `sm.values`
- `lm.values`
- `prep_features.values`
- `recsys_features.values`

The important invariant is:

> row `i` of the representation matrix must correspond to dataset `datasets[i]`.

### Series-level feature tables

Files such as `tsfresh_series_features.csv` are **not** directly passed into the selection pipeline. They are intermediate files used to build dataset-level descriptors.

Canonical structure:

```text
dataset | series_id | feature_1 | feature_2 | ...
```

### Dataset-level feature tables

Files such as `tsfresh_dataset_mean.csv`, `apriori_meta_tsfresh_probe.csv`, or `landmarking_raw.csv` are dataset-aligned tables.

Canonical structure:

```text
dataset | feature_1 | feature_2 | ...
```

or, for the original simple-feature file:

```text
Name | feature_1 | feature_2 | ...
```

These tables are typically aligned to `chosen_datasets` by sorting / reindexing and then converted to `.values` before being passed to the selection methods.

---

## Method specification format

Each experiment notebook builds a `metods_data_list` consumed by `testing_pipeline(...)`.

Each entry has the form:

```python
(method, data, sizes, need_l, need_m, args, transp, models_in_data, label)
```

Where:

- `method` — dataset-selection function,
- `data` — dataset representation matrix used by that method,
- `sizes` — subset sizes to evaluate,
- `need_l` — whether Lasso-based auxiliary preprocessing is needed (N/A),
- `need_m` — whether entropy / GP-based auxiliary preprocessing is needed (N/A),
- `args` — keyword arguments forwarded to the method,
- `transp` — whether the method returns selections for all sizes at once (N/A),
- `models_in_data` — whether the representation depends on the sampled model subset (N/A),
- `label` — experiment name used for saving and plotting.

### Minimal example

```python
metods_data_list = [
    [rand_ind_method,
     prep_features.values,
     range(2, 21),
     False,
     False,
     {},
     False,
     False,
     'Random'],

    [get_more_different_datasets,
     tsf.values,
     range(2, 21),
     False,
     False,
     {'scale_data': True},
     False,
     False,
     'Cosine_Method_TSFRESH'],
]
```

Interpretation of the example above:

- `rand_ind_method` runs the random baseline on `prep_features.values`,
- `get_more_different_datasets` runs farthest-first style selection in the TSFRESH representation,
- both methods are evaluated for subset sizes `k = 2, 3, ..., 20`,
- both receive row-aligned dataset representations,
- experiment names are used for checkpoint directories and plot legends.

---

## Saved outputs

For each method, `testing_pipeline(...)` writes a directory containing:

- `metrics.pkl` — aggregated metric curves,
- `raw_results.csv` — iteration-level values for each metric and subset size,
- `summary_ci.csv` — mean and empirical 95% interval for each metric,
- `meta.json` — run metadata,
- metric plots such as `MAE.png`, `Spearman.png`, etc.

When `run_stats_tests=True`, the pipeline also supports Friedman / Holm style significance analysis over raw repeated-trial results.

---

## Data and feature files expected by the notebooks

### Raw benchmark results

#### TSC benchmark results

The TSC benchmark results are loaded through `get_global_const()` / `load_model_results()` and are expected under:

- `data/datasets_metrics/metrics/...`

The loader is designed around the **published benchmark result tables** from:

- `https://timeseriesclassification.com/results/PublishedResults/`

In other words, the canonical TSC source in this repository is:

1. published per-model benchmark CSV files from the Time Series Classification website,
2. locally cached under `data/datasets_metrics/metrics/...`,
3. loaded into the `scores` dictionary-of-DataFrames structure.

#### TSC dataset JSON files

Feature-generation notebooks expect serialized dataset files under:

- `data/datasets_metrics/datasets/<dataset>.json`

These JSON files are the raw input for:

- `extract_all_feats.ipynb`
- `extract_tsfresh.ipynb`
- `extract_SM_feats.ipynb`

#### RecSys benchmark results

The RecSys notebooks build their benchmark tensors from a **local tabular RecSys benchmark file** that is reshaped into dataset-by-method tables inside `testing_pipeline_recsys.ipynb`.

Conceptually, the notebook starts from a long table with columns like:

- `Dataset`
- `Method`
- `Value`

and then pivots it into the structures used to derive `scores_recsys`, `ranks_recsys`, and `recsys_features`.

If you plan to make the RecSys part fully standalone, it is a good idea to keep the original raw RecSys source file under a dedicated path such as:

- `data/recsys/...`

and document that exact filename here.

### Feature tables consumed by experiment notebooks

The main notebooks assume the repository contains precomputed dataset-level feature tables such as:

- `data/datasets_features/features.csv`
- `data/datasets_features/tsfresh_important_features.csv`
- `data/datasets_features/tsfresh_full_features.csv`
- `data/datasets_features/dataset_level/tsfresh_dataset_mean.csv`
- `data/datasets_features/dataset_level/catch22_dataset_mean.csv`
- `data/datasets_features/dataset_level/minirocket_dataset_mean.csv`
- `data/datasets_features/dataset_level/summary_dataset_mean.csv`
- `data/datasets_features/apriori_meta_tsfresh_probe.csv`
- `data/datasets_features/apriori_meta_catch22_probe.csv`
- `data/datasets_features/apriori_meta_summary_probe.csv`
- `data/datasets_features/apriori_meta_minirocket_probe.csv`
- `data/datasets_features/landmarking_raw.csv`

### Which notebooks generate which feature files?

#### Generated from raw dataset JSONs

- `extract_all_feats.ipynb`
  - `tsfresh_series_features.csv`
  - `catch22_series_features.csv`
  - `summary_series_features.csv`
  - `minirocket_series_features.csv`

- `extract_tsfresh.ipynb`
  - `tsfresh_full_features.csv`

#### Generated from the series-level feature files

- `extract_SM_feats.ipynb`
  - `apriori_meta_tsfresh_probe.csv`
  - `apriori_meta_catch22_probe.csv`
  - `apriori_meta_summary_probe.csv`
  - `apriori_meta_minirocket_probe.csv`
  - `landmarking_raw.csv`

#### Used for diagnostics / hypothesis checking

- `corr_pics.ipynb`
  - reads `tsfresh_full_features.csv`, `tsfresh_important_features.csv`, `features.csv`, dataset-level means, and apriori meta feature files for correlation plots.

- `distances_hyp.ipynb`
  - reads dataset-level mean features, `features.csv`, apriori meta feature files, and `landmarking_raw.csv` to test distance / geometry hypotheses.

If these assets are not present, the notebooks will need to be adapted or the corresponding files restored.

---

## Installation

A minimal environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib tqdm requests beautifulsoup4 aeon torch gpytorch xicorpy distance-correlation
```

Depending on your local checkout, you may also need project-local modules imported by the notebooks, for example under:

- `methods/...`
- `utils/...`

Additional optional dependencies used in the feature-generation notebooks include packages such as:

- `tsfresh`
- `pycatch22`
- `sktime`

---

## Minimal quick start

```python
from get_global_const import get_global_const
from get_ranks import get_ranks_s
from testing_pipeline_stats import testing_pipeline

scores, datasets, models = get_global_const()

ranks = get_ranks_s(
    selected_datasets=datasets,
    scores=scores,
    all_datasets=datasets,
    return_ranks=True,
)

# Build metods_data_list in the same format as in the notebooks,
# then call testing_pipeline(...)
```

For full reproduction, it is better to run the notebooks listed in the paper-to-code map above.

---

## Reproducibility notes

- Randomization is controlled by `random_state`.
- Repeated trials are controlled by `test_iter`.
- The code supports loading cached ranks, cached models, and cached results.
- Most figures in the paper are notebook-generated rather than produced by a standalone script.

---
