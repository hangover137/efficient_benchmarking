# Benchmarking on Tasks That Matter: Dataset Selection for Preserving Model Rankings

Code accompanying the paper **"Benchmarking on Tasks That Matter: Dataset Selection for Preserving Model Rankings"**.

This repository studies how to replace a large benchmark with a **small, representative subset of datasets** while preserving the **global ranking of models** obtained on the full benchmark. The code implements the evaluation protocol from the paper, computes rank-preservation metrics, runs repeated subsampling experiments, stores raw and aggregated results, and produces performance-vs-subset-size plots.

---

## Overview

Multi-dataset benchmarking is often expensive: evaluating every model on every dataset can be prohibitively slow, especially when the benchmark contains many datasets. This project treats **dataset subset selection** as a standalone problem:

> given a large pool of datasets, select a subset of size `k` that preserves the model ranking induced by the full benchmark as faithfully as possible.

The repository supports experiments in the two settings discussed in the paper:

- **Time Series Classification (TSC)**
- **Recommender Systems (RecSys)**

The main comparison is between several dataset-selection strategies, including:

- random sampling,
- clustering-based selection,
- greedy farthest-first selection,
- A-/D-optimality-inspired design criteria.

Evaluation is based on repeated subsampling and uncertainty-aware aggregation of ranking metrics such as **MAE**, **Spearman**, **Kendall**, and **NDCG@k**.

---

## What is implemented here

The supplied code covers the following parts of the pipeline:

- loading benchmark results and dataset metadata,
- constructing aggregate model ranks from per-dataset fold results,
- computing rank-preservation metrics,
- running repeated subset-selection experiments through a unified pipeline,
- saving raw iteration-level results and aggregated confidence intervals,
- plotting metric curves and reporting AUC summaries.

The main experiment entry point is `testing_pipeline(...)` in `testing_pipeline_stats.py`.

---

## Repository structure

Below is the part of the project reflected by the provided files.

```text
.
├── testing_pipeline_stats.py   # main experimental pipeline
├── get_global_const.py         # loads benchmark scores / datasets / models
├── get_metrics.py              # rank-preservation metrics
├── get_ranks.py                # aggregation of fold-level model ranks
├── plot_results.py             # plotting and AUC reporting
├── data_loader.py              # download/load benchmark result tables and datasets
└── data_stats.py               # simple dataset-shape utilities
```

In the full project, these files are expected to live inside a larger package layout such as `utils/...` and `methods/...`, because `testing_pipeline_stats.py` imports helper modules from those locations.


---

## Method specification format

The main pipeline expects `metods_data_list` to be a list of tuples:

```python
(method, data, sizes, need_l, need_m, args, transp, models_in_data, label)
```

Where:

- `method`: subset-selection callable,
- `data`: dataset representation used by the method,
- `sizes`: subset sizes to evaluate,
- `need_l`: whether to build a Lasso-based reduced representation (N/A),
- `need_m`: whether to build GP/entropy-based auxiliary model features (N/A),
- `args`: keyword arguments forwarded to the method,
- `transp`: whether the method returns selections for all sizes at once (N/A),
- `models_in_data`: whether the representation depends on the sampled model subset  (N/A),
- `label`: method name used in plots and saved outputs.

Two execution regimes are supported:

- **standard mode**: evaluate subset selection over sampled dataset pools,
- **model-benchmark mode** (`models_bench=True`): evaluate ranking stability with sampled model pools.

---

## Metrics

By default, the code reports a compact set of rank-preservation metrics:

- `MAE`
- `Spearman`
- `Kendall`
- `NDCG@3`
- `NDCG@5`

The extended metrics function also supports additional measures such as `MSE`, `Pearson`, mutual-information-based scores, `Xi Correlation`, and `Distance Correlation`.

---

## Core modules

### `testing_pipeline_stats.py`
Main experiment driver.

Key responsibilities:

- repeated train/test subsampling,
- method evaluation across subset sizes,
- raw/summary result serialization,
- plot generation,
- optional preparation of alternative representations,
- helpers for Friedman-Holm significance testing.

### `get_global_const.py`
Loads benchmark scores and returns the main benchmark constants:

- model score tables,
- dataset list,
- model list.

### `get_ranks.py`
Contains utilities for aggregating model ranks across datasets and folds.

- `get_ranks_s(...)` builds rank tables directly from raw score tables,
- `get_ranks(...)` aggregates mean ranks for a selected subset of datasets.

### `get_metrics.py`
Computes similarity / discrepancy metrics between the full ranking and the ranking induced by a selected subset.

### `plot_results.py`
Plots metric curves over subset size and prints AUC summaries.

### `data_loader.py`
Utilities for:

- downloading published benchmark result CSV files,
- serializing datasets to JSON,
- loading cached datasets from JSON.

---

## Reproducing experiments

A typical reproduction flow is:

1. Download or place benchmark CSV result files under your metrics directory.
2. Load benchmark metadata using `get_global_const(...)`.
3. Build the full ranking object with `get_ranks_s(..., return_ranks=True)`.
4. Prepare dataset representations for each selection strategy.
5. Assemble `metods_data_list`.
6. Run `testing_pipeline(...)`.
7. Inspect:
   - generated plots,
   - `summary_ci.csv`,
   - `raw_results.csv`,
   - `meta.json`.

Because parts of the project are notebook-driven, you will typically launch the pipeline from an experiment notebook rather than from a standalone CLI script.

---

## Notes on reproducibility

- Randomization is controlled through `random_state`.
- Repeated subsampling is controlled by `test_iter`.
- The code supports loading cached ranks/results/models to avoid recomputation.
- The pipeline uses empirical quantiles rather than parametric assumptions for uncertainty summaries.