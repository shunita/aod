# Analyzing Deviations from Monotonic Trends through Database Repair

This repository contains code accompanying the paper **"Analyzing Deviations from Monotonic Trends through Database Repair"**. It includes implementations of:

- A **Dynamic Programming (DP)** algorithm for computing minimal repairs to enforce monotonicity.
- A **Heuristic** algorithm offering a faster, approximate alternative.

---

## Repository Structure

```
.
├── DP/               # Dynamic Programming implementation
├── Heuristic/        # Heuristic algorithm implementation
├── data/             # Example datasets
├── README.md         # This file
```

---

## Example: German Credit Dataset

We demonstrate how to run the algorithms on the **German Credit** dataset, measuring the monotonicity of `good_loan` with respect to `present_employment_since_numeric` under the `avg` aggregation function.

---

## Running the DP Algorithm

```bash
python DP/main.py AVG \
  data/german_credit/german_textual.csv good_loan present_employment_since_numeric \
  --output_folder data/german_credit/trend_results \
  --mem_opt --agg_pack_opt --prune_aggpack_by_greedy 30
```

### Arguments:
- `AVG` — Aggregation function (supported: AVG, SUM, MEDIAN, MAX, COUNT, COUNT_DISTINCT)
- `good_loan` — aggregation attribute
- `present_employment_since_numeric` — Group-by attribute(s)
- `--mem_opt` — Enable memory optimization
- `--agg_pack_opt` — Enable aggregation packing optimization (available for AVG, SUM, MEDIAN)
- `--prune_aggpack_by_greedy <N>` given a result from the heuristic algorithm (see below), enable 
  pruning the search to consider only solutions removing up to N tuples.
---

## Running the Heuristic Algorithm

```bash
python Heuristic/aggr_main.py \
  --grouping_column present_employment_since_numeric \
  --aggregation_column good_loan \
  --output_folder data/german_credit/trend_results \
  data/german_credit/german_textual.csv avg
```

---

## Comparing Heuristic and DP Variants

```bash
python Heuristic/compare_algorithms.py \
  --agg_function avg \
  --dataset_folder data/german_credit/german_textual.csv \
  --results_folder data/german_credit/trend_results \
  --timeout_min 60 \
  --grouping_column present_employment_since_numeric \
  --aggregation_column good_loan
```
