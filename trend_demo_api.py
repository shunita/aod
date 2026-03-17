import os
import pandas as pd
from scipy.stats import entropy

from DP.input_parser import get_aggregation_function
from DP.optimal_subset_with_constraint_unified import IncrementalDP, get_optimal_subset_F_first
from Heuristic.aggr_main import greedy_algorithm

pandas_function_map = {"sum": "sum", "max": "max", "avg": "mean", "median": "median"}

class TrendRepair(object):

    def __init__(self, df, agg_func, grouping_col, aggregation_col, trend_direction="non-decreasing"):
        """
        df: a dataframe to repair
        agg_func: a string out of ["sum", "max", "avg", "median"]
        grouping_col: a column name from the dataframe to group by.
        aggregation_col: a column name from the dataframe to aggregate.
        trend_direction: either "non-decreasing" or "non-increasing".
        """
        self.df = df.copy()
        self.agg_func = agg_func
        self.grouping_col = grouping_col
        self.aggregation_col = aggregation_col
        self.trend_direction = trend_direction
        self.backend_grouping_col = "__trend_backend_gid__"

        # Handle mixed types in grouping column by converting to string for sorting.
        # Also drop NaN values from grouping column to avoid sorting issues.
        self.df = self.df.dropna(subset=[self.grouping_col]).copy()

        unique_vals = self.df[self.grouping_col].unique()
        try:
            self.group_keys = sorted(unique_vals)
        except TypeError:
            self.group_keys = sorted(unique_vals, key=str)

        # Visible/UI order stays the original sorted order.
        # Backend order is reversed only for non-increasing.
        if self.trend_direction == "non-increasing":
            self.backend_order_groups = list(reversed(self.group_keys))
        else:
            self.backend_order_groups = list(self.group_keys)

        self.group_to_backend_rank = {
            group_key: rank for rank, group_key in enumerate(self.backend_order_groups)
        }
        self.backend_rank_to_group = {
            rank: group_key for group_key, rank in self.group_to_backend_rank.items()
        }
        self.backend_group_keys = list(range(len(self.backend_order_groups)))

        self.backend_df = self.df.copy()
        self.backend_df[self.backend_grouping_col] = self.backend_df[self.grouping_col].map(
            self.group_to_backend_rank
        )

        # Per-group tuple counts in the original dataset (used for UI hover tooltips)
        self.original_tuples_per_group = self.df.groupby(self.grouping_col, dropna=False).size()

        # Will be populated after running heuristic / each DP step
        self.heur_deleted_per_group = None
        self.heur_left_per_group = None
        self.last_step_deleted_per_group = None
        self.last_step_left_per_group = None

        self.computed_dp_so_far = 0
        self.heur_trend_result = None
        self.heur_trend_result_backend = None
        self.heur_removed_per_group = None
        self.heur_total_removed = None
        self.removed_by_heur = None
        self.removed_by_heur_backend = None
        self.removed_by_dp_step = []
        self.inc_dp = None
        self.removed_by_full_dp = None

    def _sort_trend_result_visible(self, trend_result):
        if trend_result is None or trend_result.empty:
            return trend_result

        ordered = trend_result.copy()
        ordered[self.grouping_col] = pd.Categorical(
            ordered[self.grouping_col], categories=self.group_keys, ordered=True
        )
        ordered = ordered.sort_values(self.grouping_col).reset_index(drop=True)
        ordered[self.grouping_col] = ordered[self.grouping_col].astype(object)
        return ordered

    def _trend_result_from_subset(self, subset_df, group_col):
        trend_result = subset_df.groupby(group_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]
        ).reset_index()

        if group_col == self.backend_grouping_col:
            trend_result[self.grouping_col] = trend_result[self.backend_grouping_col].map(
                self.backend_rank_to_group
            )
            trend_result = trend_result[[self.grouping_col, self.aggregation_col]]

        return self._sort_trend_result_visible(trend_result)

    def _removed_per_group_visible(self, removed_df, group_col):
        if removed_df is None or removed_df.empty:
            return pd.Series(dtype=int)

        if group_col == self.backend_grouping_col:
            visible_removed = self.df[self.df.index.isin(removed_df.index)]
            return visible_removed.groupby(self.grouping_col)[self.aggregation_col].agg("count")

        return removed_df.groupby(group_col)[self.aggregation_col].agg("count")

    def run_query(self):
        trend_result = self.df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]
        ).reset_index()
        return self._sort_trend_result_visible(trend_result)

    def run_heuristic(self, progress_callback=None):
        output_csv = os.path.join("demo_results", "heur_results.csv")
        result_df_backend, removed_df_backend = greedy_algorithm(
            self.backend_df,
            pandas_function_map[self.agg_func],
            grouping_column=self.backend_grouping_col,
            aggregation_column=self.aggregation_col,
            output_csv=output_csv,
            progress_callback=progress_callback,
        )

        trend_result_backend = result_df_backend.groupby(self.backend_grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]
        ).reset_index()
        trend_result = self._trend_result_from_subset(result_df_backend, self.backend_grouping_col)
        removed_per_group = self._removed_per_group_visible(removed_df_backend, self.backend_grouping_col)

        self.removed_by_heur_backend = removed_df_backend
        self.removed_by_heur = self.df[self.df.index.isin(removed_df_backend.index)]

        # Per-group tuple stats (used for UI hover tooltips)
        remaining_df = self.df.drop(index=self.removed_by_heur.index, errors="ignore")
        self.heur_left_per_group = remaining_df.groupby(self.grouping_col, dropna=False).size()
        self.heur_left_per_group = self.heur_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.heur_deleted_per_group = self.original_tuples_per_group.subtract(
            self.heur_left_per_group, fill_value=0
        ).astype(int)

        self.heur_trend_result_backend = trend_result_backend[[self.backend_grouping_col, self.aggregation_col]].copy()
        self.heur_trend_result = trend_result
        self.heur_removed_per_group = removed_per_group
        self.heur_total_removed = len(self.removed_by_heur)
        return trend_result, removed_per_group, len(self.removed_by_heur)

    def run_full_dp_no_heur(self):
        # DP parser expects uppercase aggregation names
        dp_function_map = {
            "sum": "SUM",
            "max": "MAX",
            "avg": "AVG",
            "median": "MEDIAN",
        }

        dp_agg_func = dp_function_map[self.agg_func]

        # Aggregation-pack optimization exists only for SUM / AVG / MEDIAN
        should_optimize_agg_pack = dp_agg_func in {"SUM", "AVG", "MEDIAN"}

        Agg = get_aggregation_function(dp_agg_func, agg_pack_opt=should_optimize_agg_pack)

        subset_df_backend, removed_df_backend = get_optimal_subset_F_first(
            self.backend_df,
            [self.backend_grouping_col],
            self.aggregation_col,
            Agg,
            max_removed=None,
            prune_dp_by_max_removed=None,
            prune_h=False,
            time_cutoff_seconds=None,
            htrack_file=None,
        )

        self.removed_by_full_dp = self.df[self.df.index.isin(removed_df_backend.index)]
        self.removed_by_dp_step = [self.removed_by_full_dp.copy()]
        self.computed_dp_so_far = len(self.group_keys)

        subset_df_visible = self.df[~self.df.index.isin(self.removed_by_full_dp.index)]

        # Per-group tuple stats for the final optimal solution (used by the UI)
        self.last_step_left_per_group = subset_df_visible.groupby(self.grouping_col, dropna=False).size()
        self.last_step_left_per_group = self.last_step_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.last_step_deleted_per_group = self.original_tuples_per_group.subtract(
            self.last_step_left_per_group, fill_value=0
        ).astype(int)

        trend_result = self._trend_result_from_subset(subset_df_backend, self.backend_grouping_col)
        removed_per_group = self._removed_per_group_visible(removed_df_backend, self.backend_grouping_col)
        return trend_result, removed_per_group, len(self.removed_by_full_dp)


    def __get_next_constraint(self, heur_trend_result):
        index_of_next_agg_value = self.computed_dp_so_far + 1
        heur_agg_value_of_next_group = None

        while index_of_next_agg_value < len(self.backend_group_keys):
            next_group_key = self.backend_group_keys[index_of_next_agg_value]
            if next_group_key in heur_trend_result:
                heur_agg_value_of_next_group = heur_trend_result[next_group_key]
                break
            index_of_next_agg_value += 1

        return heur_agg_value_of_next_group

    def compute_next_partial_solution(self):
        if self.removed_by_heur_backend is None:
            print("running heuristic")
            self.run_heuristic()

        heur_trend_result = self.heur_trend_result_backend.set_index(self.backend_grouping_col).to_dict()[self.aggregation_col]

        dp_function_map = {"sum": "SUM", "max": "MAX", "avg": "AVG", "median": "MEDIAN"}
        if self.inc_dp is None:
            print("initializing incremental DP")
            self.inc_dp = IncrementalDP(
                self.backend_df,
                self.backend_grouping_col,
                self.aggregation_col,
                dp_function_map[self.agg_func],
                max_removed=len(self.removed_by_heur_backend),
                time_cutoff_seconds=None,
            )

        heur_agg_value_of_next_group = self.__get_next_constraint(heur_trend_result)
        print(
            f"groups computed so far: {self.computed_dp_so_far}\n"
            f"next heur agg value: {heur_agg_value_of_next_group}"
        )
        removed_tuples_up_to_i = self.inc_dp.compute_up_to_i(self.computed_dp_so_far, heur_agg_value_of_next_group)

        remaining_group_keys = self.backend_group_keys[self.computed_dp_so_far + 1 :]
        removed_tuples_from_i_plus_1 = self.removed_by_heur_backend[
            self.removed_by_heur_backend[self.backend_grouping_col].isin(remaining_group_keys)
    ]
        print(
            "remaining backend group_keys: ",
            remaining_group_keys,
            "tuples removed by heuristic from groups i+1 to n: ",
            removed_tuples_from_i_plus_1,
        )

        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        removed_visible_df = self.df[self.df.index.isin(all_removed_index)]
        self.removed_by_dp_step.append(removed_visible_df)
        subset_df_visible = self.df[~self.df.index.isin(all_removed_index)]
        print(
            "initial df size: ",
            len(self.df),
            "\ntotal removed index: ",
            len(all_removed_index),
            "\nremaining in df: ",
            len(subset_df_visible),
        )
        intermediate_result = self._trend_result_from_subset(
            self.backend_df[~self.backend_df.index.isin(all_removed_index)],
            self.backend_grouping_col,
        )

        # Per-group tuple stats for this intermediate solution (used for UI hover tooltips)
        self.last_step_left_per_group = subset_df_visible.groupby(self.grouping_col, dropna=False).size()
        self.last_step_left_per_group = self.last_step_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.last_step_deleted_per_group = self.original_tuples_per_group.subtract(
            self.last_step_left_per_group, fill_value=0
        ).astype(int)

        self.computed_dp_so_far += 1
        return intermediate_result

    def compute_deleted_for_current_step(self):
        """
        Compute 'tuples deleted' for the *current* optimal step.
        Call immediately after compute_next_partial_solution().
        """
        if self.removed_by_heur_backend is None or self.heur_trend_result_backend is None or self.inc_dp is None:
            return None

        step_num = int(self.computed_dp_so_far)
        i = step_num - 1
        if i < 0:
            return None

        heur_map = self.heur_trend_result_backend.set_index(self.backend_grouping_col).to_dict()[self.aggregation_col]

        if i + 1 >= len(self.backend_group_keys):
            removed_tuples_up_to_i = self.inc_dp.compute_up_to_i(i, None)
            if removed_tuples_up_to_i is None:
                return None
            return int(len(removed_tuples_up_to_i))

        cutoff = None
        next_backend_rank = i + 1
        while next_backend_rank < len(self.backend_group_keys):
            if next_backend_rank in heur_map:
                cutoff = heur_map[next_backend_rank]
                break
            next_backend_rank += 1

        removed_tuples_up_to_i = self.inc_dp.compute_up_to_i(i, cutoff)
        remaining_group_keys = self.backend_group_keys[i + 1 :]
        removed_tuples_from_i_plus_1 = self.removed_by_heur_backend[
            self.removed_by_heur_backend[self.backend_grouping_col].isin(remaining_group_keys)
        ]
        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        return int(len(all_removed_index))

    def get_step_boundary_info(self, step_num):
        """Return display-facing info about the boundary after a DP prefix step."""
        try:
            step_num = int(step_num)
        except Exception:
            return {"next_group_key": None, "cutoff": None}

        if self.heur_trend_result_backend is None:
            return {"next_group_key": None, "cutoff": None}

        heur_map = self.heur_trend_result_backend.set_index(self.backend_grouping_col).to_dict()[self.aggregation_col]
        next_backend_rank = step_num
        while next_backend_rank < len(self.backend_group_keys):
            if next_backend_rank in heur_map:
                return {
                    "next_group_key": self.backend_rank_to_group.get(next_backend_rank),
                    "cutoff": heur_map[next_backend_rank],
                }
            next_backend_rank += 1

        return {"next_group_key": None, "cutoff": None}

    def summarize_distribution_differences(self,
                                           repair_name,
            # removed_subset: pd.DataFrame,
            k: int = 20,
            epsilon: float = 1e-8,
            ignore_columns=None
    ):
        """
        Compare value distributions between a subset of rows and the rest
        of the DataFrame using KL divergence.

        Parameters
        ----------
        repair_name: "heuristic" or DP step number
        k : int
            Number of top attributes to return.
        epsilon : float
            Smoothing constant to avoid zero probabilities.
        ignore_columns : list or None
            Columns to skip (e.g., IDs, continuous columns).

        Returns
        -------
        pd.DataFrame
            Columns:
            - attribute
            - kl_divergence
            - subset_distribution
            - rest_distribution
        """
        if ignore_columns is None:
            ignore_columns = [self.aggregation_col, self.grouping_col, self.backend_grouping_col]

        if repair_name == "heuristic":
            removed_subset = self.removed_by_heur
        else:
            if self.removed_by_dp_step:
                removed_subset = self.removed_by_dp_step[-1]
            elif self.removed_by_full_dp is not None:
                removed_subset = self.removed_by_full_dp
            else:
                return []
        rest = self.df[~self.df.index.isin(removed_subset.index)]

        results = []

        for col in self.df.columns:
            if col in ignore_columns:
                continue

            # Skip non-categorical columns
            if not pd.api.types.is_object_dtype(self.df[col]) \
                    and not pd.api.types.is_categorical_dtype(self.df[col]):
                continue

            # Align value supports
            subset_counts = removed_subset[col].value_counts(normalize=True)
            rest_counts = rest[col].value_counts(normalize=True)

            all_values = subset_counts.index.union(rest_counts.index)

            p = subset_counts.reindex(all_values, fill_value=0.0) + epsilon
            q = rest_counts.reindex(all_values, fill_value=0.0) + epsilon

            # Normalize after smoothing
            p /= p.sum()
            q /= q.sum()

            kl = entropy(p, q)

            # subset_distrib_summary = ", ".join([f"{v}: {freq:.2f}" for v, freq in p.to_dict().items()])
            # rest_distrib_summary = ", ".join([f"{v}: {freq:.2f}" for v, freq in q.to_dict().items()])
            results.append({
                "attribute": col,
                "kl_divergence": kl,
                "subset_distribution": p.to_dict(),
                "rest_distribution": q.to_dict(),
                # "subset_distribution": subset_distrib_summary,
                # "rest_distribution": rest_distrib_summary,
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("kl_divergence", ascending=False)

        # print(result_df)

        textual_summary = []

        # Use sorted results (by KL divergence)
        sorted_results = result_df.head(k).to_dict('records')

        for row in sorted_results:
            textual_summary.append(f"**{row['attribute']}**")
            textual_summary.append("")
            table = textify_distribution_table(row['subset_distribution'], row['rest_distribution'])
            textual_summary.extend(table)
            textual_summary.append("")  # spacing between attributes

        return textual_summary

def textify_distribution_table(
    dist_a: dict,
    dist_b: dict,
    name_a: str = "Removed",
    name_b: str = "Rest",
    # decimals: int = 3,
    sort_by: str = "diff",  # "abs_diff", "diff", "a", "b", "value"
    min_freq: float = 0.01
):
    """
    Print a text table comparing two discrete distributions.

    Parameters
    ----------
    dist_a, dist_b : dict
        value -> frequency (sums to 1)
    name_a, name_b : str
        Column names.
    decimals : int
        Number of decimal places.
    sort_by : str
        Sorting criterion.
    min_freq : float
        Drop rows where both frequencies are below this threshold.
    """

    values = sorted(set(dist_a) | set(dist_b))

    rows = []
    for v in values:
        a = dist_a.get(v, 0.0)
        b = dist_b.get(v, 0.0)

        if a < min_freq and b < min_freq:
            continue

        rows.append({
            "value": str(v),
            name_a: a,
            name_b: b,
            "Δ": a - b,
        })

    if sort_by == "abs_diff":
        rows.sort(key=lambda r: abs(r["Δ"]), reverse=True)
    elif sort_by == "diff":
        rows.sort(key=lambda r: r["Δ"], reverse=True)
    elif sort_by == "a":
        rows.sort(key=lambda r: r[name_a], reverse=True)
    elif sort_by == "b":
        rows.sort(key=lambda r: r[name_b], reverse=True)
    elif sort_by == "value":
        rows.sort(key=lambda r: r["value"])

    # Markdown table format
    output = [
        f"| Value | {name_a} | {name_b} |",
         "|-------|----------|----------|"
    ]

    for r in rows[:3]:
        output.append(f"| {r['value']} | {r[name_a]:.2f} | {r[name_b]:.2f} |")

    return output