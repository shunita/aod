import os
import pandas as pd
from scipy.stats import entropy

from DP.input_parser import get_aggregation_function
from DP.optimal_subset_with_constraint_unified import IncrementalDP, get_optimal_subset_F_first
from Heuristic.aggr_main import greedy_algorithm

pandas_function_map = {"sum": "sum", "max": "max", "avg": "mean", "median": "median"}

class TrendRepair(object):

    def __init__(self, df, agg_func, grouping_col, aggregation_col):
        """
        df: a dataframe to repair
        agg_func: a string out of ["sum", "max", "avg", "median"]
        grouping_col: a column name from the dataframe to group by.
        aggregation_col: a column name from the dataframe to aggregate.
        """
        self.df = df.copy()
        self.agg_func = agg_func
        self.grouping_col = grouping_col
        self.aggregation_col = aggregation_col

        # Handle mixed types in grouping column by converting to string
        # Also drop NaN values from grouping column to avoid sorting issues
        self.df = self.df.dropna(subset=[self.grouping_col])

        # Try to sort, fallback to string conversion if mixed types
        unique_vals = self.df[self.grouping_col].unique()
        try:
            self.group_keys = sorted(unique_vals)
        except TypeError:
            # Mixed types - convert to string for sorting
            self.group_keys = sorted(unique_vals, key=str)

        # Per-group tuple counts in the original dataset (used for UI hover tooltips)
        self.original_tuples_per_group = self.df.groupby(self.grouping_col, dropna=False).size()

        # Will be populated after running heuristic / each DP step
        self.heur_deleted_per_group = None
        self.heur_left_per_group = None
        self.last_step_deleted_per_group = None
        self.last_step_left_per_group = None


        self.computed_dp_so_far = 0
        self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed, self.removed_by_heur = None, None, None, None
        self.removed_by_dp_step = []
        self.inc_dp = None
        self.removed_by_full_dp = None


    def run_query(self):
        trend_result = self.df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]).reset_index()  # .to_dict()
        return trend_result

    def run_heuristic(self, progress_callback=None):
        output_csv = os.path.join("demo_results", "heur_results.csv")
        result_df, removed_df = greedy_algorithm(self.df, pandas_function_map[self.agg_func], grouping_column=self.grouping_col,
                                                 aggregation_column=self.aggregation_col, output_csv=output_csv,
                                                 progress_callback=progress_callback)
        trend_result = result_df.groupby(self.grouping_col)[self.aggregation_col].agg(pandas_function_map[self.agg_func]).reset_index() #.to_dict()
        removed_per_group = removed_df.groupby(self.grouping_col)[self.aggregation_col].agg("count")
        self.removed_by_heur = removed_df

        # Per-group tuple stats (used for UI hover tooltips)
        remaining_df = self.df.drop(index=removed_df.index, errors="ignore")
        self.heur_left_per_group = remaining_df.groupby(self.grouping_col, dropna=False).size()
        self.heur_left_per_group = self.heur_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.heur_deleted_per_group = self.original_tuples_per_group.subtract(
            self.heur_left_per_group, fill_value=0
        ).astype(int)

        self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed = trend_result, removed_per_group, len(removed_df)
        return trend_result, removed_per_group, len(removed_df)

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

        subset_df, removed_df = get_optimal_subset_F_first(
            self.df,
            [self.grouping_col],
            self.aggregation_col,
            Agg,
            max_removed=None,
            prune_dp_by_max_removed=None,
            prune_h=False,
            time_cutoff_seconds=None,
            htrack_file=None,
        )

        self.removed_by_full_dp = removed_df
        self.removed_by_dp_step = [removed_df.copy()]
        self.computed_dp_so_far = len(self.group_keys)

        # Per-group tuple stats for the final optimal solution (used by the UI)
        self.last_step_left_per_group = subset_df.groupby(self.grouping_col, dropna=False).size()
        self.last_step_left_per_group = self.last_step_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.last_step_deleted_per_group = self.original_tuples_per_group.subtract(
            self.last_step_left_per_group, fill_value=0
        ).astype(int)

        trend_result = subset_df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]
        ).reset_index()
        removed_per_group = removed_df.groupby(self.grouping_col)[self.aggregation_col].agg("count")
        return trend_result, removed_per_group, len(removed_df)

    # def run_full_dp_no_heur(self):
    #     # Max has no optimized aggregation packing version. The others (sum, median, avg) do.
    #     should_optimize_agg_pack = self.agg_func != 'max'
    #     Agg = get_aggregation_function(self.agg_func, agg_pack_opt=should_optimize_agg_pack)
    #     subset_df, removed_df = get_optimal_subset_F_first(
    #         self.df,
    #         self.grouping_col,
    #         self.aggregation_col,
    #         Agg,
    #         max_removed=None,
    #         prune_dp_by_max_removed=None,
    #         prune_h=False,
    #         time_cutoff_seconds=None,
    #         htrack_file = None)
    #     self.removed_by_full_dp = removed_df
    #     trend_result = subset_df.groupby(self.grouping_col)[self.aggregation_col].agg(
    #         pandas_function_map[self.agg_func]).reset_index()
    #     removed_per_group = removed_df.groupby(self.grouping_col)[self.aggregation_col].agg("count")
    #     return trend_result, removed_per_group, len(removed_df)


    def __get_next_constraint(self, heur_trend_result):
        index_of_next_agg_value = self.computed_dp_so_far + 1
        heur_agg_value_of_next_group = None
        if index_of_next_agg_value < len(self.group_keys):
            next_group_key = self.group_keys[index_of_next_agg_value]
            # next group is not included in the heuristic solution - take the following aggregate value from the heuristic
            while next_group_key not in heur_trend_result and index_of_next_agg_value < len(self.group_keys):
                index_of_next_agg_value += 1
                next_group_key = self.group_keys[index_of_next_agg_value]
            if index_of_next_agg_value < len(self.group_keys):
                heur_agg_value_of_next_group = heur_trend_result[next_group_key]
        return heur_agg_value_of_next_group

    def compute_next_partial_solution(self):
        if self.removed_by_heur is None:
            print("running heuristic")
            self.run_heuristic()
        heur_trend_result = self.heur_trend_result.set_index(self.grouping_col).to_dict()[self.aggregation_col]

        dp_function_map = {"sum": "SUM", "max": "MAX", "avg": "AVG", "median": "MEDIAN"}
        if self.inc_dp is None:
            print("initializing incremental DP")
            self.inc_dp = IncrementalDP(self.df, self.grouping_col, self.aggregation_col, dp_function_map[self.agg_func],
                                        max_removed=len(self.removed_by_heur), time_cutoff_seconds=None)

        # if only the last group remains, just compute the full solution.
        next_group_key = None
        heur_agg_value_of_next_group = self.__get_next_constraint(heur_trend_result)
        print(f"groups computed so far: {self.computed_dp_so_far}\n next group: {next_group_key}\n next heur agg value: {heur_agg_value_of_next_group}")
        removed_tuples_up_to_i = self.inc_dp.compute_up_to_i(self.computed_dp_so_far, heur_agg_value_of_next_group)

        # compute how many removed by heuristic from i+1 to the end
        remaining_group_keys = self.group_keys[self.computed_dp_so_far+1:]
        removed_tuples_from_i_plus_1 = self.removed_by_heur[self.removed_by_heur[self.grouping_col].isin(remaining_group_keys)]
        print("remaining group_keys: ", remaining_group_keys, "tuples removed by heur from groups i+1 to n: ", removed_tuples_from_i_plus_1)
        # combine the solutions
        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        self.removed_by_dp_step.append(self.df[self.df.index.isin(all_removed_index)])
        subset_df = self.df[~self.df.index.isin(all_removed_index)]
        print("initial df size: ", len(self.df),
              "\ntotal removed index: ", len(all_removed_index),
              "\nremaining in df: ", len(subset_df))
        intermediate_result = subset_df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]).reset_index()
        
        # Per-group tuple stats for this intermediate solution (used for UI hover tooltips)
        self.last_step_left_per_group = subset_df.groupby(self.grouping_col, dropna=False).size()
        self.last_step_left_per_group = self.last_step_left_per_group.reindex(
            self.original_tuples_per_group.index, fill_value=0
        ).astype(int)
        self.last_step_deleted_per_group = self.original_tuples_per_group.subtract(
            self.last_step_left_per_group, fill_value=0
        ).astype(int)

        self.computed_dp_so_far += 1
        return intermediate_result


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
            ignore_columns = [self.aggregation_col, self.grouping_col]

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