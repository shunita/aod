import os
from DP.optimal_subset_with_constraint_unified import IncrementalDP
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
        self.df = df
        self.agg_func = agg_func
        self.grouping_col = grouping_col
        self.aggregation_col = aggregation_col
        self.group_keys = sorted(self.df[self.grouping_col].unique())

        self.computed_dp_so_far = 0
        self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed, self.removed_by_heur = None, None, None, None
        self.inc_dp = None


    def run_query(self):
        trend_result = self.df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]).reset_index()  # .to_dict()
        return trend_result

    def run_heuristic(self):
        output_csv = os.path.join("demo_results", "heur_results.csv")
        result_df, removed_df = greedy_algorithm(self.df, pandas_function_map[self.agg_func], grouping_column=self.grouping_col,
                                                 aggregation_column=self.aggregation_col, output_csv=output_csv)
        trend_result = result_df.groupby(self.grouping_col)[self.aggregation_col].agg(pandas_function_map[self.agg_func]).reset_index() #.to_dict()
        removed_per_group = removed_df.groupby(self.grouping_col)[self.aggregation_col].agg("count")
        self.removed_by_heur = removed_df
        self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed = trend_result, removed_per_group, len(removed_df)
        return trend_result, removed_per_group, len(removed_df)


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
        heur_agg_value_of_next_group = None
        if self.computed_dp_so_far + 1 < len(self.group_keys):
            next_group_key = self.group_keys[self.computed_dp_so_far+1]
            heur_agg_value_of_next_group = heur_trend_result[next_group_key]
        print(f"groups computed so far: {self.computed_dp_so_far}\n next group: {next_group_key}\n next heur agg value: {heur_agg_value_of_next_group}")
        removed_tuples_up_to_i = self.inc_dp.compute_up_to_i(self.computed_dp_so_far, heur_agg_value_of_next_group)

        # compute how many removed by heuristic from i+1 to the end
        remaining_group_keys = self.group_keys[self.computed_dp_so_far+1:]
        removed_tuples_from_i_plus_1 = self.removed_by_heur[self.removed_by_heur[self.grouping_col].isin(remaining_group_keys)]
        print("remaining group_keys: ", remaining_group_keys, "tuples removed by heur from groups i+1 to n: ", removed_tuples_from_i_plus_1)
        # combine the solutions
        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        subset_df = self.df[~self.df.index.isin(all_removed_index)]
        print("initial df size: ", len(self.df),
              "\ntotal removed index: ", len(all_removed_index),
              "\nremaining in df: ", len(subset_df))
        intermediate_result = subset_df.groupby(self.grouping_col)[self.aggregation_col].agg(
            pandas_function_map[self.agg_func]).reset_index()

        self.computed_dp_so_far += 1
        return intermediate_result