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
        self.computed_dp_so_far = 0
        # self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed = self.run_heuristic()

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
        self.heur_trend_result, self.heur_removed_per_group, self.heur_total_removed = trend_result, removed_per_group, len(removed_df)
        return trend_result, removed_per_group, len(removed_df)


    def compute_next_partial_solution(self):
        heur_trend_result, heur_removed_per_group, heur_total_removed = self.run_heuristic()

        # output these somehow
        dp_function_map = {"sum": "SUM", "max": "MAX", "avg": "AVG", "median": "MEDIAN"}
        inc_dp = IncrementalDP(df, grouping_col, aggregation_col, dp_function_map[agg_func],
                     max_removed=heur_total_removed, time_cutoff_seconds=None)

        group_keys = sorted(df[grouping_col].unique())
        for i in range(len(group_keys)):
            # if only the last group remains, just compute the full solution.
            heur_agg_value_of_next_group = None
            if i+1 < len(group_keys):
                group_key_i_plus_1 = group_keys[i+1]
                heur_agg_value_of_next_group = heur_trend_result[group_key_i_plus_1]
            subset_df, removed_df = inc_dp.compute_up_to_i(i, heur_agg_value_of_next_group)
            # output this somehow as soon as it's ready
            intermediate_result = subset_df.groupby(grouping_col)[aggregation_col].agg(pandas_function_map[agg_func]).to_dict()

        return intermediate_result, heur_trend_result,