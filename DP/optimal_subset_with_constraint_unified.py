from sortedcontainers import SortedDict
import pandas as pd
from typing import Dict, List, Union, Type

from DP.input_parser import get_aggregation_function
from DP.aggregations_mem import AggregationMem


def update_H_with_pruning(F, H, group_id):
    """
    :param F: dictionary from agg value to count of remaining tuples (for group i)
    :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
    :param group_id: value of group by column that represents group i.
    :return:
    """
    agg_values = sorted(F.keys(), reverse=True)  # sort feasible aggregation values from large to small
    # for agg_value in range(len(F) - 1, 0, -1):
    for agg_value in agg_values:
        if F[agg_value] > 0:
            index = H.bisect_right(agg_value)
            largest_below_agg_value = H.keys()[index-1]
            candidate_repair_size = H[largest_below_agg_value][0] + F[agg_value]
            if (agg_value not in H) or (candidate_repair_size > H[agg_value][0]):
                new_list = H[largest_below_agg_value][1].copy()
                new_list.insert(0, (agg_value, group_id))
                H[agg_value] = (candidate_repair_size, new_list)
    return H


def update_H_no_pruning(F, H, group_id):
    """
    :param F: dictionary from agg value to count of remaining tuples (for group i)
    :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i-1, such that agg value of group i-1 = x)
    :param group_id: value of group by column that represents group i.
    :return:
    """
    agg_values = sorted(F.keys(), reverse=False)  # sort feasible aggregation values from large to small

    H_keys = sorted(H.keys(), reverse=False) # keep frozen for the iteration

    previous_max_repair_size = 0
    previous_max_repair_group_values = []
    H_index = 0
    for agg_value in agg_values:
        if F[agg_value] <= 0:
            continue
        while H_index < len(H_keys) and H_keys[H_index] <= agg_value:
            repair_size = H[H_keys[H_index]][0]
            if repair_size > previous_max_repair_size:
                previous_max_repair_size = repair_size
                previous_max_repair_group_values = H[H_keys[H_index]][1].copy()
            H_index += 1
        candidate_repair_size = previous_max_repair_size + F[agg_value]
        if (agg_value not in H) or (candidate_repair_size > H[agg_value][0]):
            new_list = previous_max_repair_group_values.copy()
            new_list.insert(0, (agg_value, group_id))
            H[agg_value] = (candidate_repair_size, new_list)
    return H


def prune_H(H, max_removed=None, sum_of_group_sizes=None):
    """
    If x1<=x2 and H(x1)>=H(x2), keep only x1, and prune x2.
    :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i, such that agg value of group i = x)
    """
    max_count = -1
    newH = {}
    for option in H.keys():
        if option > 0 and max_removed is not None:
            # compute removal from groups 1,.., i. If it's too large, no need to remember this option.
            if (sum_of_group_sizes - H[option][0]) > max_removed:
                continue
        if H[option][0] > max_count:
            newH[option] = H[option]
            max_count = H[option][0]
    return SortedDict(newH)


def prune_H_by_max_removed(H, max_removed, sum_of_group_sizes=None):
    """
    If x1<=x2 and H(x1)>=H(x2), keep only x1, and prune x2.
    :param H: sorted dict of agg value (x) to: (count, [(agg_value, group_id)]) for groups 1..i, such that agg value of group i = x)
    """
    newH = {}
    for option in H.keys():
        # compute removal from groups 1,.., i-1. If it's too large, no need to remember this option.
        if (sum_of_group_sizes - H[option][0]) > max_removed:
            continue
        newH[option] = H[option]
    return SortedDict(newH)


def get_optimal_subset_F_first(
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        agg_col: str,
        #agg_func_str: str,
        Agg: Type[AggregationMem],
        max_removed: int = None,
        prune_dp_by_max_removed: int = None,
        prune_h: bool = False,
        time_cutoff_seconds: int = None,
        htrack_file=None,
) -> (pd.DataFrame, pd.DataFrame):
    print(len(df))
    print("mem opt, F first")
    if max_removed is not None:
        print(f"prune agg pack: {max_removed}")
    if prune_dp_by_max_removed is not None:
        print(f"prune dp: {prune_dp_by_max_removed}")
    print(f"prune h: {prune_h}")
    df = df.loc[df[group_cols].notnull().all(axis=1)].reset_index(drop=True)
    print("agg result before repair:")
    print(df.groupby(group_cols)[agg_col].agg(['sum', 'count', 'mean', 'median', 'max']))

    output = {}

    H = SortedDict()
    H[0] = (0, [])  # first element is the amount of items, the second is the aggregation value in each key group
    group_keys = []

    aggs = {}
    group_sizes = {}
    # First compute F (realizable aggregations and max subset size) for each group.
    for group_key, group_df in df.groupby(group_cols):  # groupby keys are sorted by default
        print(f"working on group: {group_key}")
        agg = Agg()
        output[group_key] = agg.compute_max_subset_sizes(group_df, agg_col, max_removed, time_cutoff_seconds)
        aggs[group_key] = agg
        group_keys.append(group_key)
        group_sizes[group_key] = len(group_df)
    # Next, compute the solution (main DP).
    sum_of_group_sizes = 0

    # create a new file for tracking H sizes
    if htrack_file is not None:
        htrack = open(htrack_file, "w")
        htrack.close()

    for group_key in group_keys:
        print(f"merging + pruning group: {group_key}")
        sum_of_group_sizes += group_sizes[group_key]
        if prune_h:
            H = update_H_with_pruning(output[group_key], H, group_key)
            H = prune_H(H, prune_dp_by_max_removed, sum_of_group_sizes)
        else:
            H = update_H_no_pruning(output[group_key], H, group_key)
            if prune_dp_by_max_removed is not None:
                H = prune_H_by_max_removed(H, prune_dp_by_max_removed, sum_of_group_sizes)
        # output the num keys in H
        if htrack_file is not None:
            with open(htrack_file, "a") as out:
                out.write(f"{group_key}, num_keys in H: {len(H)}\r\n")


    ids_to_keep = []
    if prune_h:
        # We don't need to search for the best solution in H because of the pruning.
        # The solution with the largest x value will be the largest repair.
        largest_x = H.keys()[-1]
        print(f"largest_x: {largest_x}, H[largest_x]={H[largest_x]}")
        agg_values_and_group_keys = H[largest_x][1]
    else:
        # No pruning - search for the best solution in H
        best_repair_size = 0
        agg_values_and_group_keys = None
        for x in H:
            repair_size, group_values = H[x]
            if repair_size > best_repair_size:
                best_repair_size = repair_size
                agg_values_and_group_keys = group_values

    for agg_value, group_key in agg_values_and_group_keys:
        print(f"find subset for group {group_key} with agg value {agg_value}")
        ids_to_keep.extend(aggs[group_key].get_subset_for_value(agg_value))

    subset_df = df.iloc[ids_to_keep]
    removed_df = df.loc[~df.index.isin(ids_to_keep)]
    print("agg result after repair:")
    print(subset_df.groupby(group_cols)[agg_col].agg(['sum', 'count', 'mean', 'median', 'max']))
    #print(f"num_removed: {len(removed_df)}")

    return subset_df, removed_df

    # for agg_value, key in H[H.keys()[-1]][1]:
    #     items_in_key = list(get_subset_with_sum(vals[key], data[key], agg_value))
    #     for item in items_in_key:
    #         print((key[0], item[0]), item[1])
    #         needed_items[(key[0], item[0])] = item[1]
    #
    # indices_to_remove = []
    # for idx, row in df.iterrows():
    #     if ((row[group_cols[0]], row[agg_col]) in needed_items) and needed_items[
    #         (row[group_cols[0]], row[agg_col])] > 0:
    #         needed_items[(row[group_cols[0]], row[agg_col])] = needed_items[(row[group_cols[0]], row[agg_col])] - 1
    #     else:
    #         indices_to_remove.append(idx)
    # print(indices_to_remove)
    # df2 = df.drop(indices_to_remove).reset_index(drop=True)
    # df2.to_csv('df2_output.csv', index=False)
    # print(H[H.keys()[-1]])

class IncrementalDP(object):
    def __init__(self, df: pd.DataFrame,
                 # group_cols: Union[str, List[str]],
                 group_col: str,
                 agg_col: str,
                 #Agg: Type[AggregationMem],
                 agg_func_string: str,
                 max_removed: int = None,
                 time_cutoff_seconds: int = None,
                 ):
        self.df = df
        self.group_col = group_col
        self.agg_col = agg_col
        self.Agg = get_aggregation_function(agg_func_string, agg_pack_opt=True)
        # self.Agg = Agg
        self.max_removed = max_removed
        self.time_cutoff_seconds = time_cutoff_seconds
        self.H = SortedDict()

        self.H[0] = (0, [])  # first element is the amount of items, the second is the aggregation value in each key group
        self.computed_group_keys = []
        self.F_dict = {}
        self.aggs = {}
        self.group_sizes = {}
        self.raw_group_keys = sorted(self.df[group_col].unique())
        #self.group_keys_and_df = self.df.groupby(group_cols)  # groupby keys are sorted by default
        self.sum_of_group_sizes = 0

    def compute_up_to_i(self, i, agg_value_of_i_plus_1=None):
        """
        Args:
            i: how many groups to compute the solution for.
            agg_value_of_i_plus_1: aggregation value of group i+1
        Returns:

        """
        # check how many Fs we already computed
        already_computed = len(self.aggs)
        # First compute F (realizable aggregations and max subset size) for each remaining group.
        for group_key in self.raw_group_keys[already_computed:i+1]:
            group_df = self.df[self.df[self.group_col] == group_key]
            print(f"working on group: {group_key}")
            agg = self.Agg()
            self.F_dict[group_key] = agg.compute_max_subset_sizes(group_df, self.agg_col, self.max_removed, self.time_cutoff_seconds)
            self.aggs[group_key] = agg
            self.computed_group_keys.append(group_key)
            self.group_sizes[group_key] = len(group_df)

        for group_key in self.computed_group_keys[already_computed:]:
            print(f"merging group: {group_key}")
            self.sum_of_group_sizes += self.group_sizes[group_key]
            self.H = update_H_no_pruning(self.F_dict[group_key], self.H, group_key)

        # No pruning - search for the best solution in H
        best_repair_size = 0
        agg_values_and_group_keys = None
        for x in self.H:
            if agg_value_of_i_plus_1 is not None and x > agg_value_of_i_plus_1:
                continue
            repair_size, group_values = self.H[x]
            if repair_size > best_repair_size:
                best_repair_size = repair_size
                agg_values_and_group_keys = group_values

        ids_to_keep = []
        for agg_value, group_key in agg_values_and_group_keys:
            print(f"find subset for group {group_key} with agg value {agg_value}")
            ids_to_keep.extend(self.aggs[group_key].get_subset_for_value(agg_value))

        # subset_df = self.df.iloc[ids_to_keep]
        removed_so_far_df = self.df.loc[(self.df[self.group_col].isin([self.raw_group_keys[:i+1]]))
                                        & (~self.df.index.isin(ids_to_keep))]
        #print("agg result after repair:")
        #print(subset_df.groupby(self.group_cols)[self.agg_col].agg(['sum', 'count', 'mean', 'median', 'max']))
        # print(f"num_removed: {len(removed_df)}")
        # return subset_df, removed_so_far_df
        return removed_so_far_df
