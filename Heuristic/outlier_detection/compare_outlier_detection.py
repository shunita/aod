import subprocess
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
from collections import Counter

DEBUG = False

def run_greedy(dataset_path, agg_func, grouping_col, agg_col, output_folder):
    greedy_command = [
        "python", "../aggr_main.py", dataset_path, agg_func,
        "--grouping_column", grouping_col,
        "--aggregation_column", agg_col,
        "--output_folder", output_folder,
    ]

    if DEBUG:
        print("# " + " ".join(greedy_command))

    start_time = time.time()
    process = subprocess.run(greedy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    greedy_time = time.time() - start_time

    rows_removed = 0
    for line in process.stdout.splitlines():
        if "Total tuple removals" in line:
            rows_removed = int(float(line.split(":")[-1].strip()))

    return {
        "dataset": dataset_path,
        "rows_removed": rows_removed,
        "greedy_time": greedy_time,
    }

def run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, method_name, param, max_removal_pct, group_wise):
    print(f"Running outlier detection on {dataset_path} with {method_name} and param {param}")

    outlier_command = [
        "python", "outlier_detection.py", dataset_path,
        "--agg_func", agg_func,
        "--group_col", grouping_col,
        "--agg_col", agg_col,
        "--max_removal_pct", str(max_removal_pct),
        "--output_folder", output_folder,
        "--method", method_name,
        "--param", str(param),
    ]

    if group_wise:
        outlier_command.append("--group_wise")

    if DEBUG:
        print("# " + " ".join(outlier_command))

    start_time = time.time()
    process = subprocess.run(outlier_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout
    outlier_time = time.time() - start_time

    outlier_rows_removed = 0
    df_after_outlier_removal_path = None
    for line in output.splitlines():
        if "Removed" in line and "outliers" in line:
            outlier_rows_removed += int(line.split()[1])
            if DEBUG:
                print(f"outliers removed: {outlier_rows_removed}")
        if "Outlier removal results saved to" in line:
            df_after_outlier_removal_path = line.split("to ")[-1].strip()

    # Ensure we have a valid filtered dataset to run the greedy algorithm
    if not df_after_outlier_removal_path or not os.path.exists(df_after_outlier_removal_path):
        raise ValueError("Outlier detection did not produce a valid filtered dataset")

    # Run the greedy algorithm on the filtered dataset
    greedy_results = run_greedy(df_after_outlier_removal_path, agg_func, grouping_col, agg_col, output_folder)
    greedy_removed = greedy_results["rows_removed"]
    if DEBUG:
        print(f"greedy removed: {greedy_removed}")

    results.append({
        "dataset": dataset_path,
        "max_removal_pct": max_removal_pct,
        "method": method_name,
        "param": param,
        "group_wise": group_wise,
        "outlier_rows_removed": outlier_rows_removed,
        "outlier_time": outlier_time,
        "greedy_rows_removed":  greedy_results["rows_removed"],
        "greedy_time": greedy_results["greedy_time"],
        "total_rows_removed": outlier_rows_removed + greedy_results["rows_removed"],
        "total_time": outlier_time + greedy_results["greedy_time"],
    })


def run_outlier_methods(dataset_path, agg_func, grouping_col, agg_col, output_folder, results):
    #max_removal_pcts = [0.005, 0.01, 0.05]
    max_removal_pcts = [100]

    z_score_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    knn_neighbors_values = [3, 5, 8, 10]
    isolation_contamination_values = [0.001 ,0.005 ,0.01, 0.05, 0.1, 0.2]

    for group_wise in [False, True]:
        # z_score
        for param in tqdm(z_score_values, desc="z_score"):
            for max_removal_pct in max_removal_pcts:
                run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "z_score",
                                param, max_removal_pct, group_wise)

        # knn
        for param in tqdm(knn_neighbors_values, desc="knn"):
            for max_removal_pct in max_removal_pcts:
                run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "knn",
                                param, max_removal_pct, group_wise)

        # isolation_forest
        for param in tqdm(isolation_contamination_values, desc="isolation_forest"):
            for max_removal_pct in max_removal_pcts:
                run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "isolation_forest",
                                param, max_removal_pct, group_wise)


def compare_removed_rows_overlap(dataset_path, agg_func, grouping_col, agg_col, output_folder, outlier_results_df):
    """
    Compare the overlap between rows removed by outlier detection methods and greedy algorithm.

    Args:
        dataset_path (str): Path to the original dataset
        agg_func (str): Aggregation function used
        grouping_col (str): Column used for grouping
        agg_col (str): Column used for aggregation
        output_folder (str): Folder for output files
        outlier_results_df (pd.DataFrame): DataFrame containing outlier detection results

    Returns:
        pd.DataFrame: DataFrame with overlap statistics
    """
    print("Analyzing overlap between outlier detection and greedy algorithm...")

    # Load the original dataset
    original_df = pd.read_csv(dataset_path)

    # Filter results for max_removal_pct = 100
    filtered_results = outlier_results_df[outlier_results_df['max_removal_pct'] == 100]

    # Get path to the removed rows by greedy algorithm
    dataset_name_with_csv = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_name_with_csv)[0]
    greedy_removed_path = os.path.join(output_folder, f"greedy_removed-{dataset_name_with_csv}-{agg_func.upper()}.csv")

    # Load greedy removed rows
    try:
        greedy_removed_df = pd.read_csv(greedy_removed_path)
        greedy_removed_count = len(greedy_removed_df)
        print(f"Loaded {greedy_removed_count} rows removed by greedy algorithm.")
    except FileNotFoundError:
        print(f"Warning: Could not find greedy removed rows file at {greedy_removed_path}")
        return None

    original_greedy_time = None
    for _, row in outlier_results_df.iterrows():
        if row['dataset'] == dataset_path:
            original_greedy_time = row['greedy_time']
            break

    # Create a unique identifier for each row based on grouping_col and agg_col
    greedy_removed_df['row_identifier'] = greedy_removed_df[grouping_col].astype(str) + '_' + greedy_removed_df[
        agg_col].astype(str)
    greedy_identifier_counts = Counter(greedy_removed_df['row_identifier'])

    overlap_results = []

    # Process each outlier detection method and parameter combination
    for _, row in filtered_results.iterrows():
        method = row['method']
        param = row['param']
        group_wise = row['group_wise']
        outlier_time = row['outlier_time']
        greedy_time_after_ol = row['greedy_time']
        outlier_rows_removed = row['outlier_rows_removed']
        greedy_rows_removed_after_ol = row['greedy_rows_removed']

        # Path to the removed rows file for this outlier detection method
        if method == "knn":
            param = int(param)
        method_param_str = f"{method}-{param}"
        group_wise_str = "_group_wise" if group_wise else ""

        #TODO: this compares the removed by th ol alone without the greedy algorithm after
        # I can change it to: greedy_removed-after_ol_{dataset_name}_{method_param_str}_pct100.0_{agg_func}{group_wise_str}-{agg_func.upper()}.csv
        ol_removed_path = os.path.join(output_folder,
                                       f"removed_ol_{dataset_name}_{method_param_str}_pct100.0_{agg_func}{group_wise_str}.csv")

        try:
            ol_removed_df = pd.read_csv(ol_removed_path)
            ol_removed_count = len(ol_removed_df)

            # Create identifier for outlier removed rows
            ol_removed_df['row_identifier'] = ol_removed_df[grouping_col].astype(str) + '_' + ol_removed_df[
                agg_col].astype(str)
            ol_identifier_counts = Counter(ol_removed_df['row_identifier'])

            # Calculate overlap
            common_identifiers = set(greedy_identifier_counts.keys()) & set(ol_identifier_counts.keys())
            overlap_count = sum(min(greedy_identifier_counts[id], ol_identifier_counts[id])
                                for id in common_identifiers)

            # Calculate overlap percentage
            if greedy_removed_count > 0 and ol_removed_count > 0:
                overlap_pct_of_greedy = (overlap_count / greedy_removed_count) * 100
                overlap_pct_of_ol = (overlap_count / ol_removed_count) * 100
            else:
                overlap_pct_of_greedy = 0
                overlap_pct_of_ol = 0

            # Total rows removed by the combined approach
            ol_total_rows_removed = outlier_rows_removed + greedy_rows_removed_after_ol

            # Calculate runtime metrics
            total_ol_runtime = outlier_time + greedy_time_after_ol
            runtime_improvement = original_greedy_time / total_ol_runtime if total_ol_runtime > 0 else float('inf')


            # Add result to collection
            overlap_results.append({
                'method': method,
                'param': param,
                'group_wise': group_wise,
                'ol_removed_count(not including greedy)': ol_removed_count,
                'greedy_rows_removed_after_ol': greedy_rows_removed_after_ol,
                'ol_total_rows_removed': ol_total_rows_removed,
                'greedy_removed_count': greedy_removed_count,
                'overlap_count(greedy vs ol alone)': overlap_count,
                'overlap_pct_of_greedy': f"{overlap_pct_of_greedy:4f}",
                'overlap_pct_of_ol': f"{overlap_pct_of_ol:4f}",
                'outlier_runtime (not including greedy)': f"{outlier_time:4f}",
                'greedy_after_ol_runtime': f"{greedy_time_after_ol:4f}",
                'ol_total_runtime': f"{total_ol_runtime:4f}",
                'greedy_runtime': f"{original_greedy_time:4f}",
                'runtime_improvement': f"{runtime_improvement:4f}",
            })

            print(f"Method: {method}, Param: {param}, Group-wise: {group_wise}")
            print(f"  Greedy removed: {greedy_removed_count}, Outlier removed: {ol_removed_count}")
            print(
                f"  Overlap: {overlap_count} rows ({overlap_pct_of_greedy:.2f}% of greedy, {overlap_pct_of_ol:.2f}% of outlier)")

        except FileNotFoundError:
            print(f"Warning: Could not find outlier removed rows file at {ol_removed_path}")

    # Create DataFrame with results
    if overlap_results:
        overlap_df = pd.DataFrame(overlap_results)

        # Save results to CSV
        overlap_csv_path = os.path.join(output_folder, f"overlap_analysis-{dataset_name}-{agg_func}.csv")
        overlap_df.to_csv(overlap_csv_path, index=False)
        print(f"Overlap analysis saved to {overlap_csv_path}")

        return overlap_df
    else:
        print("No overlap results generated.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier Detection compare")
    parser.add_argument("dataset_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--agg_func", type=str, choices=["max", "sum", "avg", "median"],
                        help="Aggregation function to check monotonicity")
    parser.add_argument("--grouping_col", type=str, default="A", help="Column to group by")
    parser.add_argument("--agg_col", type=str, default="B", help="The column to aggregate")
    parser.add_argument("--output_folder", type=str, default="compare_outlier_results", help="Output folder")

    args = parser.parse_args()

    outlier_results = []

    os.makedirs(args.output_folder, exist_ok=True)
    run_outlier_methods(args.dataset_path, args.agg_func, args.grouping_col, args.agg_col, args.output_folder,
                       outlier_results)

    # Convert results to DataFrame and save
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    outlier_results_df = pd.DataFrame(outlier_results)
    outlier_results_df.to_csv(os.path.join(args.output_folder, f"outlier_results-{dataset_name}-{args.agg_func}.csv"), index=False)

    greedy_results = run_greedy(args.dataset_path, args.agg_func, args.grouping_col, args.agg_col, args.output_folder)
    print("The greedy results are:\n")
    print(greedy_results)

    # Compare the overlap between outlier detection and greedy algorithm
    overlap_results = compare_removed_rows_overlap(args.dataset_path, args.agg_func, args.grouping_col, args.agg_col,
                                                  args.output_folder, outlier_results_df)