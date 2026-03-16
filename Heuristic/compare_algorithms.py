import os
import sys
import pandas as pd
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import argparse

# Function to run an algorithm and measure rows removed and execution time
def run_algorithm(command, is_dp=False, timeout=600):
    start_time = time.time()
    print(command)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=timeout)
        #result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, text=True, timeout=timeout)
        end_time = time.time()
        execution_time = end_time - start_time

        rows_removed = None
        mem_usage = None
        for line in result.stdout.splitlines():
            if "Total tuple removals" in line:
                rows_removed = int(float(line.split(":")[-1].strip()))
            elif "Num removed tuples" in line:
                rows_removed = int(float(line.split(":")[-1].strip().split("/")[0]))
            elif "maximal memory usage" in line:
                mem_usage = int(line.split(":")[-1].strip())
        if rows_removed is None:
            return {"time": None, "rows_removed": None, "mem_usage":None}  # Return None for failure

        return {"time": execution_time, "rows_removed": rows_removed, "mem_usage": mem_usage}

    except (subprocess.TimeoutExpired, ValueError) as e:
        print(f"Error running command: {e}")
        return {"time": None, "rows_removed": None, "mem_usage": None}  # Return None for failure


# Helper function to process a single dataset
def process_single_dataset(dataset_path, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder,
                           group_column, agg_column, output_folder, dp_variations=False, aggpack_variations=True,
                           deduce_setting_from_agg=False):
    print(f"Processing dataset: {dataset_path}")
    filename = os.path.basename(dataset_path)

    try:
        # Extract parameters from filename
        num_rows = int(filename.split("_")[3][1:])
        num_groups = int(filename.split("_")[2][1:]) if "_g" in filename else None
        violation_percentage = int(filename.split("_")[4][1:]) if "_v" in filename else None
        index = int(filename.split("_")[-1].split(".")[0])  # Extract index from filename
    except:
        df = pd.read_csv(dataset_path, index_col=0)
        num_rows = len(df)
        num_groups = df[group_column].nunique()
        violation_percentage = "nat"
        index = filename.split(".")[0].split("index")[1]

    result_summary = {
        "dataset": filename,
        "agg": agg_function,
        "num_rows": num_rows,
        "num_groups": num_groups,
        "violation_percentage": violation_percentage,
        "index": index,}
    
    # Run greedy algorithm
    greedy_results = {'time': 'NA', 'rows_removed': 'NA'}
    greedy_command = [
        "python", greedy_algo_path, dataset_path,
        agg_function.lower(),
        "--grouping_column", group_column,
        "--aggregation_column", agg_column,
        "--output_folder", output_folder,
    ]
    if agg_function.lower() in ['avg', 'sum', 'median', 'max']:
        greedy_results = run_algorithm(greedy_command, is_dp=False, timeout=timeout)
        result_summary['greedy_time'] = greedy_results['time']
        result_summary['greedy_rows_removed'] = greedy_results['rows_removed']
        greedy_prune = str(greedy_results["rows_removed"])
    else:
        greedy_prune = None

    dp_command = [
        "python", dp_algo_path, agg_function.upper(), dataset_path, agg_column,
        group_column, "--cutoff_seconds", str(timeout), "--output_folder", output_folder,
    ]
    dp_command.append('--mem_opt')
    # Enable/disable agg pack optimizations.
    
    if deduce_setting_from_agg:
        if agg_function.upper() == 'MAX':
            prune_aggpack_by_greedy = False
            optimize_aggpack = False
        elif agg_function.upper() == 'SUM':
            prune_aggpack_by_greedy = False
            optimize_aggpack = True
        else:
            prune_aggpack_by_greedy = True
            optimize_aggpack = True
    else:
        prune_aggpack_by_greedy = False
        optimize_aggpack = False
    
    if prune_aggpack_by_greedy:
        dp_command.extend(['--prune_aggpack_by_greedy', greedy_prune])  # available for sum, avg and median only
    if optimize_aggpack:
        dp_command.append('--agg_pack_opt')  # available for sum, median, and avg but for sum does not work with pruning by greedy
    #dp_command.append('--prune_h')
    #dp_variations = False
    #aggpack_variations = True
    skip_until = 0

    if dp_variations or aggpack_variations or deduce_setting_from_agg:
        # Naive DP (no pruning)
        dp_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        dp_results = run_algorithm(dp_command, is_dp=True, timeout=timeout)
        print(dp_results)
        result_summary['naive_dp_time'] = dp_results['time']
        result_summary['naive_dp_rows_removed'] = dp_results['rows_removed']
        result_summary['naive_dp_mem_usage'] = dp_results['mem_usage']

    if dp_variations:
        # DP with greedy pruning
        dp1_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        #if num_rows >= skip_until:
        dp1_results = run_algorithm(dp_command + ['--prune_dp_by_greedy', greedy_prune], is_dp=True, timeout=timeout)
        print(dp1_results)
        result_summary['dp_greedy_prune_time'] = dp1_results['time']
        result_summary['dp_greedy_prune_rows_removed'] = dp1_results['rows_removed']
        result_summary['dp_greedy_prune_mem_usage'] = dp1_results['mem_usage']

        # DP with H pruning
        dp2_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        dp2_results = run_algorithm(dp_command + ['--prune_h'], is_dp=True, timeout=timeout)
        print(dp2_results)
        result_summary['dp_hprune_time'] = dp2_results['time']
        result_summary['dp_hprune_rows_removed'] = dp2_results['rows_removed']
        result_summary['dp_hprune_mem_usage'] = dp2_results['mem_usage']
    
        # DP with H pruning + greedy pruning 
        dp3_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        dp3_results = run_algorithm(dp_command + ['--prune_h', '--prune_dp_by_greedy', greedy_prune], is_dp=True, timeout=timeout)
        print(dp3_results)
        result_summary['dp_greedy+hprune_time'] = dp3_results['time']
        result_summary['dp_greedy+hprune_rows_removed'] = dp3_results['rows_removed']
        result_summary['dp_greedy+hprune_mem_usage'] = dp3_results['mem_usage']

    if aggpack_variations:
        # Agg pack prune by greedy
        dp4_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        #if num_rows >= skip_until:
        dp4_results = run_algorithm(dp_command + ['--prune_aggpack_by_greedy', greedy_prune], is_dp=True, timeout=timeout)
        print(dp4_results)
        result_summary['dp_greedy_aggpack_prune_time'] = dp4_results['time']
        result_summary['dp_greedy_aggpack_prune_rows_removed'] = dp4_results['rows_removed']
        result_summary['dp_greedy_aggpack_prune_mem_usage'] = dp4_results['mem_usage']

        # Agg pack optimization (knapsack/histogram)
        # already run for sum
        dp5_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
        #if num_rows>skip_until:
        dp5_results = run_algorithm(dp_command + ['--agg_pack_opt'], is_dp=True, timeout=timeout)
        print(dp5_results)
        result_summary['dp_aggpack_opt_time'] = dp5_results['time']
        result_summary['dp_aggpack_opt_rows_removed'] = dp5_results['rows_removed']
        result_summary['dp_aggpack_opt_mem_usage'] = dp5_results['mem_usage']

        # Agg pack optimizations (combined)
        # already run in DP_variations (naive)
        if agg_function.upper() in ('MEDIAN', 'AVG'):
            dp6_results = {'time': 'NA', 'rows_removed': 'NA', 'mem_usage': 'NA'}
            dp6_results = run_algorithm(dp_command + ['--agg_pack_opt', '--prune_aggpack_by_greedy', greedy_prune], is_dp=True, timeout=timeout)
            print(dp6_results)
            result_summary['dp_aggpack_opt+greedy_aggpack_prune_time'] = dp6_results['time']
            result_summary['dp_aggpack_opt+greedy_aggpack_prune_rows_removed'] = dp6_results['rows_removed']
            result_summary['dp_aggpack_opt+greedy_aggpack_prune_mem_usage'] = dp6_results['mem_usage']
    
    return result_summary


def process_datasets_parallel(dataset_folder, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, grouping_column="A", aggregation_column="B", output_folder="output"):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []

    print(f"Found {len(dataset_paths)} datasets. Processing in parallel...")
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(process_single_dataset, dataset, greedy_algo_path, dp_algo_path, agg_function, timeout,
                            results_folder, grouping_column, aggregation_column, output_folder): dataset
            for dataset in dataset_paths
        }
        for future in as_completed(future_to_dataset):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing dataset {future_to_dataset[future]}: {e}")

    return pd.DataFrame(results)


def fname_to_num_tuples(fname):
    if "_sample_" in fname:
        return int(fname.split("_sample_")[1].split("_")[0])
    else:
        return int(os.path.basename(fname).split("_r")[1].split("_v")[0])


def process_datasets_serial(dataset_folder, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, grouping_column="A", aggregation_column="B", output_folder="output"):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []
    output_path = os.path.join(output_folder, f"aggpack_variations_comparison_results_{agg_function}.csv")

    print(f"Found {len(dataset_paths)} datasets. Processing serially...")
    dataset_paths = sorted(dataset_paths, key=fname_to_num_tuples)
    for path in dataset_paths:
        fname = os.path.basename(path)
        result = process_single_dataset(path, greedy_algo_path, dp_algo_path, agg_function, timeout,
                            results_folder, grouping_column, aggregation_column, output_folder)
        if not os.path.exists(output_path):
            headers = sorted(result.keys())
            pd.DataFrame(columns=headers).to_csv(output_path, index=False)
        pd.DataFrame([result], columns=headers).to_csv(output_path, mode='a', index=False, header=False)
        results.append(result)
    return pd.DataFrame(results)


def compare_aggregations(dataset_folder, greedy_algo_path, dp_algo_path, timeout, results_folder, grouping_column="A", aggregation_column="B", output_folder="output"):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []
    output_path = os.path.join(output_folder, "aggs_comparison_results.csv")

    print(f"Found {len(dataset_paths)} datasets. Processing serially...")
    dataset_paths = sorted(dataset_paths, key=fname_to_num_tuples)
    for path in dataset_paths:
        fname = os.path.basename(path)

        for agg in ["max", "median", "sum"]:
               # , "avg", "count", "count distinct"]:
            result = process_single_dataset(path, greedy_algo_path, dp_algo_path, agg, timeout,
                                results_folder, grouping_column, aggregation_column, output_folder, 
                                deduce_setting_from_agg=True)
            if not os.path.exists(output_path):
                headers = sorted(result.keys())
                pd.DataFrame(columns=headers).to_csv(output_path, index=False)
            pd.DataFrame([result], columns=headers).to_csv(output_path, mode='a', index=False, header=False)
            results.append(result)
    return pd.DataFrame(results)



def plot_results(results, output_folder, x_axis, x_label, agg_function):
    print(f"Preparing data for plots with {x_axis} as x-axis...")

    # Ensure numeric data for aggregation
    numeric_columns = ["num_rows", "num_groups", "violation_percentage", "greedy_rows_removed", "dp_rows_removed", "greedy_time", "dp_time"]
    results = results[numeric_columns].dropna(subset=[x_axis])

    # Reset index to avoid duplicate index issues
    results = results.reset_index(drop=True)

    # Calculate the percentage of rows removed
    results["greedy_rows_removed_pct"] = (results["greedy_rows_removed"] / results["num_rows"]) * 100
    results["dp_rows_removed_pct"] = (results["dp_rows_removed"] / results[
        "num_rows"]) * 100 if "dp_rows_removed" in results else None

    # Group by the specified x-axis and calculate the mean
    grouped = results.groupby(x_axis).mean().sort_index()

    # Plot 1: Percentage of Rows Removed vs. x_axis
    plt.figure(figsize=(24, 14))
    plt.plot(grouped.index, grouped["greedy_rows_removed_pct"], marker='o', label="Greedy Algorithm", linewidth=6)
    if "dp_rows_removed_pct" in grouped and not grouped["dp_rows_removed_pct"].isna().all():
        plt.plot(grouped.index, grouped["dp_rows_removed_pct"], marker='o', linestyle="--", label="DP Algorithm",
                 linewidth=6)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel("Mean Percentage of Rows Removed (%)", fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.title(f"Percentage of Rows Removed by Algorithms ({x_label})", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, linewidth=4)
    plt.savefig(os.path.join(output_folder, f"rows_removed_percentage_comparison_{agg_function}_{x_axis}.pdf"), format='pdf')
    plt.show()

    # Plot 2: Execution Time vs. x_axis (Logarithmic Scale)
    plt.figure(figsize=(24, 14))
    plt.plot(grouped.index, grouped["greedy_time"], marker='o', label="Greedy Algorithm", linewidth=6)
    plt.plot(grouped.index, grouped["dp_time"], marker='o', linestyle="--", label="DP Algorithm", linewidth=6)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel("Mean Execution Time (seconds)", fontsize=28)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.title(f"Execution Time of Algorithms ({x_label}) [Log Scale]", fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True, linewidth=4)
    plt.savefig(os.path.join(output_folder, f"execution_time_comparison_{agg_function}_{x_axis}(log).pdf"), format='pdf')
    plt.show()

    print(f"Plots saved successfully under {output_folder} folder.")

#python/py -3.13 compare_algorithms.py --agg_function <agg_function> --dataset_folder <dataset_folder> --results_folder <results_folder> --timeout_min <timeout_min> --grouping_column <grouping_column> --aggregation_column <aggregation_column>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Greedy and DP Algorithms")
    parser.add_argument("--agg_function", type=str, choices=["max", "sum", "avg", "median"], required=True, help="The aggregation function to use")
    parser.add_argument("--dataset_folder", type=str, default="datasets", help="The folder containing the datasets")
    parser.add_argument("--results_folder", type=str, default="results", help="The folder to store results")
    parser.add_argument("--timeout_min", type=int, default=600, help="The timeout for each algorithm run in minutes")
    parser.add_argument("--grouping_column", type=str, default="A", help="The column to group by")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The column to aggregate")
    parser.add_argument("--compare_aggregations", type=bool, default=False, help="run over all supported aggregations")
    args = parser.parse_args()
    
    agg_function = args.agg_function
    results_base_folder = args.results_folder
    timeout = args.timeout_min * 60
    grouping_column = args.grouping_column
    aggregation_column = args.aggregation_column
    
    dataset_folder = args.dataset_folder
    output_folder = results_base_folder
    os.makedirs(output_folder, exist_ok=True)
    
    if args.compare_aggregations:
        compare_aggregations(dataset_folder, f"aggr-main.py", "../DP/main.py", timeout, results_folder=output_folder,
                             grouping_column="A", aggregation_column="B", output_folder=output_folder)
    else:
        results = process_datasets_serial(dataset_folder, f"aggr-main.py", "../DP/main.py",
                                            agg_function, timeout=timeout, results_folder=output_folder,
                                            grouping_column=grouping_column, aggregation_column=aggregation_column,
                                            output_folder=output_folder)
    sys.exit()



    base_folder = args.dataset_folder
    result_folder = os.path.join(results_base_folder, agg_function)

    for folder in ["rows", "groups", "violations"]:
        dataset_folder = os.path.join(base_folder, folder, "datasets")
        output_folder = os.path.join(result_folder, folder)
        os.makedirs(output_folder, exist_ok=True)

        results = process_datasets_parallel(dataset_folder, f"aggr-main.py", "Trendline-Outlier-Detection/main.py",
                                            agg_function, timeout=timeout, results_folder=result_folder,
                                            grouping_column=grouping_column, aggregation_column=aggregation_column,
                                            output_folder=output_folder)
        if results.empty:
            print(f"No results found for {folder} datasets. Skipping plotting results...")
            continue

        results.to_csv(os.path.join(output_folder, f"comparison_results-{agg_function}-{folder}.csv"), index=False)
        if folder == "rows":
            plot_results(results, output_folder, "num_rows", "Number of Rows", agg_function)
        elif folder == "groups":
            plot_results(results, output_folder, "num_groups", "Number of Groups", agg_function)
        elif folder == "violations":
            plot_results(results, output_folder, "violation_percentage", "Violation Percentage", agg_function)
