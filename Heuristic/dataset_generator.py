import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import concurrent.futures
import sys


def create_dataset_simple(num_rows, index, output_folder, violation_percent=None):
    num_groups = 10
    disrupting_groups_count = 4
    if violation_percent is None:
        violation_percent = 0.01 # 1% violation
    violation_rows = int(num_rows * violation_percent)
    regular_rows = num_rows - violation_rows
    groups = [i % num_groups + 1 for i in range(regular_rows)]
    B_values = [random.randint(1, 100) for _ in range(regular_rows)]
    df = pd.DataFrame({"A": groups, "B": B_values})


    group_ids = range(1, num_groups)
    # avoid selecting the last (rightmost) group, since increasing its aggregation value will not create a violation.
    violation_groups = random.sample(group_ids[:-1], disrupting_groups_count)
    violation_data = []
    # Evenly distribute violation rows across the selected groups
    rows_per_group = violation_rows // len(violation_groups)
    remaining_rows = violation_rows % len(violation_groups)

    for group in violation_groups:
        for _ in range(rows_per_group):
            value = random.randint(100, 120)
            violation_data.append({"A": group, "B": value})

    # Distribute remaining rows to random groups
    for _ in range(remaining_rows):
        group = random.choice(violation_groups)
        value = random.randint(100, 120)
        violation_data.append({"A": group, "B": value})

    # Convert the violations into a DataFrame and add to the dataset
    violation_df = pd.DataFrame(violation_data)
    df = pd.concat([df, violation_df], ignore_index=True)

    for agg_func in {'max', 'sum', 'median', 'mean', 'count', 'nunique'}:
        print(agg_func)
        agg_values = df.groupby('A')['B'].agg(agg_func)
        # Step 2: Sort by index or value as needed (default is index order)
        agg_values_sorted = agg_values.sort_index()
        # Step 3: Compare each value with the next one
        comparison = agg_values_sorted.values[:-1] > agg_values_sorted.values[1:]
        # Step 4: Count how many times there is a violation
        count = comparison.sum()
        print(f"count of violations: {count}")
        diffs = agg_values_sorted.values[1:] - agg_values_sorted.values[:-1]
        positive_diffs = diffs[diffs > 0]
        print(f"sum of violations: {sum(positive_diffs)}")


    datasets_output_folder = os.path.join(output_folder, "datasets_r100K")
    os.makedirs(datasets_output_folder, exist_ok=True)
    dataset_filename = os.path.join(datasets_output_folder, f"dataset_generic_g10_r{num_rows}_v{int(violation_percent*100)}_{index}.csv")
    df.to_csv(dataset_filename, index=False)
    print(f"Dataset saved to {dataset_filename}")




def create_dataset(
        num_groups, num_rows, agg_func_name, output_folder="dataset",
        index=0, violations_percentage=10, disrupting_groups_count=None, name_suffix="default"
):
    """
    Create a dataset with columns A and B that is mostly monotonic with respect to the aggregation function.
    A portion of rows will intentionally violate monotonicity.

    Args:
        num_groups (int): Number of groups in the dataset.
        num_rows (int): Total number of rows in the dataset.
        agg_func_name (str): Aggregation function ('sum', 'max', 'avg', 'median').
        output_folder (str): Folder to save the dataset.
        index (int): Index of the dataset (for unique naming).
        violations_percentage (int): Percentage of rows violating monotonicity.
        disrupting_groups_count (int): Number of groups to disrupt monotonicity. Default is all groups.
        name_suffix (str): Suffix for the output filename.
    """
    if agg_func_name not in {"sum", "max", "avg", "median"}:
        raise ValueError("Aggregation function must be 'sum', 'max', 'median', or 'avg'.")

    agg_func_map = {"sum": "sum", "max": "max", "avg": "mean", "median": "median"}
    agg_func = agg_func_map[agg_func_name]

    monotonic_rows = int((100 - violations_percentage) / 100 * num_rows)
    violation_rows = num_rows - monotonic_rows

    if disrupting_groups_count is None:
        disrupting_groups_count = num_groups

    groups = [i % num_groups + 1 for i in range(monotonic_rows)]
    B_values = [random.randint(1, 100) for _ in range(monotonic_rows)]
    df = pd.DataFrame({"A": groups, "B": B_values})

    grouped = df.groupby("A")
    group_agg = grouped["B"].agg(agg_func).reset_index()
    group_agg.sort_values(by="A", inplace=True)

    # Reorder groups to ensure monotonicity
    temp_group_agg = df.groupby("A")["B"].agg(agg_func).reset_index()
    temp_group_agg.sort_values(by="B", inplace=True)

    # Map the reordered group indices back to the DataFrame
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(temp_group_agg["A"], start=1)}
    df["A"] = df["A"].map(group_mapping)

    # Verify the reordering
    # temp_group_agg = df.groupby("A")["B"].agg(agg_func).reset_index()
    # print ("Aggregates after reordering:")
    # print(temp_group_agg)

    assert all(
        temp_group_agg.iloc[i]["B"] <= temp_group_agg.iloc[i + 1]["B"]
        for i in range(len(temp_group_agg) - 1)
    ), "re-ordered dataset is not monotonic."

    group_ids = sorted(list(group_mapping.values()))
    # avoid selecting the last (rightmost) group, since increasing its aggregation value will not create a violation.
    violation_groups = random.sample(group_ids[:-1], disrupting_groups_count)
    violation_data = []
    # Evenly distribute violation rows across the selected groups
    rows_per_group = violation_rows // len(violation_groups)
    remaining_rows = violation_rows % len(violation_groups)

    for group in violation_groups:
        for _ in range(rows_per_group):
            value = random.randint(100, 120)
            violation_data.append({"A": group, "B": value})

    # Distribute remaining rows to random groups
    for _ in range(remaining_rows):
        group = random.choice(violation_groups)
        value = random.randint(100, 120)
        violation_data.append({"A": group, "B": value})

    # Convert the violations into a DataFrame and add to the dataset
    violation_df = pd.DataFrame(violation_data)
    df = pd.concat([df, violation_df], ignore_index=True)

    datasets_output_folder = os.path.join(output_folder, "datasets")
    os.makedirs(datasets_output_folder, exist_ok=True)

    dataset_filename = os.path.join(datasets_output_folder, f"dataset_{name_suffix}_{index}.csv")
    df.to_csv(dataset_filename, index=False)
    print(f"Dataset saved to {dataset_filename}")
   # plot_dataset(df, group_agg, agg_func_name, output_folder, index, name_suffix)

    # temp_group_agg = df.groupby("A")["B"].agg(agg_func).reset_index()
    # print("Aggregates after adding violations:")
    # print(temp_group_agg)


def plot_dataset(df, group_agg, agg_func_name, output_folder, index, name_suffix):
    """
    Generate and save bar and scatter plots of the dataset.

    Args:
        df (pd.DataFrame): The dataset to visualize.
        group_agg (pd.DataFrame): Aggregated group data.
        agg_func_name (str): Aggregation function name.
        output_folder (str): Folder to save plots.
        index (int): Index of the dataset.
        name_suffix (str): Suffix for the plot filenames.
    """
    scatter_plot_folder = os.path.join(output_folder, "scatter_plots")
    bar_plot_folder = os.path.join(output_folder, "bar_plots")
    os.makedirs(scatter_plot_folder, exist_ok=True)
    os.makedirs(bar_plot_folder, exist_ok=True)

    # Scatter plot
    scatter_plot_file = os.path.join(scatter_plot_folder, f"scatter_plot_{name_suffix}_{index}.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["A"], df["B"], alpha=0.6, c="blue", label="Data Points")
    plt.xlabel("A (Groups)")
    plt.ylabel("B (Values)")
    plt.title("Scatter Plot of Dataset")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(scatter_plot_file)
    plt.close()

    # Bar plot
    bar_plot_file = os.path.join(bar_plot_folder, f"bar_plot_{name_suffix}_{index}.png")
    plt.figure(figsize=(8, 6))
    plt.bar(group_agg["A"], group_agg["B"], alpha=0.7, color="green", label=f"Group {agg_func_name.upper()}")
    plt.xlabel("A (Groups)")
    plt.ylabel(f"B ({agg_func_name.upper()})")
    plt.title(f"Bar Plot of Group Aggregation ({agg_func_name.upper()})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(bar_plot_file)
    plt.close()
    print(f"Plots saved to {scatter_plot_file} and {bar_plot_file}")


def generate_dataset(config, param_value, i, agg_func):
    """Helper function to create a dataset with given parameters."""
    create_dataset(
        num_groups=config.get('num_groups', param_value),
        num_rows=config.get('num_rows', param_value),
        agg_func_name=agg_func,
        output_folder=f"synthetic/dataset-{agg_func}/{config['folder']}",
        index=i,
        violations_percentage=config.get('violations_percentage', param_value),
        disrupting_groups_count=3,
        name_suffix=f"{agg_func}_g{config.get('num_groups', param_value)}_r{config.get('num_rows', param_value)}_v{config.get('violations_percentage', param_value)}"
    )


if __name__ == "__main__":
    num_datasets_per_setting = 3

    #for num_rows in range(200000, 1000001, 200000):
    for num_rows in [100000]:
        for violation in [0.01, 0.05, 0.1, 0.15]:
            for i in range(num_datasets_per_setting):
                create_dataset_simple(num_rows, i, output_folder="synthetic/dataset-generic/", violation_percent=violation)
    sys.exit()


    agg_funcs = ["sum", "max", "avg", "median"]
    configurations = [
        { # Rows
            'range': range(100000, 1000001, 100000),
            'folder': 'rows',
            'agg_funcs': agg_funcs,
            'num_groups': 10,
            'violations_percentage': 10,
        },
        { # Groups
            'range': range(5, 55, 5),
            'folder': 'groups',
            'agg_funcs': agg_funcs,
            'num_rows': 1000000,
            'violations_percentage': 10,
        },
        { # Violations
            'range': range(5, 55, 5),
            'folder': 'violations',
            'agg_funcs': agg_funcs,
            'num_groups': 10,
            'num_rows': 1000000,
        }
    ]

    # Use ProcessPoolExecutor to parallelize dataset creation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for config in configurations:
            for param_value in config['range']:
                for i in range(num_datasets_per_setting):
                    for agg_func in config['agg_funcs']:
                        futures.append(executor.submit(generate_dataset, config, param_value, i, agg_func))

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
