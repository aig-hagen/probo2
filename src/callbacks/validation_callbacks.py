import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
import os
from utils import hydra_utils
import re

from hydra.core.hydra_config import HydraConfig



class ValidateSolutionsAccaptanceTasks(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
            # Initialize HydraConfig if not already set

        result_file = config["combined_results_file"]
        if not os.path.exists(result_file):
            print(f"File {result_file} does not exist")
            return None
        combined_results_df = pd.read_csv(result_file)

        benchmark_name = combined_results_df.benchmark_name.unique()[0]

        # read benchmark config from config.root_dir.benchmark_config
        benchmark_config = OmegaConf.load(os.path.join(config.root_dir, "benchmark_config", f"{benchmark_name}_config.yaml"))

        print(benchmark_config)

        # Validate the solver results against given solutions
        # Check if in the benchmarks config a solutions_path is given
        if "solution_path" in benchmark_config.keys():
            # for file in os.listdir(benchmark_config.solution_path):
            #     if file.endswith(".sol"):
            #         print(file)


            # Check if solution_path is not empty and the path exists
            if benchmark_config.solution_path == "" or not os.path.exists(
                benchmark_config.solution_path
            ):
                raise FileNotFoundError("The solution_path is not valid")
            else:
                # Read every file with the following pattern <instance_name>_<task>.sol

                # Read combined results of solvers in a dataframe

                # Make sure that the exot_with_error and timed_out columns are boolean
                combined_results_df["exit_with_error"] = combined_results_df[
                    "exit_with_error"
                ].astype(bool)
                combined_results_df["timed_out"] = combined_results_df[
                    "timed_out"
                ].astype(bool)

                # Remove instances where the solver exited with an error and timed out
                # combined_results_df_clean = combined_results_df[
                #     combined_results_df["exit_with_error"] == False & combined_results_df["timed_out"] == False
                # ]

                # # Filter df for DC and DS tasks
                # accaptance_df = combined_results_df_clean[
                #     combined_results_df_clean["task"].str.contains("DC|DS")
                # ]

                unique_tasks = combined_results_df["task"].unique()

                for file in os.listdir(benchmark_config.solution_path):
                    if file.endswith(".sol"):
                        for task in unique_tasks:

                            solutions_df = load_solutions(
                                benchmark_config.solution_path, task
                            )
                            print(
                                f"Loaded {len(solutions_df)} solutions for task {task}"
                            )
                            print(solutions_df)
                            # # Merge the results with the solutions
                            # merged_df = pd.merge(combined_results_df_clean, df, how='inner', left_on=['benchmark_name'], right_on=['instance'])
                            # # Check if the solution is correct
                            # merged_df['correct'] = merged_df['solution'] == merged_df['status']
                            # # Calculate the percentage of correct solutions
                            # correct_percentage = merged_df['correct'].mean()
                            # print(f'The percentage of correct solutions for task {task} is {correct_percentage}')
                            # # Save the statistics to a csv file
                            # correct_solution_result_file = os.path.join(config['evaluation_output_dir'], 'correct_solution.csv')
                            # merged_df.to_csv(correct_solution_result_file)
                            # # Write filepath to evaluation file index
                            # hydra_utils.write_evaluation_file_to_index(correct_solution_result_file, config['evaluation_result_index_file'])


def check_solution_content_acceptance(df: pd.DataFrame, task: str) -> pd.Series:

    if "DC" in task or "DS" in task:
        valid_solutions = df["solution"].isin(["YES", "NO"])
        return valid_solutions

    return pd.Series([False] * len(df), index=df.index)


def load_solutions(
    directory: str,
    task: str,
    expected_instances=None,
    missing_file: str = "missing_instances.txt",
) -> pd.DataFrame:
    """
    Reads all files in 'directory' that follow the naming convention:
        <anything>_<task>.sol
    For example, if task="DS-CO", it will match:
        BA_23423_234234_DS-CO.sol
    The part before "_<task>.sol" is treated as the 'instance' name.
    Parameters
    ----------
    directory : str
        Path to the directory containing the solution files.
    task : str
        The task name used in the file naming convention (<anything>_<task>.sol).
    expected_instances : list or set, optional
        A list or set of instance names you expect. If provided, any of these
        that are not found in the directory will be written out as "missing."
        If None, we'll simply parse whatever is present in the directory.
    missing_file : str, optional
        Filename (or full path) to write the missing instance names.
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: [instance, solution].
        'instance' is derived from the filename, and 'solution' is the file contents.
    """
    # Regex to match filenames like <anything>_<task>.sol, where <anything> can be any string
    #   '(.+)' captures everything until '_<task>.sol'
    pattern = re.compile(rf"^(.+)_{re.escape(task)}\.sol$")
    if expected_instances is None:
        expected_instances = set()
    found_data = []
    found_instances = set()
    # Look through the directory
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            # group(1) = everything before `_<task>.sol`
            instance_name = match.group(1)
            file_path = os.path.join(directory, fname)
            with open(file_path, "r") as f:
                content = f.read().strip()
            found_data.append({"instance": instance_name, "solution": content})
            found_instances.add(instance_name)
    # Convert results to a DataFrame
    df = pd.DataFrame(found_data, columns=["instance", "solution"])
    # Check solutions are strictly YES or NO
    valid_solutions = check_solution_content_acceptance(df, task)
    if not valid_solutions.empty and not valid_solutions.all():
        invalid = df[~valid_solutions]
        print("WARNING: Some solutions are not 'YES' or 'NO':")
        print(invalid)
        # Or raise an error: raise ValueError("Some solutions are not 'YES' or 'NO'.")
    # Identify missing instances if we have an expected set
    if expected_instances:
        expected_set = set(expected_instances)
        missing = sorted(expected_set - found_instances)
        if missing:
            print(
                f"{len(missing)} expected instance(s) are missing. Writing them to {missing_file}"
            )
            missing_path = os.path.join(directory, missing_file)
            with open(missing_path, "w") as fp:
                for m in missing:
                    fp.write(m + "\n")
        else:
            print("No missing instance files.")
    return df
