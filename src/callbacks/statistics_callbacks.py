
import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import os
from utils import hydra_utils
from functools import reduce



class All(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]

        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            total_runs=('run', 'count'),
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
            num_timed_out=('timed_out', lambda x: x.sum()),
            num_not_timed_out=('timed_out', lambda x: (~x).sum()),
            proportion_timed_out=('timed_out', 'mean')
        ).reset_index()

        print(stats)

class MeanRuntime(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]

        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
        ).reset_index()

        print(stats)

        # Save the statistics to a csv file
        mean_runtime_result_file = os.path.join(config['evaluation_output_dir'], 'mean_runtime.csv')
        stats.to_csv(mean_runtime_result_file)

        # Write filepath to evaluation file index
        hydra_utils.write_evaluation_file_to_index(mean_runtime_result_file, config['evaluation_result_index_file'])


class Coverage(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]

        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            coverage=('timed_out',  lambda x: (~x).mean())
        ).reset_index()

        # Save the statistics to a csv file
        coverage_result_file = os.path.join(config['evaluation_output_dir'], 'coverage.csv')
        stats.to_csv(coverage_result_file)

        # Write filepath to evaluation file index
        hydra_utils.write_evaluation_file_to_index(coverage_result_file, config['evaluation_result_index_file'])


class AggreateEvaluationResults(Callback):
    '''This callback is used to aggregate the evaluation results into a single file'''
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        index_file = config['evaluation_result_index_file']
        output_file = config['evaluation_combined_results_file']
        print(f"Aggregating evaluation results from {index_file} into {output_file}")
        try:
        # List to store individual dataframes
            dataframes = []

         # Read the index file line by line
            with open(index_file, "r") as file:
                file_paths = file.readlines()

         # Process each CSV file
            for file_path in file_paths:
                file_path = file_path.strip()  # Remove any leading/trailing whitespace
                if file_path:  # Ensure the line is not empty
                    try:
                        # Read the CSV file
                        df = pd.read_csv(file_path,index_col=0)
                        dataframes.append(df)
                        print(f"Loaded {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            # Combine all dataframes
            if dataframes:
                # Assuming 'dataframes' is a list of DataFrames and 'key_column' is the column to merge on
                combined_df = reduce(lambda left, right: pd.merge(left, right, on=['task','benchmark_name','solver_name'], how='inner'), dataframes)
                # Save to the output CSV file
                combined_df.to_csv(output_file, index=False)
                print(f"Results aggregated into {output_file}")
            else:
                print("No dataframes to aggregate.")

        except Exception as e:
            print(f"An error occurred: {e}")

