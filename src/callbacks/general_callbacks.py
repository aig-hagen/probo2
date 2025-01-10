import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

class AggregateResults(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        index_file = config['result_index_file']
        output_file = config['combined_results_file']
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
                        df = pd.read_csv(file_path)
                        dataframes.append(df)
                        print(f"Loaded {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            # Combine all dataframes
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                # Save to the output CSV file
                combined_df.to_csv(output_file, index=False)
                print(f"Results aggregated into {output_file}")
            else:
                print("No dataframes to aggregate.")

        except Exception as e:
            print(f"An error occurred: {e}")


class AggregateResults2(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print("Second Callback")


class AggregateResults3(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print("Third Callback")