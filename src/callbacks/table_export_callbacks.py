import pandas as pd
from typing import Any,List
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from pathlib import Path
import os
import tabulate

"""
PlainTextTable Callback

The PlainTextTable class is a custom Hydra callback designed to export grouped evaluation results
from a CSV file into plain text files. It reads a results file, groups the data based on specified
columns, and saves each group as a formatted text file. Additionally, it prints the grouped data
to the console for quick visualization.

### Features:
- Reads evaluation results from a CSV file specified in the configuration.
- Dynamically groups data by specified columns (`grouping`) or defaults to `['task', 'benchmark_name']`.
- Exports each group as a plain text file in a specified output directory.
- Prints grouped data to the console for immediate inspection.

### Usage Example in Hydra Config:
hydra:
  callbacks:
    plain_text:
      _target_: src.callbacks.table_export_callbacks.PlainTextTable
      grouping: ['task', 'benchmark']
"""
class PlainTextTable(Callback):
    def __init__(self, grouping: List[str] = None):
        """
        Initializes the PlainTextTable callback.
        Args:
            grouping (List[str], optional): A list of strings to group the results by. Defaults to None.
        """
        self.grouping = grouping if grouping else []


    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Exports results as plain text.

        Args:
            config (DictConfig): Configuration object containing evaluation settings.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None: If the evaluation_combined_results_file is not specified or does not exist.

        The function performs the following steps:
        1. Checks if the evaluation_combined_results_file is specified in the config. If not, returns None.
        2. Validates if the specified file path exists. If not, prints a message and returns None.
        3. Reads the evaluation results file into a pandas DataFrame.
        4. If no grouping is specified, sets the default grouping to ['task', 'benchmark_name'].
        5. Groups the DataFrame by the specified grouping and saves each group as a text file.
        6. Iterates through the groups and prints each group along with its keys and the grouping used.
        """
        # Read the file from config.evaluation_combined_results_file if it exists, otherwise return None
        if config.evaluation_combined_results_file is None:
            # Print a message to the console
            print(f"Combined result File is None")
            return None
        # Check if the file path is valid and the file exists
        if not Path(config.evaluation_combined_results_file).exists():
            # Print a message to the console
            print(f"File {config.evaluation_combined_results_file} does not exist")
            return None
        # Read the file into an pandas dataframe
        df = pd.read_csv(config.evaluation_combined_results_file)

        # Print the dataframe to the console
        if len(self.grouping) == 0:
            self.grouping = ['task','benchmark_name']

        self.grouping = list(self.grouping)

        save_to = config.evaluation_output_dir
        grouped_df = df.groupby(self.grouping)
        grouped_df.apply(lambda _df: self._save_as_text(_df, save_to, self.grouping))


        # Iterate through groups and print them
        for group_keys, group_data in grouped_df:
            print(f"\nGroup: {group_keys}")
            print(group_data)

    def _save_as_text(self, df: pd.DataFrame, save_to, grouping, tablefmt='pretty'):
        """
        Save a pandas DataFrame as a text file with a specified table format.

        Parameters:
        df (pd.DataFrame): The DataFrame to be saved.
        save_to (str): The directory path where the file will be saved.
        grouping (list): A list of column names used to generate the file name.
        tablefmt (str, optional): The format of the table. Default is 'pretty'.

        Returns:
        str: The file path of the saved text file.
        """
        # Extract all the keys from the grouping list to generate the file name
        keys = [df[key].iloc[0] for key in grouping]
        # Generate the file name from the keys
        file_name = '_'.join(keys)
        # Add file extension to file name
        file_name += '.txt'
        # Generate the table text
        tbl_text = tabulate.tabulate(df,headers='keys',tablefmt=tablefmt,showindex=False)

        file_path = os.path.join(save_to,file_name)
        with open(file_path,'w') as f:
            f.write(tbl_text)

        return file_path