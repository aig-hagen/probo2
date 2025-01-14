import pandas as pd
from typing import Any,List
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from pathlib import Path
import os
import tabulate


# def _save_as_text(df: pd.DataFrame, save_to):
#     tag = df.tag.iloc[0]
#     benchmark = df.benchmark_name.iloc[0]
#     task = df.task.iloc[0]

#     tbl_text = tabulate.tabulate(df,headers='keys',tablefmt='pretty',showindex=False)

#     file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}.txt')
#     with open(file_path,'w') as f:
#         f.write(tbl_text)

#     return file_path

# def text(df: pd.DataFrame,config: config_handler.Config, grouping=None):
#     if grouping is None:
#         grouping = ['tag','task','benchmark_name']
#     save_to = os.path.join(config.save_to, 'stats_tables')
#     os.makedirs(save_to, exist_ok=True)

#     saved_files = df.groupby(grouping).apply(lambda _df: _save_as_text(_df,save_to))
#     return saved_files

class PlainTextTable(Callback):
    def __init__(self, grouping: List[str] = None):
        """
        Initializes the PlainTextTable callback.
        Args:
            grouping (List[str], optional): A list of strings to group the results by. Defaults to None.
        """
        self.grouping = grouping if grouping else []


    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        # Read files from config.evaluation_combined_results_file if the file does not exits present, if not return None
        if config.evaluation_combined_results_file is None:
            return None
        # Check if the file path is valid an the file exists
        if not Path(config.evaluation_combined_results_file).exists():
            # Print a message to the console
            print(f"File {config.evaluation_combined_results_file} does not exist")
            return None
        # Read the file into an pandas dataframe
        df = pd.read_csv(config.evaluation_combined_results_file)
        #print(df.to_markdown(os.path.join(config.evaluation_combined_results_file,'table.md')))

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
            print(self.grouping)
        return


    def _save_as_text(self,df: pd.DataFrame, save_to,grouping,tablefmt='pretty'):
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