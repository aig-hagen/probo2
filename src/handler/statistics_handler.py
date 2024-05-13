from functools import reduce
import src.functions.register as register
import os
import colorama
import pandas as pd
import click
from tqdm import tqdm
def calculate(stats_to_calculate, raw_results_csv_path: str, save_to: str) -> tuple:
    """
    Calculate specified statistics on the data from a CSV file.
    
    Parameters:
        stats_to_calculate (list or str): A list of statistic names to calculate or a string indicating all statistics.
        raw_results_csv_path (str): Path to the CSV file containing raw results.
    
    Returns:
        tuple: combined results path, list of individual result paths
    """
    if not os.path.isdir(save_to):
        os.makedirs(save_to)
        

    # Check if the result file exists
    if not os.path.exists(raw_results_csv_path):
        print(f"Result file not found: {raw_results_csv_path}")
        return None,[]
    
    # Read the data from the CSV file
    result_df = pd.read_csv(raw_results_csv_path)

    # Determine which statistics to calculate
    if isinstance(stats_to_calculate, str):
        stats_to_calculate = list(register.stat_dict.keys())
    
    if not stats_to_calculate:  # Handles None, empty list, or other empty values
        print(f"No stats to calculate")
        return None,[]

    # Calculate the requested statistics
    file_paths = []
    data_frames = []


    for stat in tqdm(stats_to_calculate, desc="Calculating stats", 
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (colorama.Fore.LIGHTGREEN_EX, colorama.Fore.RESET)):
        if stat in register.stat_dict:
            result = register.stat_dict[stat](result_df)[0]
            file_path = os.path.join(save_to, f"{stat}.csv")
            result.to_csv(file_path, index=False)
            file_paths.append(file_path)
            data_frames.append(result)
    
    combined_df = reduce(lambda  left,right: pd.merge(left,right,how='inner'), data_frames)
    combined_path = os.path.join(save_to, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)
    file_paths.append(combined_path)
    return combined_path, file_paths

    

    

    
    