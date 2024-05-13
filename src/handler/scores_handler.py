from functools import reduce
import src.functions.register as register
import os
import colorama
import pandas as pd
import click
from tqdm import tqdm
def calculate(scores_to_calculate, raw_results_csv_path: str, save_to: str) -> str:
    """
    Calculate specified scores on the data from a CSV file.
    
    Parameters:
        stats_to_calculate (list or str): A list of scores names to calculate or a string indicating all statistics.
        raw_results_csv_path (str): Path to the CSV file containing raw results.
    
    Returns:
        list: Results of the calculated statistics or an empty list if errors occur.
    """
    if not os.path.isdir(save_to):
        os.makedirs(save_to)
        

    # Check if the result file exists
    if not os.path.exists(raw_results_csv_path):
        print(f"Result file not found: {raw_results_csv_path}")
        return []
    
    # Read the data from the CSV file
    result_df = pd.read_csv(raw_results_csv_path)

    # Determine which statistics to calculate
    if isinstance(scores_to_calculate, str):
        scores_to_calculate = list(register.score_functions_dict.keys())
    
    if not scores_to_calculate:  # Handles None, empty list, or other empty values
        print(f"No stats to calculate")
        return []

    # Calculate the requested statistics
    file_paths = []
    data_frames = []
    for score in tqdm(scores_to_calculate, desc="Calculating scores", 
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (colorama.Fore.LIGHTGREEN_EX, colorama.Fore.RESET)):
        if score in register.score_functions_dict:
            result = register.score_functions_dict[score](result_df)[0]
            file_path = os.path.join(save_to, f"{score}.csv")
            result.to_csv(file_path, index=False)
            file_paths.append(file_path)
            data_frames.append(result)
    
    combined_df = reduce(lambda  left,right: pd.merge(left,right,how='inner'), data_frames)
    combined_path = os.path.join(save_to, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)
    file_paths.append(combined_path)
    return file_path

    

    

    
    