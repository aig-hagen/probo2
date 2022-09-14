from email import header
import src.functions.register as register
import pandas as pd
import tabulate

def print_results(validation_results: pd.DataFrame):
    validation_modes = validation_results.keys()
    print("========== Validation Summary ==========")
    for m in validation_modes:
        register.print_validation_functions_dict[m](validation_results[m])



def _print_pairwise(df: pd.DataFrame):
    print(f"Task: {df.task.iloc[0]}")
    print(f"Benchmark: {df.benchmark_name.iloc[0]}")
    print(f"Pairwise solved instances:\n{tabulate.tabulate(df['solved_pairwise'].iloc[0],headers='keys',tablefmt='pretty')}\n")
    print(f"Pairwise accordance (total):\n{tabulate.tabulate(df['same_results'].iloc[0],headers='keys',tablefmt='pretty')}\n")
    print(f"Pairwise accordance (%):\n{tabulate.tabulate(df['accordance'].iloc[0],headers='keys',tablefmt='pretty')}\n")
    
    

def print_pairwise_results(validation_results: pd.DataFrame):
    validation_results.groupby(['tag','task','benchmark_name']).apply(lambda _df: _print_pairwise(_df))
    

register.print_validation_functions_register('pairwise',print_pairwise_results)