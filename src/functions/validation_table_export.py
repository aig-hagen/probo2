import src.functions.register as register
import pandas as pd
from src.utils import definitions
import src.handler.config_handler as config_handler
import tabulate
import os
from src.handler.benchmark_handler import load_benchmark_by_identifier

def _save_as_text(df: pd.DataFrame, save_to):
    tag = df.tag.iloc[0]
    task = df.task.iloc[0]

    benchmark = df.benchmark_name.iloc[0]

    for description in ['same_results','solved_pairwise','accordance']:
        tbl_text = tabulate.tabulate(df[description].iloc[0],headers='keys',tablefmt='pretty',showindex=False)
        file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{description}.txt')
        with open(file_path,'w') as f:
            f.write(f"Tag: {tag}\nTask: {task}\nBenchmark: {benchmark}\n{tbl_text}")

    return file_path

def text(df: pd.DataFrame,config: config_handler.Config):
    save_to = os.path.join(config.save_to, 'validation_tables')
    os.makedirs(save_to, exist_ok=True)

    saved_files = df.groupby(['tag','task','benchmark_name']).apply(lambda _df: _save_as_text(_df,save_to))
    return saved_files

def _save_as_csv(df: pd.DataFrame, save_to):

    tag = df.tag.iloc[0]

    task = df.task.iloc[0]

    benchmark = df.benchmark_name.iloc[0]

    for description in ['same_results','solved_pairwise','accordance']:
        file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{description}.csv')
        with open(file_path,'w') as f:
            f.write(df[description].iloc[0].to_csv(index=False))

    return file_path

def csv(df: pd.DataFrame,config: config_handler.Config):
    save_to = os.path.join(config.save_to, 'validation_tables')
    os.makedirs(save_to, exist_ok=True)
    saved_files = df.groupby(['tag','task','benchmark_name']).apply(lambda _df: _save_as_csv(_df,save_to))
    return saved_files


def _save_as_latex(df: pd.DataFrame, save_to):

    tag = df.tag.iloc[0]

    task = df.task.iloc[0]

    benchmark = df.benchmark_name.iloc[0]

    for description in ['same_results','solved_pairwise','accordance']:
        file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{description}.tex')
        with open(file_path,'w') as f:
            f.write(df[description].iloc[0].to_latex(index=False))

    return file_path

def latex(df: pd.DataFrame,config: config_handler.Config):
    save_to = os.path.join(config.save_to, 'validation_tables')
    os.makedirs(save_to, exist_ok=True)
    saved_files = df.groupby(['tag','task','benchmark_name']).apply(lambda _df: _save_as_latex(_df,save_to))
    return saved_files

register.validation_table_export_functions_register('txt',text)
register.validation_table_export_functions_register('csv',csv)
register.validation_table_export_functions_register('latex',latex)


