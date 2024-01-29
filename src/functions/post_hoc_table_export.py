import pandas
import src.functions.register as register
import pandas as pd
from src.utils import definitions
import src.handler.config_handler as config_handler
import tabulate
import os
from src.handler.benchmark_handler import load_benchmark_by_identifier
def _save_as_text(df: pd.DataFrame, save_to, test):
    tag = df.tag.iloc[0]
    task = df.task.iloc[0]
    rep = df.repetition.iloc[0]
    benchmark = load_benchmark_by_identifier([int(df.benchmark_id.iloc[0])])[0]['name']

    tbl_text = tabulate.tabulate(df['result'].iloc[0],headers='keys',tablefmt='pretty',showindex=False)

    file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{test}_{rep}.txt')
    with open(file_path,'w') as f:
        f.write(tbl_text)

    return file_path

def text(df: pd.DataFrame,config: config_handler.Config,test):
    save_to = os.path.join(config.save_to, 'post_hoc_tables')
    os.makedirs(save_to, exist_ok=True)

    saved_files = df.groupby(['repetition','tag','task','benchmark_id']).apply(lambda _df: _save_as_text(_df,save_to,test))
    return saved_files

def _save_as_csv(df: pd.DataFrame, save_to,test):

    tag = df.tag.iloc[0]

    task = df.task.iloc[0]
    rep = df.repetition.iloc[0]
    benchmark = load_benchmark_by_identifier([int(df.benchmark_id.iloc[0])])[0]['name']


    file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{test}_{rep}.csv')
    with open(file_path,'w') as f:
        f.write(df['result'].iloc[0].to_csv(index=False))

    return file_path

def csv(df: pd.DataFrame,config: config_handler.Config,test):
    save_to = os.path.join(config.save_to, 'post_hoc_tables')
    os.makedirs(save_to, exist_ok=True)
    saved_files = df.groupby(['repetition','tag','task','benchmark_id']).apply(lambda _df: _save_as_csv(_df,save_to,test))
    return saved_files


def _save_as_latex(df: pd.DataFrame, save_to, test):

    tag = df.tag.iloc[0]

    task = df.task.iloc[0]
    rep = df.repetition.iloc[0]
    benchmark = load_benchmark_by_identifier([int(df.benchmark_id.iloc[0])])[0]['name']


    file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}_{test}_{rep}.tex')
    with open(file_path,'w') as f:
        f.write(df['result'].iloc[0].to_latex(index=False))

    return file_path

def latex(df: pd.DataFrame,config: config_handler.Config,test):
    save_to = os.path.join(config.save_to, 'post_hoc_tables')
    os.makedirs(save_to, exist_ok=True)
    saved_files = df.groupby(['repetition','tag','task','benchmark_id']).apply(lambda _df: _save_as_latex(_df,save_to,test))
    return saved_files

register.post_hoc_table_export_functions_register('txt',text)
register.post_hoc_table_export_functions_register('csv',csv)
register.post_hoc_table_export_functions_register('latex',latex)


