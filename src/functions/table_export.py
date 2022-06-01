import pandas
import src.functions.register as register
import pandas as pd
import src.utils.config_handler as config_handler
import tabulate
import os
def _save_as_text(df: pd.DataFrame, save_to):
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    task = df.task.iloc[0]

    tbl_text = tabulate.tabulate(df,headers='keys',tablefmt='fancy_grid',showindex=False)

    file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}.txt')
    with open(file_path,'w') as f:
        f.write(tbl_text)

    return file_path

def text(df: pd.DataFrame,config: config_handler.Config, grouping=None):
    if grouping is None:
        grouping = ['tag','task','benchmark_name']
    save_to = os.path.join(config.save_to, 'statistics_tables')
    os.makedirs(save_to, exist_ok=True)

    saved_files = df.groupby(grouping).apply(lambda _df: _save_as_text(_df,save_to))
    return saved_files

def _save_as_csv(df: pd.DataFrame, save_to):

    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    task = df.task.iloc[0]


    file_path = os.path.join(save_to,f'{tag}_{task}_{benchmark}.csv')
    with open(file_path,'w') as f:
        f.write(df.to_csv(index=False))

    return file_path

def csv(df: pd.DataFrame,config: config_handler.Config, grouping=None):
    if grouping is None:
        grouping = ['tag','task','benchmark_name']
    save_to = os.path.join(config.save_to, 'statistics_tables')
    os.makedirs(save_to, exist_ok=True)

    saved_files = df.groupby(grouping).apply(lambda _df: _save_as_csv(_df,save_to))
    return saved_files






register.table_export_functions_register('txt',text)
register.table_export_functions_register('csv',csv)

