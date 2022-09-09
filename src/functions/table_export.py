import pandas
import src.functions.register as register
import pandas as pd
from src.utils import definitions
import src.utils.config_handler as config_handler
import tabulate
import os
import yaml
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
    save_to = os.path.join(config.save_to, 'tables')
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
    save_to = os.path.join(config.save_to, 'tables')
    os.makedirs(save_to, exist_ok=True)

    saved_files = df.groupby(grouping).apply(lambda _df: _save_as_csv(_df,save_to))
    return saved_files



def pretty_latex(df: pd.DataFrame,config: config_handler.Config, grouping=None):
    if grouping is None:
        grouping = ['tag','task','benchmark_name']
    save_to = os.path.join(config.save_to, 'tables')
    os.makedirs(save_to, exist_ok=True)
    with open(definitions.PRETTY_LATEX_TABLE_CONFIG,'r') as config_file:
        table_config = yaml.load(config_file, Loader=yaml.FullLoader)
    #table_config = yaml.load(definitions.PRETTY_LATEX_TABLE_CONFIG, Loader=yaml.FullLoader)
    print(table_config)
    saved_files = df.groupby(grouping).apply(lambda _df: _save_as_pretty_latex(_df,save_to, table_config))
    return saved_files




def max_value_bold(data, column_max):
    if data == column_max:
        return f"\\textbf{{{data}}}"

    return data

def min_value_bold(data, column_min):
    if data == column_min:
        return  f"\\textbf{{{data}}}"
    return data


def column_header_bold(df: pd.DataFrame) -> list:
    return (df
            .columns.to_series()
            .apply(lambda r: "\\textbf{{{0}}}".format(
            r.replace("_", " ").title())))

def _save_as_pretty_latex(df:pd.DataFrame, save_to,table_config):
    df_columns = df.columns
    print(df)
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    task = df.task.iloc[0]
    if table_config['max_bold'] is not None:
        for k in  table_config['max_bold']:
            if k in df_columns:
                df[k] = df[k].apply(
                lambda data: max_value_bold(data, column_max=df[k].max()))
    if table_config['min_bold'] is not None:
        for k in  table_config['min_bold']:
            if k in df_columns:
                df[k] = df[k].apply(
                lambda data: min_value_bold(data, column_min=df[k].min()))

    if table_config['name_map'] is not None:
        df.rename(columns= table_config['name_map'],inplace=True)

    if table_config['header_bold']:
        df.columns = column_header_bold(df)


    filename = os.path.join(save_to,f'{tag}_{task}_{benchmark}.tex')
    if not filename:
        filename = 'data_tbl.tex'
    if table_config['caption']:
        caption = 'Your caption could be here'
    if table_config['label']:
        label = 'Your label'

    with open(filename,"w") as latex_file:

        # format_tbl = "l" + \
        #     "@{\hskip 12pt}" +\
        #     4*"S[table-format = 2.2]"
        replace_map = {'#': '\#', '%': '\%','_': ' '}
        latex_tbl = (df
                        .to_latex(index=False,
                          escape=False,
                          caption=caption,
                          label=label,
                          column_format=f'l*{{{df.shape[1]-1}}}{{c}}'))
        trans_table = latex_tbl.maketrans(replace_map)
        latex_tbl = latex_tbl.translate(trans_table)
        latex_file.write(latex_tbl)
    return filename




register.table_export_functions_register('txt',text)
register.table_export_functions_register('csv',csv)
register.table_export_functions_register('pretty_latex',pretty_latex)

