import os
from pathlib import Path
from tabulate import tabulate

import click
import pandas
from src.utils import definitions as definitions

def print_df(df, grouping, headers, format='fancy_grid'):
    (df
    .groupby(grouping)
    [headers]
    .apply(lambda df: print(tabulate(df,headers='keys',tablefmt=format))))

def export(df: pandas.DataFrame,
               formats,
               save_to: str,
               file_name=None,
               columns=None,
               index=False,
               css_file=None
               ) -> None:
    """[summary]

    Args:
        df (pandas.DataFrame): [description]
        formats ([type]): [description]
        save_to (str): [description]
        columns ([type], optional): [description]. Defaults to None.
        index (bool, optional): [description]. Defaults to False.
    """



    if not columns:
        columns = list(df.columns)

    if 'html' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_html(df,save_to,css_file=css_file,index=index,file_name=file_name)))
    if 'csv' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_csv(df,save_to,index=index,file_name=file_name)))
    if 'latex' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_latex(df,save_to,index=index,file_name=file_name)))
def export_latex(df: pandas.DataFrame,
               save_to: str,
               columns=None,
               index=False,
               encoding='utf-8',
               file_name=None) -> None:
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)
    if columns:
        df = df[columns]

    save_path = os.path.join(save_to, f'{file_name}.tex')
    tasks = list(df.task.unique())
    benchs = list(df.benchmark.unique())

    task_string = ",".join(tasks)
    bench_string = ",".join(benchs)

    caption_text = f'Results for tasks {task_string} and benchmark {bench_string}'

    df = df.drop(columns=['tag','benchmark','task'])
    df.to_latex(save_path,
              index=index,
              caption=caption_text,
              encoding=encoding)





def export_csv(df: pandas.DataFrame,
               save_to: str,
               columns=None,
               index=False,
               encoding='utf-8',
               file_name=None,
               compression=None) -> None:
    """[summary]

    Args:
        df (pandas.DataFrame): [description]
        save_to (str): [description]
        columns ([type], optional): [description]. Defaults to None.
        index (bool, optional): [description]. Defaults to False.
        encoding (str, optional): [description]. Defaults to 'utf-8'.
        file_name ([type], optional): [description]. Defaults to None.
        compression ([type], optional): [description]. Defaults to None.

    Returns:
        bool: [description]
    """
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)
    if columns:
        df = df[columns]

    save_path = os.path.join(save_to, f'{file_name}.csv')

    df.to_csv(save_path,
              index=index,
              compression=compression,
              encoding=encoding)


def export_html(df: pandas.DataFrame,
                save_to: str,
                columns=None,
                index=False,
                encoding='utf-8',
                file_name="html_data.html",
                css_file=None):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        save_to (str): [description]
    """
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)

    if columns:
        df = df[columns]
    save_path = os.path.join(save_to, f'{file_name}.html')

    if css_file:
        css_file_path = os.path.join(definitions.CSS_TEMPLATES_PATH,"tables",css_file)
        name_css_file = Path(css_file_path).stem
        table_html_string = df.to_html(escape=False,
                                       index=index,
                                       classes=name_css_file)

        with open(css_file_path, encoding="utf-8") as file:
            content_css_file = file.read()
        content_css_file = "<style>\n" + content_css_file + "\n</style>"

        with open(save_path, "w", encoding=encoding) as file_to_write:
            file_to_write.write(content_css_file + table_html_string)
    else:
        df.to_html(save_path, index=index, encoding=encoding)


def run_experiment(parameters: dict):
    """[summary]

    Args:
        parameters (dict): [description]
    """
    tag = parameters['tag']
    tasks = parameters['task']
    benchmarks = parameters['benchmark']
    timeout = parameters['timeout']
    dry = parameters['dry']
    session = parameters['session']
    select = parameters['select']
    n = parameters['n_times']

    for task in tasks:
        task_symbol = task.symbol.upper()
        click.echo(f"**********{task_symbol}***********")
        for bench in benchmarks:
            click.echo(f"Benchmark: {bench.benchmark_name}")
            if select:
                solvers_to_run = set(task.solvers).intersection(
                    set(parameters['solver']))
            else:
                print(task.solvers)
                solvers_to_run = task.solvers

            for solver in solvers_to_run:
                click.echo(solver.solver_full_name, nl=False)
                solver.run(task,
                           bench,
                           timeout,
                           save_db=(not dry),
                           tag=tag,
                           session=session,
                           n=n)
                click.echo("---FINISHED")

def create_file_name(df: pandas.DataFrame) -> str:
    """[summary]

    Args:
        df ([type]): [description]
        post_hoc ([type]): [description]

    Returns:
        [type]: [description]
    """
    print(df)
    unique_tags = "_".join(list(df['tag'].unique()))
    unique_benchmarks = "_".join(list(df['benchmark'].unique()))
    unique_tasks = "_".join(list(df['task'].unique()))
    file_name = f'{unique_tags}-{unique_benchmarks}-{unique_tasks}'
    return file_name

def dispatch_on_value(func):
    """
    Value-dispatch function decorator.

    Transforms a function into a value-dispatch function,
    which can have different behaviors based on the value of the first argument.
    """

    registry = {}

    def dispatch(value):

        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):

        if func is None:
            return lambda f: register(value, f)

        registry[value] = func

        return func

    def wrapper(*args, **kw):
        return dispatch(args[0])(*args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper
