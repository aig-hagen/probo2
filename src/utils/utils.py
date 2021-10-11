import os
from pathlib import Path

import click
import pandas


def export_csv(df: pandas.DataFrame,
               save_to: str,
               columns=None,
               index=False,
               encoding='utf-8',
               file_name="csv_data.csv",
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

    if columns:
        df = df[columns]

    save_path = os.path.join(save_to, file_name)

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

    if columns:
        df = df[columns]
    save_path = os.path.join(save_to, file_name)

    if css_file:
        name_css_file = Path(css_file).stem
        table_html_string = df.to_html(escape=False,
                                       index=index,
                                       classes=name_css_file)

        with open(css_file, encoding="utf-8") as file:
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
                           session=session)
                click.echo("---FINISHED")
