"""Module for significance testing"""
import os
from src.database_models.Benchmark import Benchmark
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns
from src.utils.utils import dispatch_on_value
import click

def create_report(result_dict: dict):
    pass


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    calculate_cols = [
        'id', 'tag', 'solver_id', 'benchmark_id', 'task_id', 'runtime',
        'anon_1', 'benchmark_name', 'symbol','instance'
    ]

    return (df[calculate_cols].rename(columns={
        "anon_1": "solver_full_name",
        "symbol": "task"
    }).astype({
        'solver_id': 'int16',
        'benchmark_id': 'int16',
        'task_id': 'int8',
        'runtime': 'float32'
    }))

def _interpretation_of_result(alpha,p_value):
    if p_value <= alpha:
        return f'Reject null hypothesis at significance level {alpha=}'
    if p_value > alpha:
        return f'Fail to reject null hypothesis at significance level {alpha=}'

def _get_experiment_summary(row: pd.Series) -> str:
    return (f'Tag: {", ".join(row.tag)}\n'
            f'Benchmark: {", ".join(row.benchmark_name)}\n'
            f'Task: {", ".join(row.task)}\n'
            f'Solver: {", ".join(row.solver)}\n'
            f'Test: {(row.kind)}\n'
            )

def print_significance_results(alpha, row):
    #experiment_summary= _get_experiment_summary(row)
    stat = row.stat
    p = row.p_value
    test_statistics=f'{stat=} {p=}'
    result_interpretation = _interpretation_of_result(alpha,p)
    #print("------------------------------------------------------------------------------")
    print(f'Solver: {row.solver}\n{test_statistics}\n\n{result_interpretation}\n')
    #print("------------------------------------------------------------------------------")

def print_post_hoc_results(row: pd.Series):
    experiment_summary= _get_experiment_summary(row)
    ph_result = row.result
    print("------------------------------------------------------------------------------")
    print(f'{experiment_summary}\n{ph_result}\n')
    print("------------------------------------------------------------------------------")


def test(kind_test, df: pd.DataFrame, equal_sample_size: bool)->pd.Series:
    info = get_unique_experiment_infos(df,to_string=True)

    if kind_test=='t-test' and len(info['solver']) > 2:
        raise click.ClickException(
                    "Dataset contains more than two samples. Please use option ANOVA for a parametric test of two or more samples."
                )
    if kind_test=='mann-whitney-u' and len(info['solver']) > 2:
                raise click.ClickException(
                    "Dataset contains more than two samples. Please use option kruskal for a non-parametric test of two or more samples."
                )

    runtimes,num_solved_intersection = extract_runtimes(df,equal_sample_size=equal_sample_size)
    stat, p_value = test_significance(kind_test, runtimes)
    info['num_solved_intersection'] = num_solved_intersection
    info.update({'stat': stat,'p_value': p_value,'kind':kind_test})
    return pd.Series(info)

@dispatch_on_value
def test_significance(kind_test, runtimes):
    print(f"Test {kind_test} not supported.")

@test_significance.register('ANOVA')
def _anova_oneway_parametric(kind_test: str, runtimes)->tuple:
    stat, p_value = ss.f_oneway(*runtimes)
    return stat,p_value

@test_significance.register('kruskal')
def _kruskal_non_parametric(kind_test: str, runtimes)->tuple:
    stat, p_value = ss.kruskal(*runtimes)
    return stat,p_value

@test_significance.register('mann-whitney-u')
def _mann_whitney_u(kind_test, runtimes) -> tuple:
    """[summary]

    Args:
        df ([type]): [description]
        alpha ([type]): [description]
    """
    stat, p_value = ss.mannwhitneyu(*runtimes)
    return stat, p_value

@test_significance.register('t-test')
def _student_t_test(kind_test, runtimes) -> tuple:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        alpha (float): [description]
    """
    stat, p_value = ss.ttest_ind(*runtimes)
    return stat, p_value

def test_post_hoc(df: pd.DataFrame,post_hoc: str, equal_sample_size: bool, p_adjust=None)->dict:
    if not p_adjust:
        p_adjust = 'holm'
    info = get_unique_experiment_infos(df)

    if equal_sample_size:
        intersection_instances = get_intersection_of_solved_instances(df)
        result_post_hoc = _post_hoc_test(df[df.instance.isin(intersection_instances)], post_hoc, p_adjust)
    else:
        result_post_hoc = _post_hoc_test(df, post_hoc, p_adjust)


    info.update({'kind':post_hoc,'p_adjust':p_adjust,'result': result_post_hoc})
    return pd.Series(info)

def _post_hoc_test(df : pd.DataFrame, post_hoc: str, p_adjust: str)->pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        post_hoc (str): [description]
        p_adjust (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    result = getattr(sp,
                         "posthoc_" + post_hoc)(df,
                                                val_col='runtime',
                                                group_col='solver_full_name',
                                                p_adjust=p_adjust)

    result = result.sort_index()[sorted(result.columns)]
    return result


def get_intersection_of_solved_instances(df: pd.DataFrame)-> list:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        list: [description]
    """
    set_list = []
    for name, group in df.groupby('solver_id'):
        set_list.append(set(group['instance'].unique()))
    return list(set.intersection(*set_list))

def extract_runtimes(df: pd.DataFrame,equal_sample_size=None) -> list:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        list: [description]
    """
    if equal_sample_size is None:
        equal_sample_size = True
    unique_solver_ids = sorted(df['solver_id'].unique())
    run_times = []
    if equal_sample_size:
        instances = get_intersection_of_solved_instances(df)

        for u_id in unique_solver_ids:
            run_times.append((df[(df.solver_id == u_id) &
                                 (df.instance.isin(instances))].runtime.values))
        return run_times,len(instances)

    else:
        for u_id in unique_solver_ids:
            run_times.append(df[df.solver_id == u_id].runtime.values)
        return run_times,len(list(df['instance'].unique()))

def get_unique_experiment_infos(df,to_string=None):

    if to_string is None:
        to_string=False
    unique_tags = list(df['tag'].unique())
    unique_benchmarks = list(df['benchmark_name'].unique())
    unique_tasks = list(df['task'].unique())
    unique_solver_names = list(df['solver_full_name'].unique())
    if to_string:
        unique_tags = ','.join(unique_tags)
        unique_benchmarks = ','.join(unique_benchmarks)
        unique_tasks = ','.join(unique_tasks)
        unique_solver_names = ','.join(unique_solver_names)

    return {'tag':unique_tags, 'benchmark_name':unique_benchmarks,'task':unique_tasks,'solver': unique_solver_names}

def plot_heatmap(df: pd.DataFrame,
                 save_to: str,
                 color_map='Blues_r',
                 set_over='white'):
    """Create a heatmap plot of p-values and save it.

        Args:
            df: A pandas.Dataframe instance containing p-values of post-hoc pairwise comparision.
            save_to: Save path.
            color_map: Name of Seaborn color palatte.
            set_over: Color to set cell to if p-value is not significant.

        Returns:
            None

        Raises:
            None
    """
    heatmap_args = {
        'linewidths': 0.25,
        'linecolor': '0.5',
        'clip_on': False,
        'square': True
    }
    cmap = sns.mpl.cm.get_cmap(color_map).copy()
    cmap.set_over(set_over)
    ax = sns.heatmap(df,
                     vmin=0,
                     vmax=0.05,
                     cmap=cmap,
                     cbar_kws={'label': 'p-value'},
                     **heatmap_args,
                     annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       rotation_mode='anchor',
                       ha='right')
    ax.figure.axes[-1].yaxis.label.set_size(15)

    figure = ax.get_figure()
    figure.savefig(f"{save_to}.png",
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()


def export_styled_table(df: pd.DataFrame, save_to: str):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        save_to (str): [description]
    """
    table_html_string = df.to_html(escape=False)
    with open(
            "/home/jklein/new_dev/Probo2_1.0/src/css/tables/styled-table.css",
            encoding="utf-8") as css_file:
        css_file_content = css_file.read()
        css_file_content = "<style>\n" + css_file_content + "\n</style>"

    table_class = "styled-table"
    table_html_string = table_html_string.replace(
        "<table ", "<table class=\"" + table_class + "\" ")

    with open(save_to, "w", encoding="utf-8") as file_to_write:
        file_to_write.write(css_file_content + table_html_string)


def export_result(result: pd.DataFrame, formats: list, save_to: str) -> None:
    """[summary]

    Args:
        result (pd.DataFrame): [description]
        formats (list): [description]
        save_to (str): [description]
    """
    if 'heatmap' in formats:
        plot_heatmap(result, save_to)
        formats.remove('heatmap')
    if 'html' in formats:
        save_to = save_to + ".html"
        export_styled_table(result, save_to)
        formats.remove('html')

    for export_format in formats:
        getattr(result,
                "to_" + export_format)("{}.{}".format(save_to, export_format))


def create_file_name(df: pd.DataFrame, post_hoc: str) -> str:
    """[summary]

    Args:
        df ([type]): [description]
        post_hoc ([type]): [description]

    Returns:
        [type]: [description]
    """
    unique_tags = list(df['tag'].unique())
    unique_benchmarks = list(df['benchmark_name'].unique())
    unique_tasks = list(df['task'].unique())
    file_name = '{}-{}-{}-{}'.format('_'.join(unique_tags),
                                     '_'.join(unique_benchmarks),
                                     '_'.join(unique_tasks), post_hoc)
    return file_name

def _gen_file_name(df):
     return (f'{"_".join(df.tag)}_{"_".join(df.task)}_{"_".join(df.benchmark_name)}_ {(df.kind)}')

@dispatch_on_value
def export(export_format, df_to_export, save_to):
    pass

@export.register('csv')
def csv_export(export_format, df_to_export,save_to):
    df_to_export.to_csv(save_to,index=False)


@dispatch_on_value
def print_result(type:str, df: pd.DataFrame, alpha=None):
    pass


@print_result.register('significance')
def _print_significance_results(type:str, df: pd.DataFrame, alpha=None):
    if alpha is None:
        alpha = 0.05
    df.groupby('kind').apply(lambda _df: _print_result_per_test(_df, alpha))

def _print_result_per_test(df,alpha):
    kind = df.kind.iloc[0]
    print(f'\n***** Test: {kind} *****')
    df.groupby('tag').apply(lambda _df: _print_result_per_tag(_df, alpha))

def _print_result_per_tag(df,alpha):
    tag = df.tag.iloc[0]
    print(f'\n++++ Tag: {tag} ++++')
    df.groupby('benchmark_id').apply(lambda _df: _print_result_per_benchmark(_df, alpha))

def _print_result_per_benchmark(df,alpha):
    benchmark = df.benchmark_name.iloc[0]
    benchmark_id = df.benchmark_id.iloc[0]
    print(f'\n--- Benchmark: {benchmark} (ID: {benchmark_id}) ---')
    df.groupby('task').apply(lambda _df: _print_result_per_task(_df, alpha))

def _print_result_per_task(df,alpha):
    task = df.task.iloc[0]
    task_id = df.task_id.iloc[0]
    print(f'\n· Task: {task} (ID: {task_id})\n')
    print_significance_results(alpha,df.iloc[0])
    #df.groupby('benchmark_id').apply(lambda _df: _print_result_per_benchmark(_df))