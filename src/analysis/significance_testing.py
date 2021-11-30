"""Module for significance testing"""
import os
from src.database_models.Benchmark import Benchmark
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns


def print_result_dict(result_dict: dict)-> None:
    """[summary]

    Args:
        result_dict (dict): [description]
    """
    parametric = result_dict['parametric']
    non_parametric = result_dict['non_parametric']
    if parametric:
        print("**********PARAMETRIC TESTS RESULTS**********")
        result_print(parametric)
    if non_parametric:
        print("**********NON-PARAMETRIC TESTS RESULTS**********")
        result_print(non_parametric)


def result_print(parametric):
    for res in parametric:
       res_info = res['info']
       tag = ', '.join(res_info['tag'])
       task = ', '.join(res_info['task'])
       bench = ', '.join(res_info['benchmark_name'])
       solv = ', '.join(res_info['solver'])
       num_solved_by_all = res_info['num_solved_intersection']
       test = res['Test']
       p_value = res['p-value']
       stat = res['stat']
       alpha = res['alpha']
       print(f'Tag: {tag}\nTask: {task}\nBenchmark: {bench}\nSolver: {solv}\n#Solved by all: {num_solved_by_all}\nTest: {test}\nP-value: {p_value}\nStat:{stat}\nAlpha: {alpha}\n')
       print_result(alpha,stat,p_value)
       if res['Post-hoc']:
           ph = res['Post-hoc']
           p_adjust = res['p-adjust']

           print(f'Post-hoc: {ph}\nP-adjust: {p_adjust}\n')
           print('Result:\n')
           print(res['post_hoc_result'])
           print("")
       print("------------------------------------------------------------------------------------------------")


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

def extract_runtimes(df: pd.DataFrame,intersection_solved_instances=True) -> list:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        list: [description]
    """
    unique_solver_ids = sorted(df['solver_id'].unique())
    run_times = []
    if intersection_solved_instances:
        instances = get_intersection_of_solved_instances(df)
        for u_id in unique_solver_ids:
            run_times.append((df[(df.solver_id == u_id) &
                                 (df.instance.isin(instances))].runtime.values))
        return run_times,len(instances)

    else:
        for u_id in unique_solver_ids:
            run_times.append(df[df.solver_id == u_id].runtime.values)
        return run_times,len(list(df['instance'].unique()))



def print_info(df, test, alpha):
    """[summary]

    Args:
        df ([type]): [description]
        test ([type]): [description]
        alpha ([type]): [description]
    """
    info = get_unique_experiment_infos(df)
    print("----------{}----------".format(test))
    print("Significance level:", alpha)
    print("Tag:", ", ".join(info['tag']))
    print("Benchmark:", ", ".join(info['benchmark_name']))
    print("Task:", ", ".join(info['task']))
    print("Solver:", ", ".join(info['solver']))

def get_unique_experiment_infos(df):
    unique_tags = list(df['tag'].unique())
    unique_benchmarks = list(df['benchmark_name'].unique())
    unique_tasks = list(df['task'].unique())
    unique_solver_names = list(df['solver_full_name'].unique())
    return {'tag':unique_tags, 'benchmark_name':unique_benchmarks,'task':unique_tasks,'solver': unique_solver_names}


def student_t_test(df: pd.DataFrame, alpha: float):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        alpha (float): [description]
    """
    runtimes = extract_runtimes(df)
    stat, p_value = ss.ttest_ind(*runtimes)
    print_info(df, "Student’s t-test", alpha)
    print("")
    print('stat=%.3f, p=%.3f' % (stat, p_value))
    if p_value > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    print("")


def mann_whitney_u(df, alpha):
    """[summary]

    Args:
        df ([type]): [description]
        alpha ([type]): [description]
    """
    runtimes = extract_runtimes(df)
    stat, p_value = ss.mannwhitneyu(*runtimes)
    print_info(df, "Mann-Whitney U Test", alpha)

    print("")
    print_result(alpha, stat, p_value)
    print("")


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


def kruskal_non_parametric(df: pd.DataFrame,
                           alpha: float,
                           post_hoc=None,
                           p_adjust=None,
                           export=None,
                           save_to=None):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        alpha (float): [description]
        post_hoc ([type], optional): [description]. Defaults to None.
        p_adjust ([type], optional): [description]. Defaults to None.
        export ([type], optional): [description]. Defaults to None.
        save_to ([type], optional): [description]. Defaults to None.
    """
    info = get_unique_experiment_infos(df)
    runtimes, num_solved_intersection = extract_runtimes(df)
    info['num_solved_intersection'] = num_solved_intersection
    stat, p_value = ss.kruskal(*runtimes)
    if post_hoc:
        result = post_hoc_test(df,post_hoc,p_adjust)
        if export:
            export = list(export)
            file_name = create_file_name(df, post_hoc)
            save_to = os.path.join(save_to, file_name)
            export_result(result, export, save_to)

        return {"post_hoc_result": result, "Test": 'Kruskal', "Post-hoc": post_hoc, "p-adjust":p_adjust, "p-value": p_value, "stat" : stat, "alpha": alpha,"info":info}
    return {"post_hoc_result": None, "Test": 'Kruskal', "Post-hoc": None, "p-adjust":None, "p-value": p_value, "stat" : stat, "alpha": alpha, "info":info}

def anova_oneway_parametric(df: pd.DataFrame,
                            alpha: float,
                            post_hoc=None,
                            p_adjust=None,
                            export=None,
                            save_to=None):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        alpha (float): [description]
        post_hoc (str, optional): [description]. Defaults to None.
        p_adjust (str, optional): [description]. Defaults to None.
        export (str, optional): [description]. Defaults to None.
        save_to (str, optional): [description]. Defaults to None.
    """
    info = get_unique_experiment_infos(df)
    runtimes,num_solved_intersection = extract_runtimes(df)
    stat, p_value = ss.f_oneway(*runtimes)
    info['num_solved_intersection'] = num_solved_intersection

    if post_hoc:

        result = post_hoc_test(df, post_hoc, p_adjust)
        if export:
            export = list(export)
            file_name = create_file_name(df, post_hoc)
            save_to = os.path.join(save_to, file_name)
            export_result(result, export, save_to)
        return {"post_hoc_result": result, "Test": 'ANOVA', "Post-hoc": post_hoc, "p-adjust":p_adjust, "p-value": p_value, "stat" : stat, "alpha": alpha, 'info': info}
    return  {"post_hoc_result": None, "Test": 'ANOVA', "Post-hoc": None, "p-adjust":None, "p-value": p_value, "stat" : stat, "alpha": alpha,'info': info}

def post_hoc_test(df : pd.DataFrame, post_hoc: str, p_adjust: str)->pd.DataFrame:
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


def print_result(alpha: float, stat: float, p: float):
    """[summary]

    Args:
        alpha (float): [description]
        stat (float): [description]
        p (float): [description]
    """
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    print("")


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
