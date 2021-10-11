"""Validation module"""
import functools
from operator import index
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import click


def prepare_data(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    calculate_cols = [
        'id', 'tag', 'solver_id', 'benchmark_id', 'task_id', 'anon_1',
        'benchmark_name', 'symbol', 'instance', 'result'
    ]
    return (df[calculate_cols].rename(columns={
        "anon_1": "solver_full_name",
        "symbol": "task"
    }).astype({
        'solver_id': 'int16',
        'benchmark_id': 'int16',
        'task_id': 'int8'
    }))

def analyse(df: pd.DataFrame) -> pd.Series:
    analysis = dict()
    analysis['solver'] = df['solver_full_name'].iloc[0]
    analysis['task'] = df['task'].iloc[0]
    analysis['benchmark_name'] = df['benchmark_name'].iloc[0]
    analysis['correct_solved'] = df[df.correct == 'correct']['correct'].count()
    analysis['incorrect_solved'] = df[df.correct == 'incorrect']['correct'].count()
    analysis['no_reference'] = df[df.correct == 'no_reference']['correct'].count()
    analysis['total'] = df['instance'].count()
    analysis['percentage_validated'] = (analysis['total'] - analysis['no_reference']) / analysis['total'] * 100
    return pd.Series(analysis)





def compare_results_enumeration(actual, correct):
    """[summary]

    Args:
        actual ([type]): [description]
        correct ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not correct:
        return "no_reference"
    if functools.reduce(lambda i, j: i and j,
                        map(lambda m, k: m == k, actual, correct), True):
        return "correct"
    else:
        return "incorrect"


def compare_results_decision(actual, correct):
    """[summary]

    Args:
        actual ([type]): [description]
        correct ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not correct:
        return "no_reference"
    if actual == correct:
        return "correct"
    else:
        return "incorrect"


def get_reference_result_enumeration(path, instance_name, task):
    """[summary]

    Args:
        path ([type]): [description]
        instance_name ([type]): [description]
        task ([type]): [description]

    Returns:
        [type]: [description]
    """
    correct_result_file = ""
    for f in os.listdir(path):
        if (instance_name in f) and ((task + ".out") in f):
            correct_result_file = f
            break
    if correct_result_file:
        with open(os.path.join(path, correct_result_file),encoding='utf-8') as f:
            correct_result = f.read()
        return correct_result
    else:
        return None


def get_reference_result_decision(path, instance_name, task):
    """[summary]

    Args:
        path ([type]): [description]
        instance_name ([type]): [description]
        task ([type]): [description]

    Returns:
        [type]: [description]
    """
    correct_result_file = ""
    for f in os.listdir(path):
        if (instance_name in f) and ((task + ".out") in f):
            correct_result_file = f
            break
    if correct_result_file:
        with open(os.path.join(path, correct_result_file),encoding='utf-8') as f:
            correct_result = f.readline()
        return correct_result.rstrip('\n')
    else:
        return ""


def single_extension_string_to_list(res):
    """[summary]

    Args:
        res ([type]): [description]

    Returns:
        [type]: [description]
    """

    if res.partition('\n')[0] == 'NO':
        return 'NO'
    if not res:
        return []
    extensions = re.findall('\[(.*?)\]', res)

    if extensions[0] == '':
        return []
    else:
        return sorted(extensions[0].split(","))


def multiple_extensions_string_to_list(res):
    """[summary]

    Args:
        res ([type]): [description]

    Returns:
        [type]: [description]
    """

    if res is None:
        return None
    start = res.find('[')
    end = res.rfind(']')
    res = res[start + 1:end]
    extensions = re.findall('\[(.*?)\]', res)
    sorted_extension = []
    for ex in extensions:
        if ex:
            sorted_extension.append(sorted(ex.split(",")))
        else:
            sorted_extension.append([])

    return sorted(sorted_extension)


def parse_result(task: str, result: str):
    """[summary]

    Args:
        task (str): [description]
        result (str): [description]

    Returns:
        [type]: [description]
    """
    if 'EE' in task:
        return multiple_extensions_string_to_list(result)
    elif 'SE' in task:
        print(result)
        return single_extension_string_to_list(result)
    else:
        return result


def validate_ee(df: pd.DataFrame, reference_path: str):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        reference_path (str): [description]

    Returns:
        [type]: [description]
    """

    instance_name = df['instance'].iloc[0]
    task = df['task'].iloc[0]
    reference_result = get_reference_result_enumeration(
        reference_path, instance_name, task)
    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: compare_results_enumeration(
        multiple_extensions_string_to_list(row['result']), reference_extensions
    ),
                    axis=1)

def print_solver_summary(df: pd.DataFrame):
        num_correct = df[df.correct == 'correct']['correct'].count()
        num_incorrect = df[df.correct == 'incorrect']['correct'].count()
        num_no_reference = df[df.correct == 'no_reference']['correct'].count()
        total = df['instance'].count()
        percentage_validated = (total - num_no_reference) / total * 100

        solver = df['solver_full_name'].iloc[0]

        print(f'Solver: {solver}')
        print(f'Total instances: {total}')
        print(f'Correct instances: {num_correct}')
        print(f'Incorrect instances: {num_incorrect}')
        print(f'No reference: {num_no_reference}')
        print(f'Percentage valdated: {percentage_validated}\n')

def print_summary(df: pd.DataFrame):
    task = df['task'].iloc[0]
    print(f'**********{task}***********')
    df.groupby('solver_id').apply(lambda df: print_solver_summary(df))

def pie_chart(df: pd.DataFrame, save_to,title=None):
    save_to = os.path.join(save_to,create_file_name(df))
    labels = ['Correct','Incorrect','No reference']
    total_correct = df['correct_solved'].sum()
    total_incorrect = df['incorrect_solved'].sum()
    total_no_ref = df['no_reference'].sum()
    data = [total_correct,total_incorrect,total_no_ref]

    if total_correct == 0:
       del data[0]
       del labels[0]
    if total_incorrect == 0:
       del data[len(data)-2]
       del labels[len(labels)-2]
    if total_no_ref == 0:
       del data[len(data)-1]
       del labels[len(labels) -1]


    colors = sns.color_palette('Blues')[0:5]
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    if title:
        plt.title(title)
        plt.savefig(f"{save_to}_{title}_pie.png",
                   bbox_inches='tight',
                   transparent=True)
    else:
        title = df['benchmark_name'].iloc[0] +" " + df['task'].iloc[0]
        plt.title(title)
        plt.savefig(f"{save_to}_pie.png",
                   bbox_inches='tight',
                   transparent=True)

    plt.clf()


def count_plot(df: pd.DataFrame,save_to,title=None,grid=False):
    save_to = os.path.join(save_to,create_file_name(df))
    df = df.rename(columns={'solver_full_name': 'Solver', 'correct': 'Status'})
    if grid:
        grid_plot = sns.catplot(x="Solver", hue="Status", col="task",
                data=df, kind="count",
                height=4, aspect=.7)

        grid_plot.set_xticklabels(rotation=40, ha="right")
        figure = grid_plot.fig
        figure.savefig(f"{save_to}_grid.png",
                     bbox_inches='tight',
                   transparent=True)

    else:
        ax = sns.countplot(data=df,x='Solver',hue='Status')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


        for p in ax.patches:
            ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if title:
            ax.set_title(title)
            figure = ax.get_figure()
            figure.savefig(f"{save_to}_{title}.png",
                   bbox_inches='tight',
                   transparent=True)
        else:
            title = df['benchmark_name'].iloc[0] +" " + df['task'].iloc[0]
            ax.set_title(title)
            figure = ax.get_figure()
            figure.savefig(f"{save_to}_{title}.png",
                   bbox_inches='tight',
                   transparent=True)

    plt.clf()

def create_file_name(df: pd.DataFrame) -> str:
    """[summary]

    Args:
        df ([type]): [description]
        post_hoc ([type]): [description]

    Returns:
        [type]: [description]
    """
    tag = df['tag'].iloc[0]
    benchmark = df['benchmark_name'].iloc[0]
    task = df['task'].iloc[0]
    file_name = '{}-{}-{}-{}'.format('_'.join(tag),
                                     '_'.join(benchmark),
                                     '_'.join(task),"validation")
    return file_name



def compare_result_some_extension(actual,correct):
    if correct is None:
        return "no_reference"
    if actual == 'NO' and not correct:
        return 'correct'

    if actual in correct:
        return "correct"
    else:
        return "incorrect"

def validate_se(df, reference_path):
    """[summary]

    Args:
        df ([type]): [description]
        reference_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    instance_name = df['instance'].iloc[0]
    task = df['task'].iloc[0]
    task = 'EE-' + task.split("-")[1]
    reference_result = get_reference_result_enumeration(reference_path, instance_name, task)

    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: compare_result_some_extension(
         single_extension_string_to_list(row['result']), reference_extensions),
                     axis=1)


def validate_decision(df, reference_path):
    """[summary]

    Args:
        df ([type]): [description]
        reference_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    instance_name = df['instance'].iloc[0]
    task = df['task'].iloc[0]
    reference_result = get_reference_result_decision(reference_path,
                                                     instance_name, task)
    print(reference_result.rstrip("\n"))
    return df.apply(lambda row: compare_results_enumeration(
        multiple_extensions_string_to_list(row['result']), reference_result),
                    axis=1)


def validate_instance(df, references):
    """[summary]

    Args:
        df ([type]): [description]
        references ([type]): [description]

    Returns:
        [type]: [description]
    """
    reference_path = references[(df['benchmark_id'].iloc[0])]
    if not os.path.exists(reference_path):
        raise click.BadParameter("Reference path not found!")

    if 'EE' in str(df['task'].iloc[0]):
        df['correct'] = validate_ee(df, reference_path)
    elif 'SE' in str(df['task'].iloc[0]):
        df['correct'] = validate_se(df, reference_path)
    elif 'DC' in str(df['task'].iloc[0]):
        df['correct'] = validate_decision(df, reference_path)
    elif 'DS' in str(df['task'].iloc[0]):
        df['correct'] = validate_decision(df, reference_path)

    return df

def validate(df, references):
    """[summary]

    Args:
        df ([type]): [description]
        references ([type]): [description]

    Returns:
        [type]: [description]
    """
    val = (df
    #[(df["task"].str.contains("DC")) | (df["task"].str.contains("DS")) ]
    .groupby(['benchmark_id','instance','task'])
    .apply(lambda _df: validate_instance(_df,references))
    #[['id', 'tag', 'solver_id', 'solver_full_name','benchmark_name','task','instance','correct']]
    )
    val['validated'] = np.where(val.correct.values == "no_reference", False,
                                True)
    return val


def plot_accordance_heatmap(df, save_to, color_map='Blues', set_over='white'):
    """[summary]

    Args:
        df ([type]): [description]
        save_to ([type]): [description]
        color_map (str, optional): [description]. Defaults to 'Blues'.
        set_over (str, optional): [description]. Defaults to 'white'.
    """
    print(df.dtypes)
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
                     vmax=100,
                     cmap=cmap,
                     cbar_kws={
                         'format': '%.0f%%',
                         'ticks': [0, 100]
                     },
                     **heatmap_args,
                     annot=True,
                     fmt='.2f')
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


def export_result(result: pd.DataFrame, formats: list, save_to: str) -> None:
    """[summary]

    Args:
        result (pd.DataFrame): [description]
        formats (list): [description]
        save_to (str): [description]
    """
    if 'heatmap' in formats:
        plot_accordance_heatmap(result, save_to)
        formats.remove('heatmap')
    if 'html' in formats:
        save_to = save_to + ".html"
        export_styled_table(result,save_to)
        formats.remove('html')

    for export_format in formats:
        getattr(result, "to_" + export_format)(f"{save_to}.{export_format}")


def create_file_name(df: pd.DataFrame)->str:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        str: [description]
    """
    unique_tags = list(df['tag'].unique())
    unique_benchmarks = list(df['benchmark_name'].unique())
    unique_tasks = list(df['task'].unique())
    file_name = '{}-{}-{}'.format('_'.join(unique_tags),
                                  '_'.join(unique_benchmarks),
                                  '_'.join(unique_tasks))
    return file_name


def get_intersection_of_solved_instances(df: pd.DataFrame) -> list:
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


def validate_pairwise(df: pd.DataFrame,
                      export=None,
                      save_to=None) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        export ([type], optional): [description]. Defaults to None.
        save_to ([type], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """
    solvers = sorted(list(df['solver_full_name'].unique()))
    accordance_df = pd.DataFrame(columns=solvers, index=solvers)
    intersection_solved_instances = get_intersection_of_solved_instances(df)
    for solver in solvers:
        accordance_map = dict.fromkeys(solvers, 0.0)
        solver_results = df[df.solver_full_name.values == solver]
        for instance in intersection_solved_instances:

            result_current_solver = parse_result(
                df['task'].iloc[0], solver_results[
                    solver_results.instance.values == instance].result.iloc[0])
            instance_results_other_solvers = df[df.instance.values == instance]
            for index, row in instance_results_other_solvers.iterrows():
                other_solver_result = parse_result(row.task, row.result)
                if 'EE' in row.task or 'SE' in row.task:
                    validation_result = compare_results_enumeration(
                        result_current_solver, other_solver_result)
                if 'DC' in row.task or 'DS' in row.task:
                    validation_result = compare_results_decision(
                        result_current_solver, other_solver_result)
                if validation_result == 'correct':
                    accordance_map[row.solver_full_name] += 1
                else:
                    print(instance)
                    print("Current Solver:", solver)
                    print("Result:", result_current_solver)
                    print("Other solver:", row.solver_full_name)
                    print("Result:", other_solver_result)
                    print(
                        "_----------------------------------------------------"
                    )

        for other_solver in accordance_map:
            accordance_df.at[solver, other_solver] = round(
                (accordance_map[other_solver] /
                 len(intersection_solved_instances)) * 100, 2)

        accordance_df = accordance_df.astype("float64")
    print(accordance_df)
    if export:
        export = list(export)
        file_name = create_file_name(df)
        save_to = os.path.join(save_to, file_name)
        export_result(accordance_df, export, save_to)

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

def report_single_solver(df: pd.DataFrame):
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """

    correct = (df.correct.values == 'correct').sum()
    incorrect = (df.correct.values == 'incorrect').sum()
    no_ref = (df.correct.values == 'no_reference').sum()
    total = correct + incorrect + no_ref
    return correct / total * 100, incorrect / total * 100, no_ref / total * 100


def update_result_object(result_obj, correct, validated):
    """[summary]

    Args:
        result_obj ([type]): [description]
        correct ([type]): [description]
        validated ([type]): [description]
    """
    result_obj.correct = correct
    result_obj.validated = validated
