"""Validation module"""
import functools
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import click

from itertools import chain
from glob import glob

from src.utils.utils import dispatch_on_value
from src.reporting import pretty_latex_table
import src.utils.utils as utils

def test_table_export(df, save_to):
    (pretty_latex_table.generate_table(df[['solver','correct','incorrect','no_reference']],
                                       save_to, 'test_table.tex', max_bold=['correct','incorrect','no_reference'],
                                       columns_headers_bold=True,caption='Test caption',label='TEST_LABEL'))

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

@dispatch_on_value
def validate_task(task, df, references ,file_extension):
    print(f"Task {task} not supported for validation.")

@validate_task.register("EE")
def validate_ee(task, df: pd.DataFrame, references : dict, file_extension):
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
        references, instance_name, task,ref_file_extension=file_extension)
    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: compare_results_enumeration(
        multiple_extensions_string_to_list(row['result']), reference_extensions
    ),
                    axis=1)
@validate_task.register("SE")
def validate_se(task,df, references,extension):
    """[summary]

    Args:
        df ([type]): [description]
        reference_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    instance_name = df['instance'].iloc[0]
    task = df['task'].iloc[0]
    task = 'EE-' + task.split("-")[1] # Use also the results from EE as SE is not unique
    reference_result = get_reference_result_enumeration(references, instance_name, task,ref_file_extension=extension)

    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: compare_result_some_extension(
         single_extension_string_to_list(row['result']), reference_extensions),
                     axis=1)

def analyse(df: pd.DataFrame) -> pd.Series:
    analysis = dict()
    analysis['solver'] = df['solver_full_name'].iloc[0]
    analysis['task'] = df['task'].iloc[0]
    analysis['benchmark_name'] = df['benchmark_name'].iloc[0]
    analysis['correct'] = df['correct'].sum()
    analysis['incorrect'] = df['incorrect'].sum()
    analysis['no_reference'] = df['no_reference'].sum()
    analysis['total'] = df['instance'].count()
    analysis['percentage_validated'] = ( analysis['correct'] + analysis['incorrect'] ) / analysis['total'] * 100
    return pd.Series(analysis)

validaten_result_interpretation = {1:'correct', 0:'incorrect',-1: 'no_reference'}

def compare_results_enumeration(actual, correct):
    """[summary]

    Args:
        actual ([type]): [description]
        correct ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not correct:
        return -1
    if functools.reduce(lambda i, j: i and j,
                        map(lambda m, k: m == k, actual, correct), True):
        return 1
    else:
        return 0


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


def get_reference_result_enumeration(references: dict, instance_name: str, task: str,ref_file_extension=None):
    """[summary]

    Args:
        path ([type]): [description]
        instance_name ([type]): [description]
        task ([type]): [description]

    Returns:
        [type]: [description]
    """
    if ref_file_extension is None or (not ref_file_extension):
        ref_file_extension = references.keys()

    correct_result_file = ""
    print(f'{instance_name=} {task=} {ref_file_extension=}')
    for f_extension in ref_file_extension:
        print(references[f_extension].keys())
        if task in references[f_extension].keys():
            print(task)
            references_file_extensions_task = references[f_extension][task]
            for reference_result in references_file_extensions_task:
                if all(substring in reference_result for substring in [instance_name,task,f_extension]):
                    with open(reference_result,'r',encoding='utf-8') as ref_file:
                        ref_result_str = ref_file.read()
                    return ref_result_str
    return None

def get_reference_result_decision(path, instance_name, task, extension='out'):
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
        if (instance_name in f) and ((task + f".{extension}") in f):
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


def multiple_extensions_string_to_list(res: str):
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
    if correct is None: # No reference file found -> correct is None

        return -1
    if actual == 'NO' and not correct: #If correct is empty, than there is a ref file with no content
        return 1

    if actual in correct:
        return 1
    else:
        return 0

@validate_task.register("SE")
def validate_se(task,df, reference_path,extension):
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
    reference_result = get_reference_result_enumeration(reference_path, instance_name, task,ref_file_extension=extension)
    #print(f'{task=}\n{reference_result=}')

    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: compare_result_some_extension(
         single_extension_string_to_list(row['result']), reference_extensions),
                     axis=1)

@validate_task.register("CE")
def validate_ce(task, df: pd.DataFrame, ref_path: str,extension):
     return validate_decision(df,ref_path,extension)

@validate_task.register("EC")
def validate_ec(task, df: pd.DataFrame, ref_path: str, extension):
    pass

def compare_argument_list(result: list, actual: list ) -> str:
    if not actual:
        return 'no_reference'
    else:
        result.sort()
        actual.sort()


@validate_task.register("DC")
def validate_dc(task, df: pd.DataFrame, ref_path: str,extension):
    return validate_decision(df,ref_path,extension)

@validate_task.register("DS")
def validate_ds(task, df: pd.DataFrame, ref_path: str,extension):
    return validate_decision(df,ref_path,extension)

def validate_decision(df, reference_path,extension):
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
                                                     instance_name, task, extension=extension)
    print(reference_result.rstrip("\n"))
    return df.apply(lambda row: compare_results_enumeration(
        multiple_extensions_string_to_list(row['result']), reference_result),
                    axis=1)


def validate_instance(df, references ,ref_file_extension):
    """[summary]

    Args:
        df ([type]): [description]
        references ([type]): [description]

    Returns:
        [type]: [description]
    """
    # reference_path = references[(df['benchmark_id'].iloc[0])]
    # if not os.path.exists(reference_path):
    #     raise click.BadParameter("Reference path not found!")
    task = str(df['task'].iloc[0]).split("-")[0]

    df['validation_result'] = validate_task(task, df, references, ref_file_extension)

    # if 'EE' in str(df['task'].iloc[0]):
    #     df['correct'] = validate_ee(df, reference_path)
    # elif 'SE' in str(df['task'].iloc[0]):
    #     df['correct'] = validate_se(df, reference_path)
    # elif 'DC' in str(df['task'].iloc[0]):
    #     df['correct'] = validate_decision(df, reference_path)
    # elif 'DS' in str(df['task'].iloc[0]):
    #     df['correct'] = validate_decision(df, reference_path)

    return df

def validate(df, references, extension):
    """[summary]

    Args:
        df ([type]): [description]
        references ([type]): [description]

    Returns:
        [type]: [description]
    """
    val = (df
    .groupby(['benchmark_id','instance','task'])
    .apply(lambda _df: validate_instance(_df,references,extension))
    )

    val['no_reference'] = np.where(val.validation_result.values == -1, True,
                                False)
    val['correct'] = np.where(val.validation_result.values == 1, True,
                                False)
    val['incorrect'] = np.where(val.validation_result.values == 0, True,
                                False)
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

def validate_intersection(a_solver, other_solver, task, intersection_solved_instances):
    pass

def _validate_pairwise(df:pd.DataFrame)-> pd.DataFrame:
    unique_solvers = sorted(list(df['solver_full_name'].unique()))
    other_solver = unique_solvers.copy()
    accordance_df = pd.DataFrame(columns=unique_solvers, index=unique_solvers).fillna(0)

    for s in unique_solvers:
        s_solved = set(df[df.solver_full_name == s].instance.values)
        accordance_df[s][s] = 100.0
        other_solver.remove(s)
        for other in other_solver:
            o_solved = set(df[df.solver_full_name == other].instance.values)
            intersection_solved = set.intersection(s_solved,o_solved)
            print(f'Solver: {s} Other: {other}\n{intersection_solved=}\n\n')
            for instance in intersection_solved:

                s_res_list = multiple_extensions_string_to_list("".join(df[(df.solver_full_name == s) & (df.instance == instance)].result.values))
                other_res_list = multiple_extensions_string_to_list("".join(df[(df.solver_full_name == other) & (df.instance == instance)].result.values))

                val_result = compare_results_enumeration(s_res_list,other_res_list)
                if val_result == 'correct':
                    accordance_df[s][other] =  accordance_df[s][other] + 1
                    accordance_df[other][s] =  accordance_df[other][s] + 1
            accordance_df[s][other] =  accordance_df[s][other] / len(intersection_solved) * 100.0
            accordance_df[other][s] =  accordance_df[other][s] / len(intersection_solved) * 100.0
    print(accordance_df)




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
                if 'DC' in row.task or 'DS' in row.task or "CE" in row.task:
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
    if correct == 'correct':
        result_obj.correct_solved = True
    elif correct == 'incorrect':
        result_obj.incorrect_solved = True
    elif correct == "no_reference":
        result_obj.no_reference = True

    result_obj.correct = correct
    result_obj.validated = validated

def _get_reference_by_extensions(extensions: tuple, unique_tasks: list, reference)-> dict:
    ref_dict = dict()
    for f_extension in extensions:
        instances_per_task = _init_task_dict(unique_tasks)
        instances_with_f_extension = (list((chain.from_iterable(glob(os.path.join(x[0], f'*.{f_extension}')) for x in os.walk(reference)))))
        for instance in instances_with_f_extension:
            match = next((x for x in unique_tasks if x in instance), False)
            print(match)
            if match:
                instances_per_task[match].append(instance)
        ref_dict[f_extension] = instances_per_task
    return ref_dict

def _init_task_dict(tasks:list) -> dict:
    task_dict = {}
    for t in tasks:
        task_dict[t] = []
    return task_dict
def _get_reference_all_extensions(unique_tasks: list, reference)-> dict:
    instances_with_f_extension = (list((chain.from_iterable(glob(os.path.join(x[0], f'*.*')) for x in os.walk(reference)))))
    different_extensions = []
    ref_dict = dict()
    for instance in instances_with_f_extension:
        file_extension = pathlib.Path(instance).suffix.split(".")[1]
        if not file_extension in different_extensions:
            different_extensions.append(file_extension)
            ref_dict[file_extension] = _init_task_dict(unique_tasks)
        match = next((x for x in unique_tasks if x in instance),False)
        #print(f'{instance=} {match=}')
        if match:
            #print(f'ref_dict[{file_extension=}][{match=}].append({instance})')
            ref_dict[file_extension][match].append(instance)
    return ref_dict

def get_reference(extensions: tuple, unique_tasks: list, reference)-> dict:
    if not extensions:
        return _get_reference_all_extensions(unique_tasks,reference)
    else:
        return _get_reference_by_extensions(extensions,unique_tasks,reference)

def _print_summary(df):
    unique_solver = ','.join(df.solver.unique())
    unique_benchmark = ','.join(df.benchmark_name.unique())
    unique_tasks = ','.join(df.task.unique())

    sum_correct = df.correct.sum()
    sum_incorrect = df.incorrect.sum()
    sum_no_reference = df.no_reference.sum()
    sum_total = df.total.sum()

    percentage_validated_total = (sum_correct + sum_incorrect) / sum_total * 100.0

    print(f'Benchmark:{unique_benchmark}\nTask: {unique_tasks}\nSolver: {unique_solver}')
    print(f'#Correct: {sum_correct}\n#Incorrect: {sum_incorrect}\n#No_Reference: {sum_no_reference}\n#Total: {sum_total}\nValidated(%): {percentage_validated_total}\n')




def _print_results_per_task(df):
    task = df.task.iloc[0]
    print(f'+++++ Task {task} +++++')
    _print_summary(df)
    utils.print_df(df,headers=['solver','benchmark_name','correct','incorrect','no_reference','total','percentage_validated'])




def print_validate_with_reference_results(df):
    tag = df.tag.iloc[0]
    print(f'***** Validation results experiment {tag} *****')
    _print_summary(df)
    df.groupby(['benchmark_id','task']).apply(lambda df_: _print_results_per_task(df_))


