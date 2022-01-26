"""Validation module"""
import functools
import itertools
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

from sqlalchemy.sql.expression import column
from src.database_models.Benchmark import Benchmark


from src.utils.utils import dispatch_on_value
from src.reporting import pretty_latex_table
import src.utils.utils as utils

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
    .apply(lambda _df: _validate_instance(_df,references,extension))
    )
    val['no_reference'] = np.where(val.validation_result.values == -1, True,
                                False)
    val['correct'] = np.where(val.validation_result.values == 1, True,
                                False)
    val['incorrect'] = np.where(val.validation_result.values == 0, True,
                                False)
    return val

def _validate_instance(df, references ,ref_file_extension):
    """[summary]

    Args:
        df ([type]): [description]
        references ([type]): [description]

    Returns:
        [type]: [description]
    """

    task = str(df['task'].iloc[0]).split("-")[0]
    val_result = validate_task(task, df, references, ref_file_extension)
    df['validation_result'] = val_result
    return df


@dispatch_on_value
def validate_task(task, df, references ,file_extension):
    print(f"Task {task} not supported for validation.")


@validate_task.register("DC")
def validate_dc(task, df: pd.DataFrame, references: dict,file_extension):
    instance_name = df['instance'].iloc[0]
    task = df['task'].iloc[0]
    reference_result = _get_reference_result(
        references, instance_name, task,ref_file_extension=file_extension)
    return df.apply(lambda row: _compare_results_decision(row['result'],reference_result),axis=1)

@validate_task.register("DS")
def validate_ds(task, df: pd.DataFrame, references: dict,file_extension):
    return validate_dc(task, df, references,file_extension)

@validate_task.register("EC")
def validate_ec(task, df: pd.DataFrame, references: dict,file_extension):
    return validate_dc(task, df, references,file_extension)

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
    reference_result = _get_reference_result(
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
    reference_result = _get_reference_result(references, instance_name, task,ref_file_extension=extension)

    reference_extensions = multiple_extensions_string_to_list(reference_result)
    return df.apply(lambda row: _compare_result_some_extension(
         single_extension_string_to_list(row['result']), reference_extensions),
                     axis=1)

def _compare_result_some_extension(actual,correct):
    if correct is None: # No reference file found -> correct is None

        return -1
    if actual == 'NO' and not correct: #If correct is empty, than there is a ref file with no content
        return 1

    if actual in correct:
        return 1
    else:
        return 0

#  -----------------------Pairwise comparision-----------------------
@dispatch_on_value
def pairwise_comparision(task,current_solver_result, other_solver_result):
    pass
    #print(f"Task {task} not supported for pairwise comparision.")

@pairwise_comparision.register("EE")
def _ee_pairwise_comparision(task,current_solver_result: str, other_solver_result: str):
    current_solver_res_list = multiple_extensions_string_to_list(current_solver_result)
    other_res_list = multiple_extensions_string_to_list(other_solver_result)

    if functools.reduce(lambda i, j: i and j,
                        map(lambda m, k: m == k, current_solver_res_list, other_res_list), True):
        return 1
    else:
        return 0

def _acceptance_comaprision(current_solver_result: str, other_solver_result: str):
    current_solver_result = re.sub(r"[\n\t\s]*", "", current_solver_result)
    other_solver_result = re.sub(r"[\n\t\s]*", "", other_solver_result)
    if current_solver_result == other_solver_result:
        return 1
    else:
        return 0


@pairwise_comparision.register('DC')
def _dc_pairwise_comparision(task,current_solver_result: str, other_solver_result: str) -> bool:
    return _acceptance_comaprision(current_solver_result,other_solver_result)

@pairwise_comparision.register('DS')
def _ds_pairwise_comparision(task,current_solver_result: str, other_solver_result: str) -> bool:
   return _acceptance_comaprision(current_solver_result,other_solver_result)

@pairwise_comparision.register('CE')
def _ce_pairwise_comparision(task,current_solver_result: str, other_solver_result: str) -> bool:
   return _acceptance_comaprision(current_solver_result,other_solver_result)

def _validate_pairwise(df:pd.DataFrame)-> pd.DataFrame:
    benchmark_name = df.benchmark_name.iloc[0]
    unique_solvers = sorted(list(df['solver_full_name'].unique()))
    other_solver = unique_solvers.copy()
    accordance_df = pd.DataFrame(columns=unique_solvers, index=unique_solvers).fillna(0)
    num_intersection_solved_df = pd.DataFrame(columns=unique_solvers, index=unique_solvers).fillna(0)
    task = df.task.iloc[0].split("-")[0]

    for s in unique_solvers:
        s_solved = set(df[df.solver_full_name == s].instance.values)
        accordance_df[s][s] = 100.0
        num_intersection_solved_df[s][s] = len(s_solved)
        other_solver.remove(s)
        for other in other_solver:
            o_solved = set(df[df.solver_full_name == other].instance.values)
            intersection_solved = set.intersection(s_solved,o_solved)

            current_solver_df = df[(df.solver_full_name == s) & (df.instance.isin(intersection_solved))][['instance','result']].sort_values('instance')
            other_solver_df = df[(df.solver_full_name == s) & (df.instance.isin(intersection_solved))][['instance','result']].rename(columns={'result':'result_other'}).sort_values('instance')
            combined = pd.concat([other_solver_df,current_solver_df],axis=1)

            validation_results = combined.apply(lambda row: pairwise_comparision(task,row.result,row.result_other), axis=1)
            num_accordance = validation_results.sum()

            accordance_df[s][other] =  num_accordance / len(intersection_solved) * 100.0
            accordance_df[other][s] =  num_accordance / len(intersection_solved) * 100.0

            num_intersection_solved_df[s][other] =len(intersection_solved)
            num_intersection_solved_df[other][s] =len(intersection_solved)
    return pd.Series(data = {'benchmark_name': benchmark_name,'accordance': accordance_df.to_dict(),'intersection_solved': num_intersection_solved_df.to_dict()})

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


def _compare_results_decision(actual, correct):
    """[summary]

    Args:
        actual ([type]): [description]
        correct ([type]): [description]

    Returns:
        [type]: [description]
    """



    if not correct:
        return - 1
    else:
        actual = re.sub(r"[\n\t\s]*", "", actual)
        correct = re.sub(r"[\n\t\s]*", "", correct)
        if actual == correct:
            return 1
        else:
            return 0


def _get_reference_result(references: dict, instance_name: str, task: str,ref_file_extension=None):
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

    for f_extension in ref_file_extension:

        if task in references[f_extension].keys():

            references_file_extensions_task = references[f_extension][task]
            for reference_result in references_file_extensions_task:
                if all(substring in reference_result for substring in [instance_name,task,f_extension]):
                    with open(reference_result,'r',encoding='utf-8') as ref_file:
                        ref_result_str = ref_file.read()
                    return ref_result_str
    return None

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


def update_result_object(result_obj, correct,incorrect,no_reference):
    """[summary]

    Args:
        result_obj ([type]): [description]
        correct ([type]): [description]
        validated ([type]): [description]
    """

    result_obj.correct_solved = correct
    result_obj.incorrect_solved = incorrect
    result_obj.no_reference = no_reference
    result_obj.validated = not(no_reference)

def _get_reference_by_extensions(extensions: tuple, unique_tasks: list, reference)-> dict:
    ref_dict = dict()
    for f_extension in extensions:
        instances_per_task = _init_task_dict(unique_tasks)
        instances_with_f_extension = (list((chain.from_iterable(glob(os.path.join(x[0], f'*.{f_extension}')) for x in os.walk(reference)))))
        for instance in instances_with_f_extension:
            match = next((x for x in unique_tasks if x in instance), False)
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


# ----------------------------- Printing -----------------------------
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

def _print_results_per_task_reference_validation(df):
    task = df.task.iloc[0]
    print(f'+++++ Task {task} +++++')
    _print_summary(df)
    utils.print_df(df,headers=['solver','benchmark_name','correct','incorrect','no_reference','total','percentage_validated'])



def print_validate_with_reference_results(df):
    tag = df.tag.iloc[0]
    print(f'***** Validation results experiment {tag} *****')
    _print_summary(df)
    df.groupby(['benchmark_id','task']).apply(lambda df_: _print_results_per_task_reference_validation(df_))

def _print_results_per_task_pairwise_validation(df):
    task = df.task.iloc[0]
    print(f'+++++ Task {task} +++++\n')
    intersection_solved = pd.DataFrame.from_dict(df.intersection_solved.iloc[0])
    accordance = pd.DataFrame.from_dict(df.accordance.iloc[0])
    print(f'Number of instances which were solved by both solvers:\n\n {intersection_solved}\n')
    print(f'Accordance in percentage:\n\n{accordance}\n')

def print_validate_pairwise(df):
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    print(f'***** Pairwise comparision results for experiment {tag} on benchmark {benchmark} *****\n')
    df.groupby(['benchmark_id','task']).apply(lambda _df: _print_results_per_task_pairwise_validation(_df))

def _print_incorrect_and_missing_instances_per_solver(df):
    """Print incorrect and not validated instances per solver""

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """

    current_solver = df.solver_full_name.iloc[0]
    no_ref_instances = ', '.join(df[df.no_reference]['instance'].to_list())
    incorrect_instances = ', '.join(df[df.incorrect]['instance'].to_list())
    if not no_ref_instances:
        no_ref_instances ="None"
    if not incorrect_instances:
        incorrect_instances = "None"
    print(f'##### {current_solver} #####\nIncorrect instances:\n{incorrect_instances}\n\nInstances with no reference:\n{no_ref_instances}\n')


def _print_not_validated_per_task(df):
    task = df.task.iloc[0]
    print(f'+++++ Task {task} +++++\n')
    df.groupby('solver_id').apply(lambda df: _print_incorrect_and_missing_instances_per_solver(df))


def print_not_validated(df):
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    print(f'\n***** Not validated and incorrect instances for {tag} on benchmark {benchmark} *****\n')

    df.groupby(['benchmark_id','task']).apply(lambda _df: _print_not_validated_per_task(_df))



# ----------------------------- Exporting and plotting -----------------------------
@dispatch_on_value
def export(export_format: str,df:pd.DataFrame, save_to, file_name_suffix=""):
    print(f"Export format {export_format} not supported.")

@export.register('latex')
def _export_latex(export_format: str,df:pd.DataFrame, save_to,file_name_suffix=""):

    column_mapping = {'correct':'#Correct','incorrect': '#Incorrect','no_reference':'#No_Reference','total':'#Total','percentage_validated':'Validated(%)'}
    max_bold = set(['correct','total','percentage_validated'])
    min_bold = set(['incorrect','no_reference'])
    columns_str = ['correct','incorrect','no_reference',]
    #caption = f'{calculated_str} values for task {df.task.iloc[0]} on {df.benchmark.iloc[0]} benchmark. Best results in bold.'
    caption = f'Total number of correct solved, incorrect solved, without reference and checked instances for {df.task.iloc[0]} task on {df.benchmark_name.iloc[0]} benchmark.'
    label = f'{df.tag.iloc[0]}_tbl'
    filename = f'{df.task.iloc[0]}_{df.benchmark_name.iloc[0]}_{df.tag.iloc[0]}{file_name_suffix}.tex'
    return pretty_latex_table.generate_table(df[['solver']  + ['correct','incorrect','no_reference','total','percentage_validated']].round(2),
                                      save_to,
                                      max_bold=max_bold,
                                      min_bold=min_bold,
                                      caption=caption,
                                      label=label,
                                      filename=filename,
                                      column_name_map=column_mapping
                                      )
@export.register('csv')
def _export_csv(export_format, df, save_to, file_name_suffix=""):
    filename = f'{df.task.iloc[0]}_{df.benchmark_name.iloc[0]}_{df.tag.iloc[0]}{file_name_suffix}.csv'
    save_path = os.path.join(save_to,filename)

    with open(save_path,'w') as csv_file:
        csv_file.write(df.to_csv(index=False))
    return save_path
@export.register('json')
def _export_json(export_format, df, save_to, file_name_suffix=""):
    filename = f'{df.task.iloc[0]}_{df.benchmark_name.iloc[0]}_{df.tag.iloc[0]}{file_name_suffix}.json'
    save_path = os.path.join(save_to,filename)

    with open(save_path,'w', encoding='utf-8') as json_file:
        df.to_json(json_file)
    return save_path

@dispatch_on_value
def plot(kind: str,df:pd.DataFrame, save_to):
    print(f"Plot kind {kind} not supported.")
@plot.register('pie')
def _pie_chart(kind,df: pd.DataFrame, save_to,title=None):
    print(df)
    exit()
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
@plot.register('count')
def _count_plot(kind, df: pd.DataFrame,save_full_path,title=None,grid=False):
    task = df['task'].iloc[0]
    print(f'Creating {kind} plot for task {task}...',end='')
    validation_result_interpretation = {1:'Correct', 0:'Incorrect',-1: 'No Reference'}
    save_full_path = f'{os.path.join(save_full_path,create_file_name(df))}_count.png'
    df = df.rename(columns={'solver_full_name': 'Solver', 'validation_result': 'Status'})
    df['Status'] = df.Status.map(validation_result_interpretation)

    if grid:
        grid_plot = sns.catplot(x="Solver", hue="Status", col="task",
                data=df, kind="count",
                height=4, aspect=.7)

        grid_plot.set_xticklabels(rotation=40, ha="right")
        figure = grid_plot.fig
        figure.savefig(f"{save_full_path}_grid.png",
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

        else:
            title = df['benchmark_name'].iloc[0] +" " + task
            ax.set_title(title)
        figure = ax.get_figure()
        figure.savefig(save_full_path,
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    print("finished.")
    return save_full_path

def plot_accordance_heatmap(df, save_to,type, color_map='Blues', set_over='white',cbar_kws={}):
    """[summary]

    Args:
        df ([type]): [description]
        save_to ([type]): [description]
        color_map (str, optional): [description]. Defaults to 'Blues'.
        set_over (str, optional): [description]. Defaults to 'white'.
    """
    print(f'Creating {type} heatmap...',end='')
    file_name = create_file_name(df)
    df = df[type].iloc[0]
    if isinstance(df,dict):
        df = pd.DataFrame.from_dict(df)

    heatmap_args = {
        'linewidths': 0.25,
        'linecolor': '0.5',
        'clip_on': False,
        'square': True
    }
    cmap = sns.mpl.cm.get_cmap(color_map).copy()
    cmap.set_over(set_over)
    if type == 'accordance':
        ax = sns.heatmap(df,vmin=0,vmax=100,
                        cmap=cmap,
                        cbar_kws=cbar_kws,
                        **heatmap_args,
                        annot=True,
                        fmt='.2f')
    else:
         ax = sns.heatmap(df,
                        cmap=cmap,
                        cbar_kws=cbar_kws,
                        **heatmap_args,
                        annot=True,
                        fmt='.2f')

    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       rotation_mode='anchor',
                       ha='right')
    ax.figure.axes[-1].yaxis.label.set_size(15)
    figure = ax.get_figure()
    save_path_full = f'{os.path.join(save_to,file_name)}_{type}.png'
    figure.savefig(save_path_full,
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    print('finished.')
    return save_path_full

@dispatch_on_value
def export_pairwise(export_format: str,df:pd.DataFrame, save_to,file_name_suffix=""):
    print(f"Export format {export_format} not supported.")

@export_pairwise.register('json')
def _export_json_pairwise(export_format: str,df:pd.DataFrame, save_to,file_name_suffix=""):
    return _export_json(export_format,df,save_to,file_name_suffix=file_name_suffix)

@export_pairwise.register('csv')
def _export_csv_pairwise(export_format: str,df:pd.DataFrame, save_to,file_name_suffix=""):
    return _export_csv(export_format,df,save_to,file_name_suffix=file_name_suffix)

@dispatch_on_value
def plot_pairwise(kind: str,df:pd.DataFrame, save_to):
    print(f"Plot kind {kind} not supported.")

@plot_pairwise.register('heatmap')
def _plot_accordance_heatmap(kind: str,df:pd.DataFrame, save_to):
    cbar_kws={
                         'format': '%.0f%%',
                         'ticks': [0, 100]
             }

    saved_files_accordance = df.groupby(['tag','benchmark_id','task']).apply(lambda df: plot_accordance_heatmap(df,save_to,type='accordance',cbar_kws=cbar_kws))
    saved_files_intersection_solved = df.groupby(['tag','benchmark_id','task']).apply(lambda df: plot_accordance_heatmap(df,save_to,type='intersection_solved'))
    # saved_files_accordance = saved_files_accordance.to_frame().rename(columns={0:'saved_files'}).reset_index()['saved_files'].to_list()
    # saved_files_intersection_solved = saved_files_intersection_solved.to_frame().rename(columns={0:'saved_files'}).reset_index()['saved_files'].to_list()
    return pd.concat([saved_files_accordance,saved_files_intersection_solved])


