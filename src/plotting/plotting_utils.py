import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import shutil
from src import data
from src.utils import utils
from src.plotting import CactusPlot, DistributionPlot, ScatterPlot
from src.analysis.statistics import get_info_as_strings
import functools
from src.utils.utils import dispatch_on_value
#TODO: Json defaults for all plots, finish Scatter plot implementation,pie chart titles
def read_default_options(path: str) -> dict:
    with open(path,'rb') as fp:
        options =json.load(fp)
        options['def_path'] = path
    return options

def set_user_options(user_options, default_options):
    for key, value in user_options.params.items():
        if value is not None:
            default_options['settings'][key] = value

def create_archive_from_plots(save_to: str, name,saved_files_paths,compress):
    temp_path = os.path.join(save_to,'_temp_plots')
    if not os.path.exists(temp_path):
        os.mkdir(os.path.join(save_to,'_temp_plots'))
    for file_to_copy in saved_files_paths:
        shutil.copy(file_to_copy,temp_path)
    archive_save_to = os.path.join(save_to,f'{name}_plots')
    directory_save_path =utils.compress_directory(temp_path, archive_save_to,compress)
    shutil.rmtree(temp_path)
    return directory_save_path

@dispatch_on_value
def create_plots(kind, df, save_to, options, grouping):
    print(f"Plot type {kind} not supported.")


def create_file_name(df: pd.DataFrame,info,save_to: str,kind: str) -> str:
    task_symbol = info['task']
    benchmark_name = info['benchmark']
    curr_tag = info['tag']
    return os.path.join(save_to, "{}_{}_{}_{}".format(task_symbol, benchmark_name,curr_tag,kind))

def prepare_grid(df: pd.DataFrame)-> pd.DataFrame:
    return (df
            .groupby(['tag','task_id','benchmark_id','solver_id'],as_index=False)
            .apply(lambda g: g.assign(rank=g['runtime'].rank(method='dense',ascending=True)))
            .droplevel(0)
            .reset_index(drop=True)
          )

def get_overlapping_instances(df,solver_id):
    solver_df = df[df.solver_id.isin(solver_id)]
    unique_instances = solver_df.groupby('solver_id').apply(lambda solver: set(solver[~(solver.runtime.isnull())]['instance'].unique())).tolist()
    overlapping = list(set.intersection(*unique_instances))
    return solver_df[solver_df.instance.isin(overlapping)]

#Scatter plots
def prep_scatter_data(df):
    overlapping_instances_df = get_overlapping_instances(df)
    unique_solvers = list(overlapping_instances_df.solver.unique())
    runtimes_solver = dict()
    for s in unique_solvers:
        runtimes_solver[s]  = overlapping_instances_df[overlapping_instances_df.solver == s].sort_values('instance')['runtime'].values

    return pd.DataFrame.from_dict(runtimes_solver).fillna(0)



def _get_intersection_instances(df):
    set_list = []
    for name, group in df.groupby('solver_id'):
        set_list.append(set(group['instance'].unique()))
    if set_list:
        return list(set.intersection(*set_list))
    else:
        return []


def _prep_data_sactter_plot(df):
    df_solved  = df[(df.exit_with_error == False)]
    intersection_solved = _get_intersection_instances(df_solved)
    if intersection_solved:
        return df_solved[df_solved.instance.isin(intersection_solved)]
    else:
        return None

def _create_scatter_plot(df, save_to, options):
    solver_name_x = df.columns[0]
    solver_name_y = df.columns[1]

    solver_name_x_file_name = str(df.columns[0]).replace(".","-")
    solver_name_y_file_name = str(df.columns[1]).replace(".","-")
    save_file_name = f'{save_to}_{solver_name_x_file_name}_{solver_name_y_file_name}_scatter'
    data = [[solver_name_x,df[solver_name_x].values],[solver_name_y,df[solver_name_y].values]]
    options['settings']['save_to'] = save_file_name
    ScatterPlot.Scatter(options).create(data)
    return save_file_name

def _create_pairwise_scatter_plot(df, save_to, options):
    timeout = df.cut_off.max()
    options['settings']['timeout'] = timeout
    saved_files_list = []
    unique_solver_ids = list(df.solver_id.unique())
    remaining_solver_ids = unique_solver_ids.copy()
    info = get_info_as_strings(df)
    save_file_name = create_file_name(df,info,save_to,'scatter')
    if options['settings']['gen_title']:
        options['settings']['title'] = info['task'] + " " + info['benchmark']

    clean_df = _prep_data_sactter_plot(df)
    if clean_df is None:
        return []
    for current_solver_id in unique_solver_ids:
        current_solver_data = clean_df[clean_df.solver_id == current_solver_id]
        current_solver_data['instance'].sort_values()
        if current_solver_data.empty:
                continue
        current_solver_name = current_solver_data.solver_full_name.iloc[0]
        current_solver_data['runtime'] = np.where(current_solver_data.timed_out == True,timeout,current_solver_data.runtime)

        remaining_solver_ids.remove(current_solver_id)
        for remaining_solver_id in remaining_solver_ids:
            remaining_solver_data = clean_df[clean_df.solver_id == remaining_solver_id]
            if remaining_solver_data.empty:
                continue
            remaining_solver_data['instance'].sort_values()
            remaining_solver_name = remaining_solver_data.solver_full_name.iloc[0]

            remaining_solver_data['runtime'] = np.where(remaining_solver_data.timed_out == True,timeout,remaining_solver_data.runtime)

            data_dict = {current_solver_name:current_solver_data.runtime.values, remaining_solver_name: remaining_solver_data.runtime.values}
            scatter_data = pd.DataFrame(data_dict)
            saved_files_list.append(_create_scatter_plot(scatter_data,save_file_name,options))

    return saved_files_list





@create_plots.register("scatter")
def _scatter_plot(kind, df, save_to, options,grouping):
    print("Creating scatter plots...",end="")
    scatter_grouping = grouping.copy()
    if 'solver_id' in scatter_grouping:
        scatter_grouping.remove('solver_id')
    saved_files = df.groupby(scatter_grouping).apply(lambda _df: _create_pairwise_scatter_plot(_df,save_to,options))
    print("finished.")
    return saved_files



# Cactus plots
@create_plots.register("cactus")
def cactus_plot(kind, df, save_to, options,grouping):
    print("Creating cactus plots...",end="")
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    ranked = (df[only_solved_mask].groupby(grouping,as_index=False)
                    .apply(lambda df: df.assign(rank=df['runtime'].rank(method='dense',ascending=True))))
    plot_grouping = grouping.copy()
    plot_grouping.remove('solver_id')
    saved_files = ranked.groupby(plot_grouping).apply(lambda df: create_cactus_plot(df,save_to,options))
    print("finished.")
    return saved_files

def create_cactus_plot(df,save_to,options):
    info = get_info_as_strings(df)
    df['Solver'] = df['solver_full_name']
    options['settings']['title'] = info['task'] + " " + info['benchmark']

    save_file_name = create_file_name(df,info,save_to,'cactus')
    options['settings']['save_to'] = save_file_name
    CactusPlot.Cactus(options).create(df)
    return save_file_name
# Count plots
def prep_data_count_plot(df):
    conds = [((df.timed_out == False) & (df.exit_with_error == False)),df.timed_out == True,df.exit_with_error == True]
    choices = ['Solved','Timed out','Aborted']
    df['Status'] = np.select(conds,choices)
    return df

def create_count_plot(df, save_to, options):

    info = get_info_as_strings(df)
    save_file_name = create_file_name(df,info,save_to,'count')
    ax = sns.countplot(data=df,x='solver_full_name',hue='Status')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    benchmark_name = info['benchmark']
    task = info['task']
    ax.set_title(f'{benchmark_name} {task}')
    ax.set(xlabel='solver')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figure = ax.get_figure()
    figure.savefig(f"{save_file_name}.png",
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_file_name

@create_plots.register("count")
def count_plot(kind,df,save_to,options,grouping):
    print("Creating count plots...",end="")
    preped_data = df.groupby(grouping,as_index=False).apply(lambda df: prep_data_count_plot(df))
    plot_grouping = grouping.copy()
    plot_grouping.remove('solver_id')
    saved_files = preped_data.groupby(plot_grouping).apply(lambda df: create_count_plot(df,save_to,options))
    print("finished.")
    return saved_files

# Pie charts
def create_pie_chart(df: pd.DataFrame,save_to: str, options: dict):
    info = get_info_as_strings(df)
    if len(df.solver_id.unique()) > 1:
        name_ending = 'pie_summary'
    else:
        solver = df.solver_full_name.iloc[0]
        name_ending = f'pie_{solver}'
    save_file_name = create_file_name(df,info,save_to,name_ending)
    labels = ['solved','timed out','exit with error']

    timed_out = df['timed_out'].sum()
    exit_error = df['exit_with_error'].sum()
    solved = df.shape[0] -(timed_out - exit_error)
    data = [solved,timed_out,exit_error]

    if solved == 0:
       del data[0]
       del labels[0]
    if timed_out == 0:
       del data[len(data)-2]
       del labels[len(labels)-2]
    if exit_error == 0:
       del data[len(data)-1]
       del labels[len(labels) -1]


    colors = sns.color_palette(options['color_palette'])[0:4]
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    benchmark_name = info['benchmark']
    task = info['task']
    plt.title(f'{benchmark_name} {task}')
    plt.savefig(f"{save_file_name}.png",
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_file_name



@create_plots.register("pie")
def pie_chart(kind,df: pd.DataFrame, save_to: str, options: dict, grouping: list):
    print("Creating pie charts...",end="")
    with open(options['def_path'], 'r') as fp:
        pie_options = json.load(fp)['pie_options']
    df.groupby(grouping).apply(lambda df: create_pie_chart(df,save_to,pie_options))
    pie_grouping = grouping.copy()
    pie_grouping.remove('solver_id')
    saved_files = df.groupby(pie_grouping).apply(lambda df: create_pie_chart(df,save_to,pie_options))
    print("finished.")
    return saved_files

# Dist plots
@create_plots.register("dist")
def dist_plot(kind,df: pd.DataFrame,save_to: str,options: dict,grouping: list):
    print("Creating distribution plots...",end="")
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    dist_grouping = grouping.copy()
    dist_grouping.remove('solver_id')
    saved_files = df[only_solved_mask].groupby(dist_grouping).apply(lambda df: create_dist_plot(df,save_to,options))
    print("finished.")
    return saved_files

def create_dist_plot(df: pd.DataFrame,save_to: str,options: dict) -> str:
    info = get_info_as_strings(df)
    df['Solver'] = df['solver_full_name']
    options['settings']['title'] = info['task'] + " " + info['benchmark']
    save_file_name = create_file_name(df,info,save_to,'dist')
    options['settings']['save_to'] = save_file_name
    DistributionPlot.Distribution(options).create(df['runtime'])
    return save_file_name

@create_plots.register("box")
def box_plot(kind,df: pd.DataFrame,save_to: str,options: dict,grouping: list):
    print("Creating box plots...",end="")
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    box_grouping = grouping.copy()
    box_grouping.remove('solver_id')
    saved_files = df[only_solved_mask].groupby(box_grouping).apply(lambda df: create_box_plot(df,save_to,options))
    print("finished.")
    return saved_files

def create_box_plot(df: pd.DataFrame,save_to: str,options: dict):
    info = get_info_as_strings(df)
    df['Solver'] = df['solver_full_name']
    save_file_name = create_file_name(df,info,save_to,'box')
    ax=sns.boxplot(data=df,y='runtime', x='solver_full_name')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    benchmark_name = info['benchmark']
    task = info['task']
    ax.set_title(f'{benchmark_name} {task}')
    ax.set(xlabel='solver')
    figure = ax.get_figure()
    figure.savefig(f"{save_file_name}.png",
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_file_name



