import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from src.plotting import CactusPlot, DistributionPlot, ScatterPlot
from src.analysis.statistics import get_info_as_strings
import functools
from src.utils.utils import dispatch_on_value
#TODO: Json defaults for all plots, finish Scatter plot implementation,pie chart titles



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

def scatter_plot(df: pd.DataFrame, save_to: str, options: dict, grouping: list):
    pass

def create_scatter_plot(df: pd.DataFrame, save_to: str, options: dict)->str:
    info = get_info_as_strings(df)
    df['Solver'] = df['solver_full_name']
    options['title'] = info['task'] + " " + info['benchmark']
    save_file_name = create_file_name(df,info,save_to,'cactus')
    options['save_to'] = save_file_name
    ScatterPlot.Scatter(options).create(df)

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
    options['title'] = info['task'] + " " + info['benchmark']

    save_file_name = create_file_name(df,info,save_to,'cactus')
    options['save_to'] = save_file_name
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

@create_plots.register("count")
def count_plot(kind,df,save_to,options,grouping):
    print("Creating count plots...",end="")
    preped_data = df.groupby(grouping,as_index=False).apply(lambda df: prep_data_count_plot(df))
    plot_grouping = grouping.copy()
    plot_grouping.remove('solver_id')
    preped_data.groupby(plot_grouping).apply(lambda df: create_count_plot(df,save_to,options))
    print("finished.")

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



@create_plots.register("pie")
def pie_chart(kind,df: pd.DataFrame, save_to: str, options: dict, grouping: list):
    print("Creating pie charts...",end="")
    with open(options['def_path'], 'r') as fp:
        pie_options = json.load(fp)['pie_options']
    df.groupby(grouping).apply(lambda df: create_pie_chart(df,save_to,pie_options))
    pie_grouping = grouping.copy()
    pie_grouping.remove('solver_id')
    df.groupby(pie_grouping).apply(lambda df: create_pie_chart(df,save_to,pie_options))
    print("finished.")

# Dist plots
@create_plots.register("dist")
def dist_plot(kind,df: pd.DataFrame,save_to: str,options: dict,grouping: list):
    print("Creating distribution plots...",end="")
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    dist_grouping = grouping.copy()
    dist_grouping.remove('solver_id')
    df[only_solved_mask].groupby(dist_grouping).apply(lambda df: create_dist_plot(df,save_to,options))
    print("finished.")

def create_dist_plot(df: pd.DataFrame,save_to: str,options: dict):
    info = get_info_as_strings(df)
    df['Solver'] = df['solver_full_name']
    options['title'] = info['task'] + " " + info['benchmark']
    save_file_name = create_file_name(df,info,save_to,'dist')
    options['save_to'] = save_file_name
    DistributionPlot.Distribution(options).create(df['runtime'])

@create_plots.register("box")
def box_plot(kind,df: pd.DataFrame,save_to: str,options: dict,grouping: list):
    print("Creating box plots...",end="")
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    box_grouping = grouping.copy()
    box_grouping.remove('solver_id')
    df[only_solved_mask].groupby(box_grouping).apply(lambda df: create_box_plot(df,save_to,options))
    print("finished.")

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



