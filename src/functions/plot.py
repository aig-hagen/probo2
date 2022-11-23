import src.functions.register as register
from src.utils import config_handler
import pandas as pd
import numpy as np
from src.plotting import CactusPlot, ScatterPlot
import os
from src.utils import definitions
from src.plotting import plotting_utils as pl_util

import seaborn as sns
import matplotlib.pyplot as plt

def create_plots(result_df: pd.DataFrame, cfg: config_handler.Config):
    saved_files = []
    print("========== PLOTTING ==========")
    if cfg.plot =='all' or 'all' in cfg.plot:
        cfg.plot = register.plot_dict.keys()
    saved_plots = []
    default_plt_options = pl_util.read_default_options(str(definitions.PLOT_JSON_DEFAULTS))
    for plt in cfg.plot:
        saved = register.plot_dict[plt](result_df,cfg,default_plt_options)
        saved_plots.append(saved)
    saved_files = pd.concat(saved_plots).to_frame().rename(columns={0:'saved_files'})
    print('')
    return  saved_files.saved_files.to_list()


def _get_avg_reps(df: pd.DataFrame):
    result_dict = {}
    for col in df.columns:
        result_dict[col] = df[col].iloc[0]
    result_dict['runtime'] = df.runtime.mean()
    return pd.Series(result_dict)

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

def get_info_as_strings(df: pd.DataFrame) -> dict:
    tags = ",".join(df.tag.unique())
    solvers = ",".join(df.solver_name.unique())
    tasks = ",".join(df.task.unique())
    benchmarks = ",".join(df.benchmark_name.unique())
    return {'tag': tags,'solver': solvers, 'task': tasks,'benchmark': benchmarks}


def _create_file_name(df: pd.DataFrame,info,save_to: str,kind: str) -> str:
    task_symbol = info['task']
    benchmark_name = info['benchmark']
    curr_tag = info['tag']
    return os.path.join(save_to, "{}_{}_{}_{}".format(task_symbol, benchmark_name,curr_tag,kind))

def _create_scatter_plot(df, save_to, options):
    solver_name_x = df.columns[0]
    solver_name_y = df.columns[1]

    solver_name_x_file_name = str(df.columns[0]).replace(".","-")
    solver_name_y_file_name = str(df.columns[1]).replace(".","-")
    save_file_name = f'{save_to}_{solver_name_x_file_name}_{solver_name_y_file_name}_scatter'
    data = [[solver_name_x,df[solver_name_x].values],[solver_name_y,df[solver_name_y].values]]

    ScatterPlot.Scatter(options,save_file_name).create(data)
    return save_file_name

def _create_pairwise_scatter_plot(df, config: config_handler.Config, options):
    timeout = config.timeout
    options['settings']['timeout'] =  timeout
    saved_files_list = []
    unique_solver_ids = list(df.solver_id.unique())
    remaining_solver_ids = unique_solver_ids.copy()
    info = get_info_as_strings(df)
    save_file_name = _create_file_name(df,info,options['settings']['save_to'],'scatter')
    if options['settings']['gen_title']:
        options['settings']['title'] = info['task'] + " " + info['benchmark']

    clean_df = _prep_data_sactter_plot(df)
    clean_df['solver_full_name'] = clean_df['solver_name'] +'_' + clean_df['solver_version']
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

def scatter_plot(df: pd.DataFrame,config: config_handler.Config, plot_options):
    print("Creating scatter plots...",end="")
    plot_directory = os.path.join(config.save_to,'plots')
    os.makedirs(plot_directory,exist_ok=True)
    plot_options['settings']['save_to'] = plot_directory
    grouping = config.grouping
    scatter_grouping = grouping.copy()
    if 'solver_id' in scatter_grouping:
        scatter_grouping.remove('solver_id')
    if plot_options['settings']['include_timeouts']:
        mask = (df.exit_with_error == False)
    else:
        mask = (df.timed_out == False) & (df.exit_with_error == False)
    rep_avg_df = df[mask].groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))

    saved_files = rep_avg_df.groupby(scatter_grouping).apply(lambda _df: _create_pairwise_scatter_plot(_df,config,plot_options))
    print("done!")
    return saved_files



def cactus_plot(df: pd.DataFrame,config: config_handler.Config, plot_options):
    print("Creating cactus plots...",end="")

    plot_directory = os.path.join(config.save_to,'plots')
    os.makedirs(plot_directory,exist_ok=True)
    plot_options['settings']['save_to'] = plot_directory
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    grouping = config.grouping
    rep_avg_df = df[only_solved_mask].groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    ranked = (rep_avg_df.groupby(grouping,as_index=False)
                     .apply(lambda df: df.assign(rank=df['runtime'].rank(method='dense',ascending=True))))
    plot_grouping = grouping.copy()
    if 'solver_id' in plot_grouping:
        plot_grouping.remove('solver_id')
    saved_files = ranked.groupby(plot_grouping).apply(lambda df: _create_cactus_plot(df,config,plot_options))
    print("done!")
    return saved_files


def _create_cactus_plot(df,config: config_handler.Config, plot_options):

#     info = get_info_as_strings(df)
    df['Solver'] = df['solver_name'] +'_' + df['solver_version']

    #save_file_name = create_file_name(df,info,save_to,'cactus')
    task = df.task.iloc[0]
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    plot_options['settings']['title'] = f"{task} {benchmark}"
    save_file_name = os.path.join( plot_options['settings']['save_to'],f'{tag}_{task}_{benchmark}_cactus.png')

    return CactusPlot.Cactus(plot_options,save_file_name).create(df)


def _create_count_plot(df,config: config_handler.Config, plot_options):
    task = df.task.iloc[0]
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    plot_options['settings']['title'] = f"{task} {benchmark}"
    save_file_name = os.path.join(plot_options['settings']['save_to'],f'{tag}_{task}_{benchmark}_count.png')
    df['Solver'] = df['solver_name'] +'_' + df['solver_version']
    #ax = sns.countplot(data=df,x='Solver',hue='Status')
    grid_axes = sns.catplot(data=df,x='Solver',hue='Status',kind='count')
    for ax in grid_axes.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set(xlabel='solver')
    #plt.xticks(rotation=40)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figure = grid_axes.fig
    figure.suptitle( plot_options['settings']['title'])
    figure.savefig(save_file_name,
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_file_name

def _prep_data_count_plot(df):
    conds = [((df.timed_out == False) & (df.exit_with_error == False)),df.timed_out == True,df.exit_with_error == True]
    choices = ['Solved','Timed out','Aborted']
    df['Status'] = np.select(conds,choices)
    return df

def count_plot(df,config: config_handler.Config, plot_options):
    print("Creating count plots...",end="")
    plot_directory = os.path.join(config.save_to,'plots')
    os.makedirs(plot_directory,exist_ok=True)
    plot_options['settings']['save_to'] = plot_directory
    grouping = config.grouping
    preped_data = df.groupby(grouping,as_index=False).apply(lambda df: _prep_data_count_plot(df))
    plot_grouping = grouping.copy()
    plot_grouping.remove('solver_id')
    saved_files = preped_data.groupby(plot_grouping).apply(lambda df: _create_count_plot(df,config,plot_options))
    print("done!")
    return saved_files

register.plot_functions_register('cactus',cactus_plot)
register.plot_functions_register('count',count_plot)
register.plot_functions_register('scatter',scatter_plot)