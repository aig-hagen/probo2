import src.functions.register as register
from src.utils import config_handler
import pandas as pd
import numpy as np
from src.plotting import CactusPlot
import os

import seaborn as sns
import matplotlib.pyplot as plt
def _get_avg_reps(df: pd.DataFrame):
    result_dict = {}
    for col in df.columns:
        result_dict[col] = df[col].iloc[0]
    result_dict['runtime'] = df.runtime.mean()
    return pd.Series(result_dict)

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
    plot_grouping.remove('solver_id')
    saved_files = ranked.groupby(plot_grouping).apply(lambda df: _create_cactus_plot(df,config,plot_options))
    print("finished.")
    return saved_files


def _create_cactus_plot(df,config: config_handler.Config, plot_options):

#     info = get_info_as_strings(df)
    df['Solver'] = df['solver_name'] +'_' + df['solver_version']

    #save_file_name = create_file_name(df,info,save_to,'cactus')
    task = df.task.iloc[0]
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    plot_options['settings']['title'] = task
    save_file_name = os.path.join( plot_options['settings']['save_to'],f'{tag}_{task}_{benchmark}_cactus.png')

    return CactusPlot.Cactus(plot_options,save_file_name).create(df)


def _create_count_plot(df,config: config_handler.Config, plot_options):
    task = df.task.iloc[0]
    tag = df.tag.iloc[0]
    benchmark = df.benchmark_name.iloc[0]
    plot_options['settings']['title'] = task
    save_file_name = os.path.join(plot_options['settings']['save_to'],f'{tag}_{task}_{benchmark}_count.png')
    df['Solver'] = df['solver_name'] +'_' + df['solver_version']
    #ax = sns.countplot(data=df,x='Solver',hue='Status')
    grid_axes = sns.catplot(data=df,x='Solver',hue='Status',col='repetition',kind='count',col_wrap=3)
    for ax in grid_axes.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set(xlabel='solver')
    #plt.xticks(rotation=40)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figure = grid_axes.fig
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
    print("finished.")
    return saved_files

register.plot_functions_register('cactus',cactus_plot)
register.plot_functions_register('count',count_plot)