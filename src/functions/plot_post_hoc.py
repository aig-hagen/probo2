
from distutils.command.config import config
import src.functions.register as register
from src.handler import config_handler
import pandas as pd
import os
import seaborn as sns
from src.handler.benchmark_handler import load_benchmark_by_identifier
import matplotlib.pyplot as plt

def create_plots(result_df: pd.DataFrame, cfg: config_handler.Config,test):
    saved_files = []
    print("========== PLOTTING ==========")
    saved_plots = []
    for plt in cfg.significance['plot']:
        saved = register.plot_post_hoc_functions_dict[plt](result_df,cfg,test)
        saved_plots.append(saved)
    saved_files = pd.concat(saved_plots).to_frame().rename(columns={0:'saved_files'})
    print("")
    return  saved_files.saved_files.to_list()

def _heat_map(df:pd.DataFrame, cfg:config_handler.Config,plot_directory,test):
    tag = df.tag.iloc[0]
    task = df.task.iloc[0]
    rep = df.repetition.iloc[0]
    benchmark = load_benchmark_by_identifier([int(df.benchmark_id.iloc[0])])[0]['name']

    heatmap_args = {
        'linewidths': 0.25,
        'linecolor': '0.5',
        'clip_on': False,
        'square': True
    }
    color_map='Blues_r'
    set_over='white'
    cmap = sns.mpl.cm.get_cmap(color_map).copy()
    cmap.set_over(set_over)
    ax = sns.heatmap(df.result.iloc[0],
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
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=45,
                       rotation_mode='anchor',
                       ha='right')
    ax.figure.axes[-1].yaxis.label.set_size(15)
    figure = ax.get_figure()
    file_name = f"{tag}_{task}_{benchmark}_{test}_{rep}_post_hoc"
    save_path_full = f'{os.path.join(plot_directory,file_name)}.png'
    figure.savefig(save_path_full,
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_path_full

def heat_map(df: pd.DataFrame, cfg:config_handler.Config,test):
    plot_directory = os.path.join(cfg.save_to,'post_hoc_plots')
    os.makedirs(plot_directory,exist_ok=True)
    print('Creating heatmap...',end='')
    saved_files = df.groupby(['repetition','tag','task','benchmark_id']).apply(lambda _df: _heat_map(df,cfg,plot_directory,test))
    print("done!")
    return saved_files

register.plot_post_hoc_functions_register('heatmap',heat_map)