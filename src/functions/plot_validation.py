import src.functions.register as register
import pandas as pd
import seaborn as sns
import os
import math
from src.handler import config_handler

import matplotlib.pyplot as plt

def create_plots(result_df: pd.DataFrame, cfg: config_handler.Config):
    saved_files = []
    print("========== PLOTTING ==========")
    saved_plots = []
    for plt in cfg.validation['plot']:
        saved = register.plot_validation_functions_dict[plt](result_df,cfg)
        saved_plots.append(saved)
    saved_files = pd.concat(saved_plots).to_frame().rename(columns={0:'saved_files'})
    print('')
    return  saved_files.saved_files.to_list()

def _plot_accordance_heatmap(df: pd.DataFrame, cfg: config_handler.Config, plot_directory):
    color_map='Blues'
    set_over='white'
    cbar_kws={}

    tag = df.tag.iloc[0]
    task = df.task.iloc[0]
    benchmark = df.benchmark_name.iloc[0]

    heatmap_args = {
        'linewidths': 0.25,
        'linecolor': '0.5',
        'clip_on': False,
        'square': True
    }
    cmap = sns.mpl.cm.get_cmap(color_map).copy()
    cmap.set_over(set_over)

    ax = sns.heatmap(df['accordance'].iloc[0],
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
    file_name = f"{tag}_{task}_{benchmark}_accordance"
    save_path_full = f'{os.path.join(plot_directory,file_name)}.png'
    figure.suptitle( f"{task} {benchmark}")
    figure.savefig(save_path_full,
                   bbox_inches='tight',
                   transparent=True)
    plt.clf()
    return save_path_full

def plot_accordance_heatmap(df: pd.DataFrame, cfg: config_handler.Config):
    """[summary]

    Args:
        df ([type]): [description]
        save_to ([type]): [description]
        color_map (str, optional): [description]. Defaults to 'Blues'.
        set_over (str, optional): [description]. Defaults to 'white'.
    """

    print(f'Creating heatmap...',end='')
    plot_directory = os.path.join(cfg.save_to,'validation_plots')
    os.makedirs(plot_directory,exist_ok=True)
    saved_files = df.groupby(['tag','task','benchmark_name']).apply(lambda _df: _plot_accordance_heatmap(_df,cfg,plot_directory))
    print("done!")
    return saved_files


register.plot_validation_functions_register('heatmap', plot_accordance_heatmap)

