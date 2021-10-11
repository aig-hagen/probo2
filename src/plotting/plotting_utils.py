import pandas as pd
import os
from src.plotting import CactusPlot

def rank_data(df: pd.DataFrame)-> pd.DataFrame:
    return (df
            .sort_values(['runtime'],ascending=True)
            .groupby('solver_id')
            .apply(lambda group:group.assign(rank=range(1,len(group.index) + 1)))
            .droplevel(0)
            .reset_index(drop=True)
            )

def prepare_data_cactus_plot(df: pd.DataFrame):
    cols=['id', 'tag', 'solver_id', 'benchmark_id', 'task_id',
       'cut_off', 'runtime','anon_1','benchmark_name','symbol','instance']
    return (df
     [cols]
     .rename(columns={"anon_1": "solver_full_name","symbol": "task"})
     .astype({'solver_id': 'int16','benchmark_id': 'int16','task_id': 'int8','cut_off': 'int16', 'runtime': 'float32'})
     .groupby(['tag','task_id','benchmark_id','solver_id'])
     .apply(lambda g: g.assign(rank=g['runtime'].rank(method='dense',ascending=True)))
     .droplevel(0)
     .reset_index(drop=True)
     .groupby(['tag','task_id','benchmark_id'])
     )
def prepare_grid(df: pd.DataFrame)-> pd.DataFrame:
    cols=['id', 'tag', 'solver_id', 'benchmark_id', 'task_id',
       'cut_off', 'runtime','anon_1','benchmark_name','symbol']
    return (df
            [cols]
            .rename(columns={"anon_1": "solver_full_name","symbol": "task"})
            .astype({'solver_id': 'int16','benchmark_id': 'int16','task_id': 'int8','cut_off': 'int16', 'runtime': 'float32'})
            .groupby(['tag','task_id','benchmark_id','solver_id'],as_index=False)
            .apply(lambda g: g.assign(rank=g['runtime'].rank(method='dense',ascending=True)))
            .droplevel(0)
            .reset_index(drop=True)
          )


def cactus_plot(df,save_to,options):
    df['Solver'] = df['solver_full_name']
    task_symbol = df['task'].iloc[0]
    benchmark_name = df['benchmark_name'].iloc[0]
    curr_tag = df['tag'].iloc[0]
    options['title'] = task_symbol + " " + benchmark_name
    save_file_name = os.path.join(save_to, "{}_{}_{}_".format(task_symbol, benchmark_name,curr_tag))
    options['save_to'] = save_file_name
    CactusPlot.Cactus(options).create(df)


def cactus_plot_group(grouped_df, save_to, options):
   grouped_df.apply(lambda df: cactus_plot(df,save_to,options))
