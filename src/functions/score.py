
from select import select
from typing_extensions import runtime
import src.functions.register as register
import pandas as pd

from src.functions.statistics import _get_avg_reps
import numpy as np
from math import log




def _calculate_par_score(df: pd.DataFrame, penalty_factor, cutoff):
    """
    Calculates the Penalized Average Runtime (PAR) score for solver runs.
    
    Parameters:
    - df: Pandas dataframe containing solver run data, including 'timed_out' and 'runtime' columns.
    - penalty_factor: The penalty factor to apply to runs that timed out.
    - cutoff: The cutoff time in seconds for considering a run as timed out.
    
    Returns:
    - The PAR score as a float.
    """
    # Apply penalty for timed-out runs
    df['penalized_runtime'] = np.where(df['timed_out'], penalty_factor * cutoff, df['runtime'])
    
    # Calculate the average of the penalized runtimes
    par_score = df['penalized_runtime'].mean()

    return (pd.Series({'solver_name': df.solver_name.iloc[0],
                    'solver_version': df.solver_version.iloc[0],
                    'benchmark_name':df.benchmark_name.iloc[0],
                    f'PAR{penalty_factor}':par_score}))

def par10(df: pd.DataFrame):
    df_clean = df[df.exit_with_error == False]
    cut_off = df_clean.cut_off.iloc[0]
    df_clean.runtime = df_clean.runtime.fillna(cut_off)
    rep_avg_df = df_clean.groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_par_score(_df,10,cut_off))
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes

def par2(df: pd.DataFrame):
    df_clean = df[(df.exit_with_error == False)]
    cut_off = df_clean.cut_off.iloc[0]
    df_clean.runtime = df_clean.runtime.fillna(cut_off)
    rep_avg_df = df_clean.groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_par_score(_df,2,cut_off))
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes


def _calculate_ipc(df):
    
    cut_off = df.cut_off.iloc[0]
    runtimes = df.runtime.values
    runtime_conditions = [ runtimes <= 1.0,runtimes >=cut_off, (runtimes > 1.0) & (runtimes < cut_off)]
    choices = [1,0,np.minimum(np.ones(runtimes.size), (np.log10(runtimes)/np.log10(cut_off)) * (-1) + 1)]
    ipc_scores = np.select(runtime_conditions,choices)
    return (pd.Series({'solver_name': df.solver_name.iloc[0],
                    'solver_version': df.solver_version.iloc[0],
                    'benchmark_name':df.benchmark_name.iloc[0],
                    'IPC':ipc_scores.sum()}))



def ipc(df: pd.DataFrame):
    df_clean = df[df.exit_with_error == False]
    df_clean.runtime = df_clean.runtime.fillna(df.cut_off.iloc[0])
    rep_avg_df = df.groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_ipc(_df))
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes

register.register_score_function('ipc',ipc)
register.register_score_function('PAR10',par10)
register.register_score_function('PAR2',par2)