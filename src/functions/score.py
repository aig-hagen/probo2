
from select import select
from typing_extensions import runtime
import src.functions.register as register
import pandas as pd

from src.functions.statistics import _get_avg_reps
import numpy as np
from math import log



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

    df.runtime = df.runtime.fillna(df.cut_off.iloc[0])
    rep_avg_df = df.groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_ipc(_df))
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes

register.register_score_function('ipc',ipc)