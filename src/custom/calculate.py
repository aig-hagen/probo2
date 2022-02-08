import src.custom.register as register
import pandas as pd


def _increment_best_solver(df, solver_dict):
    fastes = df[df.runtime == df.runtime.min()].solver_full_name.unique()

    for fastes_solver_name in fastes:
        solver_dict[fastes_solver_name] += 1

def _increment_worst_solver(df,solver_dict):
    slowest = df[df.runtime == df.runtime.max()].solver_full_name.unique()

    for slowest_solver_name in slowest:
        solver_dict[slowest_solver_name] += 1


def get_number_best_runtime(df):

    grouping = ['tag', 'task_id', 'benchmark_id', 'instance']
    solver_dict = {s:0 for s in df.solver_full_name.unique()}
    df[df.timed_out == False].groupby(grouping).apply(lambda _df: _increment_best_solver(_df,solver_dict))

    res_df = pd.DataFrame.from_dict(solver_dict,orient='index').reset_index()
    res_df.columns = ['solver','#best']
    return res_df

def get_number_worst_runtime(df):
    grouping = ['tag', 'task_id', 'benchmark_id', 'instance']
    solver_dict = {s:0 for s in df.solver_full_name.unique()}
    df.groupby(grouping).apply(lambda _df: _increment_worst_solver(_df,solver_dict))

    res_df = pd.DataFrame.from_dict(solver_dict,orient='index').reset_index()
    res_df.columns = ['solver','#worst']
    return res_df

register.register_stat("best",get_number_best_runtime)
register.register_stat("worst",get_number_worst_runtime)