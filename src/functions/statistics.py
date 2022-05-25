import src.functions.register as register
import pandas as pd

def _get_avg_reps(df: pd.DataFrame):
    result_dict = {}
    for col in df.columns:
        result_dict[col] = df[col].iloc[0]
    result_dict['runtime'] = df.runtime.mean()
    return pd.Series(result_dict)

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

def _calculate_sum(df):
    return (pd.Series({'solver_name': df.solver_name.iloc[0],
                    'solver_version': df.solver_version.iloc[0],
                    'benchmark_name':df.benchmark_name.iloc[0],
                    'sum':df.runtime.sum()}))
def sum(df: pd.DataFrame):
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    rep_avg_df = df[only_solved_mask].groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_sum(_df))
    print(groups)
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes

def _calculate_mean(df):
    return (pd.Series({'solver_name': df.solver_name.iloc[0],
                    'solver_version': df.solver_version.iloc[0],
                    'benchmark_name':df.benchmark_name.iloc[0],
                    'mean':df.runtime.mean()}))
def mean(df: pd.DataFrame):
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    rep_avg_df = df[only_solved_mask].groupby(['tag', 'task', 'benchmark_id', 'solver_id','instance'],as_index=False).apply(lambda _df: _get_avg_reps(_df))
    grouping = ['tag', 'task', 'benchmark_id','solver_id']
    groups = rep_avg_df.groupby(grouping,as_index=False).apply(lambda _df: _calculate_mean(_df))
    print(groups)
    return groups, True # indicates that this dataframe should be merged with other "mergable" dataframes

# Bei mehreren Runs muss immer der Durschnitt der Runs genommen werden

register.register_stat("best",get_number_best_runtime)
register.register_stat("worst",get_number_worst_runtime)
register.register_stat("sum",sum)
register.register_stat("mean",mean)