"""Statistics module"""

from numpy.lib.utils import info
import pandas as pd
pd.options.mode.chained_assignment = None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    calculate_cols = [
        'id', 'tag', 'solver_id', 'benchmark_id', 'task_id', 'cut_off',
        'timed_out', 'exit_with_error', 'runtime', 'anon_1', 'benchmark_name',
        'symbol','validated','correct','instance'
    ]
    return (df[calculate_cols].rename(columns={
        "anon_1": "solver_full_name",
        "symbol": "task"
    }).astype({
        'solver_id': 'int16',
        'benchmark_id': 'int16',
        'task_id': 'int8',
        'cut_off': 'int16',
        'runtime': 'float32'
    }))


def calculate_total_runtimes(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    return (df[(df['timed_out'] == False)
               & (df['exit_with_error'] == False)].groupby([
                   'solver_id', 'benchmark_id', 'task_id', 'tag'
               ])["runtime"].sum().reset_index(name='total_runtime_solved'))

def calculate_total_runtime(df: pd.DataFrame) -> pd.Series:
    info = get_info_as_strings(df)
    total_runtime = (df[(df['timed_out'] == False) &
                        (df['exit_with_error'] == False)]
                        ['runtime']
                        .values.sum()
                    )
    info['total_runtime'] = total_runtime

    return pd.Series(info)


def calculate_average_runtimes(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    return (df[(df['timed_out'] == False)
               & (df['exit_with_error'] == False)].groupby([
                   'solver_id', 'benchmark_id', 'task_id', 'tag'
               ])["runtime"].mean().reset_index(name='mean_runtime_solved'))


def get_info_as_strings(df: pd.DataFrame) -> dict:
    tags = ",".join(df.tag.unique())
    solvers = ",".join(df.solver_full_name.unique())
    tasks = ",".join(df.task.unique())
    benchmarks = ",".join(df.benchmark_name.unique())
    return {'tag': tags,'solver': solvers, 'task': tasks,'benchmark': benchmarks}

def calculate_par_score(df: pd.DataFrame, penalty: int) -> dict:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        penalty (int): [description]

    Returns:
        float: [description]
    """
    info = get_info_as_strings(df)
    cut_off = df['cut_off'].iloc[0]
    num_unsolved = df[(df['timed_out'] == True) |
                      (df['exit_with_error'] == True)].shape[0]
    sum_runtime_unsolved = num_unsolved * cut_off * penalty
    sum_runtime_solved = df[(df['timed_out'] == False) & (
        df['exit_with_error'] == False)]['runtime'].values.sum()
    num_total_instances = df.shape[0]
    par_score = (sum_runtime_solved + sum_runtime_unsolved) / num_total_instances
    info[f'PAR{penalty}'] = par_score
    return pd.Series(info)


def calculate_par_scores(df: pd.DataFrame, penalty: int) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        penalty (int): [description]

    Returns:
        pd.DataFrame: [description]
    """
    return (df.groupby([
        'tag', 'task_id', 'benchmark_id', 'solver_id'
    ]).apply(lambda df: calculate_par_score(df, penalty)).reset_index(
        name=f'PAR{penalty}'))


def calculate_coverage(df: pd.DataFrame) -> pd.Series:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        float: [description]
    """
    info = get_info_as_strings(df)
    total = df.shape[0]
    solved = df[(df['timed_out'] == False)
                & (df['exit_with_error'] == False)].shape[0]
    coverage = solved / total * 100
    info['coverage'] = coverage
    info['total'] = total
    info['solved'] = solved
    return pd.Series(info)

def calculate_iccma_score(df: pd.DataFrame)-> pd.Series:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        : [description]
    """

    info = get_info_as_strings(df)
    iccma_score = df[(df['timed_out'] == False)
                & (df['exit_with_error'] == False) & (df.validated == True) & (df.correct == "correct")].shape[0]
    print(df['solver_full_name'].iloc[0],iccma_score)
    info['iccma_score'] = iccma_score
    return pd.Series(info)



def merge_dataframes(data_frames, on):
    """[summary]

    Args:
        data_frames ([type]): [description]
        on ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = data_frames[0]
    for df_ in data_frames[1:]:
        df = df.merge(df_, on=on)
    return df

def create_vbs(df,vbs_id):


    best_runtime = df['runtime'].min()


    row = df.iloc[0]
    row['solver_id'] = vbs_id
    row['solver_full_name'] = "vbs"
    row['id'] = 0
    row['runtime'] = best_runtime
    row['validated'] = True
    row['correct'] = 'correct'
    if  pd.isna(best_runtime):
        row['timed_out'] = True
    else:
        row['timed_out'] = False


    return row
