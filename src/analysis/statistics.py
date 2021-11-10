"""Statistics module"""


import pandas as pd



def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    calculate_cols = [
        'id', 'solver_id', 'anon_1', 'instance', 'solver_format', 'runtime',
       'task_id', 'symbol', 'cut_off', 'timed_out',
       'exit_with_error', 'error_code', 'additional_argument', 'benchmark_id',
        'validated', 'benchmark_name', 'tag','correct_solved', 'incorrect_solved', 'no_reference','correct'
    ]
    return (df[calculate_cols]
            .rename(columns={
        "anon_1": "solver_full_name",
        "symbol": "task"
     }).astype({
         'solver_id': 'int16',
         'benchmark_id': 'int16',
        'task_id': 'int8',
         'cut_off': 'int16',
         'runtime': 'float32',
         'instance': 'str',
         'solver_format': 'str',
         'task': 'str',
         'tag': 'str',
        'benchmark_name': 'str',
        'correct_solved': 'bool', 'incorrect_solved': 'bool', 'no_reference': 'bool'
    })
    )

def get_info_as_strings(df: pd.DataFrame) -> dict:
    tags = ",".join(df.tag.unique())
    solvers = ",".join(df.solver_full_name.unique())
    tasks = ",".join(df.task.unique())
    benchmarks = ",".join(df.benchmark_name.unique())
    return {'tag': tags,'solver': solvers, 'task': tasks,'benchmark': benchmarks}

def sum_runtimes(df):
    return df.runtime.sum()

def mean_runtimes(df):
    return df.runtime.mean()

def min_runtimes(df):
    return df.runtime.min()

def max_runtimes(df):
    return df.runtime.max()
def sum_timed_out(df):
    return df.timed_out.sum()
def sum_exit_with_error(df):
    return df.exit_with_error.sum()
def var_runtimes(df):
    return df.runtime.var()
def median_runtimes(df):
    return df.runtime.median()
def std_runtimes(df):
    return df.runtime.std()

def coverage(df):
    total = df.instance.shape[0]
    solved = total -(sum_timed_out(df) + sum_exit_with_error(df))

    return solved / total * 100


def dispatch_function(df,functions,par_penalty=10):

    dispatch_map = {'mean':mean_runtimes(df),
                    'sum': sum_runtimes(df),
                    'min':min_runtimes(df),
                    'max':max_runtimes(df),
                    'median':median_runtimes(df),
                    'var': var_runtimes(df),
                    'std': std_runtimes(df),
                    f'PAR{par_penalty}': penalised_average_runtime(df,par_penalty),
                    'coverage': coverage(df)
    }
    functions_to_call = {key: dispatch_map[key] for key in functions}
    info = get_info_as_strings(df)
    ser_dict = dict(info,**functions_to_call)
    return pd.Series(ser_dict)


def penalised_average_runtime(df, penalty):
    cut_off = df.cut_off.iloc[0]
    solved_instances = df[(df.timed_out == False)&(df.exit_with_error == False)]
    solved_runtime = solved_instances.runtime.values.sum()
    total_num = df.shape[0]
    solved_num = solved_instances.shape[0]
    unsolved_num =  total_num - solved_num
    unsolved_runtime = unsolved_num * cut_off * penalty
    return (solved_runtime + unsolved_runtime) / total_num

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
    row['correct_solved'] = True
    row['correct'] = 'correct'
    if  pd.isna(best_runtime):
        row['timed_out'] = True
    else:
        row['timed_out'] = False


    return row


# def calculate_total_runtimes(df: pd.DataFrame,grouping: list) -> pd.DataFrame:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]

#     Returns:
#         pd.DataFrame: [description]
#     """
#     return (df
#             .groupby(grouping,as_index=False)
#             .apply(lambda group: calculate_total_runtime(group))
#                         )

# def calculate_total_runtime(df: pd.DataFrame) -> pd.Series:
#     info = get_info_as_strings(df)
#     total_runtime = (df[(df['timed_out'] == False) &
#                         (df['exit_with_error'] == False)]
#                         ['runtime']
#                         .values.sum()
#                     )
#     info['total_runtime'] = total_runtime

#     return pd.Series(info)

# def calculate_avg_runtime(df: pd.DataFrame) -> pd.Series:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]

#     Returns:
#         pd.Series: [description]
#     """
#     info = get_info_as_strings(df)
#     avg_runtime = (df[(df['timed_out'] == False) &
#                         (df['exit_with_error'] == False)]
#                         ['runtime']
#                         .values.mean()
#                     )
#     info['average_runtime'] = avg_runtime

#     return pd.Series(info)

# def calculate_avg_runtimes(df: pd.DataFrame, grouping: list) -> pd.DataFrame:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]

#     Returns:
#         pd.DataFrame: [description]
#     """
#     return (df
#             .groupby(grouping,as_index=False)
#             .apply(lambda group: calculate_avg_runtime(group))
#                         )


# def calculate_par_score(df: pd.DataFrame, penalty: int) -> dict:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]
#         penalty (int): [description]

#     Returns:
#         float: [description]
#     """
#     info = get_info_as_strings(df)
#     cut_off = df['cut_off'].iloc[0]
#     num_unsolved = df[(df['timed_out'] == True) |
#                       (df['exit_with_error'] == True)].shape[0]
#     sum_runtime_unsolved = num_unsolved * cut_off * penalty
#     sum_runtime_solved = df[(df['timed_out'] == False) & (
#         df['exit_with_error'] == False)]['runtime'].values.sum()
#     num_total_instances = df.shape[0]
#     par_score = (sum_runtime_solved + sum_runtime_unsolved) / num_total_instances
#     info[f'PAR{penalty}'] = par_score
#     return pd.Series(info)


# def calculate_par_scores(df: pd.DataFrame, penalty: int, grouping: list) -> pd.DataFrame:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]
#         penalty (int): [description]
#         grouping (list): [description]

#     Returns:
#         pd.DataFrame: [description]
#     """
#     return (df
#             .groupby(grouping,as_index=False)
#             .apply(lambda group: calculate_par_score(group, penalty))
#             )

# def calculate_coverages(df: pd.DataFrame, grouping: list) -> pd.DataFrame:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]
#         grouping (list): [description]

#     Returns:
#         pd.DataFrame: [description]
#     """
#     return (df
#             .groupby(grouping,as_index=False)
#             .apply(lambda group: calculate_coverage(group))
#             )
# def calculate_coverage(df: pd.DataFrame) -> pd.Series:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]

#     Returns:
#         float: [description]
#     """
#     info = get_info_as_strings(df)
#     total = df.shape[0]
#     solved = df[(df['timed_out'] == False)
#                 & (df['exit_with_error'] == False)].shape[0]
#     coverage = solved / total * 100
#     info['coverage'] = coverage
#     info['total'] = total
#     info['solved'] = solved
#     return pd.Series(info)

# def calculate_iccma_scores(df: pd.DataFrame, grouping: list) -> pd.DataFrame:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]
#         grouping (list): [description]

#     Returns:
#         pd.DataFrame: [description]
#     """
#     return (df
#             .groupby(grouping,as_index=False)
#             .apply(lambda group: calculate_iccma_score(group))
#             )
# def calculate_iccma_score(df: pd.DataFrame)-> pd.Series:
#     """[summary]

#     Args:
#         df (pd.DataFrame): [description]

#     Returns:
#         : [description]
#     """

#     info = get_info_as_strings(df)
#     iccma_score = df[(df['timed_out'] == False)
#                 & (df['exit_with_error'] == False) & (df.validated == True) & (df.correct_solved == True)].shape[0]
#     print(df['solver_full_name'].iloc[0],iccma_score)
#     info['iccma_score'] = iccma_score
#     return pd.Series(info)




