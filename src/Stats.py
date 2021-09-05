import pandas
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.functions import count

from functools import reduce

pandas.options.mode.chained_assignment = None

def calculate_total_runtime(data):
    copy = data.copy(deep=True)
    data_solved_instances = copy.loc[copy['timed_out'] == False]
    grouped_df = data_solved_instances.groupby(['task_id','benchmark_id','solver_id'])
    return  grouped_df['runtime'].sum().to_frame().rename(columns={"runtime": "Total"})

def calculate_average_runtime(data):
    copy = data.copy(deep=True)
    data_solved_instances = copy.loc[copy['timed_out'] == False]
    grouped_df = data_solved_instances.groupby(['task_id','benchmark_id','solver_id'])
    return  grouped_df['runtime'].mean().to_frame().rename(columns={"runtime": "Average"})

def get_count_timed_out(data):
    data_solved_instances = data.loc[data['timed_out'] == True]
    
    grouped_df = data_solved_instances.groupby(['task_id','benchmark_id','solver_id'])
    return grouped_df['instance'].count().to_frame().rename(columns={"instance": "#Timeout"})


def calculate_par_scores(data,penalty):
    copy = data.copy(deep=True)
    col_name = 'PAR' + str(penalty)
    copy['instance_count'] = copy.groupby(['task_id','benchmark_id','solver_id'])['instance'].transform('count')
    copy[col_name] = copy.apply(lambda row : calculate_par_score(row['timed_out'],
                     row['runtime'], penalty,row['cut_off']) * (1/row['instance_count']), axis = 1)
    grouped_df = copy.groupby(['task_id','benchmark_id','solver_id'])
    
    return grouped_df[col_name].sum().to_frame()

def calculate_coverage(data):
    copy = data.copy(deep=True)

    inst_count =  copy.groupby(['task_id','benchmark_id','solver_id'])['instance'].count()
    solved = copy.loc[copy['timed_out'] == False].groupby(['task_id','benchmark_id','solver_id'])['instance'].count()
    coverage = (solved / inst_count * 100).to_frame() 
    return coverage.rename(columns={"instance": "Coverage"})
    
   

     


def calculate_par_score(timed_out,runtime,penalty,cut_off):
    if timed_out:
        return cut_off * penalty
    else:
        return runtime

def merge_dataframes(data_frames):
    df = data_frames[0]
    for df_ in data_frames[1:]:
        df = df.merge(df_, left_index=True, right_index=True)
    return df
    
    
