import pandas as pd
def get_pipeline_table(df: pd.DataFrame,reps,tasks,benchmarks,solvers):
    i_df = df.interactive()
    filtered_i_df = (
     i_df[
          (i_df.repetition.isin(reps)) &
          (i_df.task.isin(tasks)) &
          (i_df.benchmark_name.isin(benchmarks)) &
          
         (i_df.solver_full_name.isin(solvers))
     ])
    return filtered_i_df 


def get_pipeline_with_rep_cactus(df,reps,tasks,benchmarks,solvers):
    i_df = df.interactive()
    filtered_i_df = (
     i_df[
          (i_df.repetition.isin(reps)) &
          (i_df.task.isin(tasks)) &
          (i_df.benchmark_name.isin(benchmarks)) &
          
         (i_df.solver_full_name.isin(solvers)) &
         (i_df.timed_out == False) &
         (i_df.exit_with_error == False) 
    ])
    return filtered_i_df 

def get_pipeline_with_rep_count_plots(df,reps,tasks,benchmarks,solvers):
    i_df = df.interactive()
    filtered_i_df = (
     i_df[
          (i_df.repetition.isin(reps)) &
          (i_df.task.isin(tasks)) &
          (i_df.benchmark_name.isin(benchmarks))
     ])
    return filtered_i_df 

