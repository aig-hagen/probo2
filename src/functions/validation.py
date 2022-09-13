
import src.functions.register as register
import pandas as pd
import numpy as np
import src.utils.config_handler as config_handler
import matplotlib.pyplot as plt

def validate(df: pd.DataFrame,config: config_handler.Config):
    validation_cfg = config.validation
    if validation_cfg['mode'] =='all' or 'all' in validation_cfg['mode']:
            validation_cfg['mode'] = register.validation_functions_dict.keys()
    results = {}
    for m in validation_cfg['mode']:
        if m in register.validation_functions_dict.keys():
            _res = register.validation_functions_dict[m](df, config)
            results[m] = _res
    
    return results

    

def _map_index_names(df:pd.DataFrame, map):
    pass
def _map_column_names(df:pd.DataFrame, map):
    pass



def pairwise(df: pd.DataFrame, config: config_handler.Config ):
    if config.save_output:
        validation_results = df.groupby(['tag','task','benchmark_name'],as_index=False).apply(lambda _df: _pairwise_validation(_df, config))
        # id_names = df[['solver_id','solver_name','solver_version']]
        # id_names['solver_full_name'] = id_names['solver_name'] +'_'+id_names['solver_version']
        # print(id_names)
        return validation_results

    else:
        print("No computational results available for this experiment.")
        exit()

def _pairwise_validation(df: pd.DataFrame, config: config_handler.Config):
    only_solved_mask = (df.timed_out == False) & (df.exit_with_error == False)
    instance_groups = df[only_solved_mask].groupby('instance')
    counts = df[only_solved_mask].groupby('solver_id',as_index=False).apply(lambda _df: len(_df.instance.unique()))
    counts.columns = counts.columns.fillna('instance_count')
    unique_solvers = df.solver_id.unique()
    solved_pairwise_matrix = pd.DataFrame(columns=unique_solvers, index=unique_solvers).fillna(0)
    same_result_matrix = pd.DataFrame(columns=unique_solvers, index=unique_solvers).fillna(0)

    id_names_unique = list(df.groupby(['solver_id','solver_name','solver_version']).groups)
    name_map = { x[0]: f'{x[1]}_{x[2]}' for x in id_names_unique }


    for name,instance_group in instance_groups:
        solver_group = instance_group.groupby('solver_id',as_index=False).apply(lambda _df: _df.iloc[0])
        solver_id = solver_group.solver_id.unique()
        _increment_solved_pairwise_matrix(solved_pairwise_matrix,solver_id)
        _increment_same_result_matrix(same_result_matrix, solver_group,solver_id)
    accordance_matrix = (same_result_matrix / solved_pairwise_matrix * 100).replace([np.inf,-np.inf], 0)

    accordance_matrix.index = accordance_matrix.index.map(name_map)
    accordance_matrix.columns = accordance_matrix.columns.map(name_map)

    same_result_matrix.index = same_result_matrix.index.map(name_map)
    same_result_matrix.columns = same_result_matrix.columns.map(name_map)

    solved_pairwise_matrix.index = solved_pairwise_matrix.index.map(name_map)
    solved_pairwise_matrix.columns = solved_pairwise_matrix.columns.map(name_map)

    return pd.Series({'same_results': same_result_matrix,'solved_pairwise': solved_pairwise_matrix, 'accordance': accordance_matrix})

def _increment_solved_pairwise_matrix(solved_pairwise_matrix: pd.DataFrame, solver_id: list):
    for i in range(0,len(solver_id)):
        solved_pairwise_matrix[solver_id[i]][solver_id[i]] += 1
        for j in range(i+1,len(solver_id)):
            solved_pairwise_matrix[solver_id[i]][solver_id[j]] += 1
            solved_pairwise_matrix[solver_id[j]][solver_id[i]] += 1

def _increment_same_result_matrix(same_result_matrix: pd.DataFrame, result_df: pd.DataFrame, solver_id: list):
    for i in range(0,len(solver_id)):

        same_result_matrix[solver_id[i]][solver_id[i]] += 1
        current_row = result_df[result_df.solver_id == solver_id[i]]
        current_solver_result = _read_result(current_row.task.iloc[0],current_row.result.iloc[0])
        for j in range(i+1,len(solver_id)):

            other_row =  result_df[result_df.solver_id == solver_id[j]]
            solver_id[j]
            other_solver_results = _read_result(other_row.task.iloc[0],other_row.result.iloc[0])
            if current_solver_result == other_solver_results:
                same_result_matrix[solver_id[i]][solver_id[j]] += 1
                same_result_matrix[solver_id[j]][solver_id[i]] += 1


def _read_decision_task_file(path):
    with open(path,'r') as res_file:
        res = res_file.read().rstrip()
    return res


def _read_result(task,path):
    if any(decision_task in task for decision_task in ['DC-','DS-','CE-']):
        return _read_decision_task_file(path)
    else:
        pass

register.validation_functions_register("pairwise",pairwise)
