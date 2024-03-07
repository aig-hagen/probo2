"""Module to generate node and graph labels"""
from src.handler import solver_handler
from dataclasses import dataclass
from src.utils.experiment_handler import get_accepted_format
from src.handler import benchmark_handler
import pandas as pd
import os
from tqdm import tqdm

@dataclass(frozen=False)
class AccaptanceLabelOptions():
    solver: dict
    tasks: list
    benchmarks: list
    timeout: str
    save_to: str

def _task_supported(task: str, solver: dict):
    if task in solver['tasks']:
        return True
    else:
        return False

def tgf_extract_arguments(tgf_file):
    with open(tgf_file,'r') as f:
        tgf_file_content = f.read()
    args_attacks = tgf_file_content.split('#\n')
    args = args_attacks[0].split()
    return args

def apx_extract_arguments(apx_file):
    with open(apx_file,'r') as f:
        apx_file_content = f.read()
    args_attacks = apx_file_content.split('att')
    args = args_attacks[0].split()
    args = [ s[s.find("(")+1:s.find(")")] for s in args]
    return args

def i23_extract_arguments(i23_file):
    pass

arguments_extraction_functions = {'apx': apx_extract_arguments,'tgf': tgf_extract_arguments,'i23': i23_extract_arguments}

def get_label_accaptance(options: AccaptanceLabelOptions):
    print("========== GENERATING LABELS ==========")
    for task in tqdm(options.tasks,position=0,desc=f'Tasks'):
        if not _task_supported(task,options.solver):
            print(f'Solver {options.solver["name"]} does not support the task {task}')

        for benchmark in tqdm(options.benchmarks,desc=f'{task}-Benchmarks',position=1,leave=False):
            save_to = os.path.join(options.save_to,f"{benchmark['name']}_{task}_labels")
            os.makedirs(save_to,exist_ok=True)
            format = get_accepted_format(options.solver['format'], benchmark['format'])
            if format is None:
                print(f'Solver {options.solver["name"]} does not support the format: {benchmark["format"]}')
                exit()
            instances = benchmark_handler.get_instances(benchmark['path'], format)

            for instance in tqdm(instances,position=2,desc=f'{benchmark["name"]}-Instances',leave=False):
                instance_name = os.path.basename(instance)[:-4]
                arguments = arguments_extraction_functions[format](instance)
                instance_results = []
                for arg in tqdm(arguments,position=3,desc=f"{instance_name}-Arguments",leave=False):
                    result = solver_handler.run_solver_accaptence(options.solver,task,instance,arg,options.timeout,format)
                    instance_results.append(result)

                _write_instance_labels_to_csv(save_to, instance_name, instance_results)
        print("========== DONE ==========")

def _write_instance_labels_to_csv(save_to, instance_name, instance_results):
    instance_label_save_to = os.path.join(save_to, f'{instance_name}.csv')
    instance_df = pd.DataFrame(instance_results)
    instance_df['label'] = instance_df.accepted.astype(int)
    instance_df.to_csv(instance_label_save_to)

def label_fastes_solvers(df: pd.DataFrame):
    pass



if __name__ == "__main__":
    solver = solver_handler.load_solver([5])[0]
    benchmark = benchmark_handler.load_benchmark(['3'])
    tasks = ['DS-PR','DC-PR']
    timout = 600
    save_to = os.getcwd()
    print(solver)
    print(benchmark)

    options = AccaptanceLabelOptions(solver=solver,tasks=tasks,benchmarks=benchmark,timeout=timout,save_to=save_to)
    get_label_accaptance(options)