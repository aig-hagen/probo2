"""Module to generate node and graph labels"""
from src.utils import solver_handler
from dataclasses import dataclass
from src.utils.experiment_handler import get_accepted_format
from src.utils import benchmark_handler

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
    for task in options.tasks:
        if not _task_supported(task,options.solver):
            print(f'Solver {options.solver["name"]} does not support the task {task}')
        
        for benchmark in options.benchmarks:
            format = get_accepted_format(options.solver['format'], benchmark['format'])
            if format is None:
                print(f'Solver {options.solver["name"]} does not support the format: {benchmark["format"]}')
                exit()
            instances = benchmark_handler.get_instances(benchmark['path'], format)

            for instance in instances:
                arguments = arguments_extraction_functions[format](instance)

                for arg in arguments:
                    solver_handler.run_solver_accaptence(options.solver,task,instance,arg,options.timeout,format)
                

if __name__ == "__main__":
    solver = solver_handler.load_solver([5])[0]
    benchmark = benchmark_handler.load_benchmark(['2'])
    tasks = ['DS-PR','DC-PR']
    timout = 600
    print(solver)
    print(benchmark)
    
    options = AccaptanceLabelOptions(solver=solver,tasks=tasks,benchmarks=benchmark,timeout=timout,save_to=None)
    get_label_accaptance(options)