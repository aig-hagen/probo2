
import csv
import json
from csv import DictWriter, DictReader
from src.utils import config_handler
import os
import pandas as pd

from src.utils import solver_handler,benchmark_handler
from src.utils import definitions
from tqdm import tqdm
import shutil


def need_additional_arguments(task: str):
    if 'DC-' in task or 'DS-' in task:
        return True
    else:
        return False

def need_dynamic_arguments(task: str):
    if task.endswith('-D'):
        return True
    else:
        return False

def need_dynamic_files_lookup(task):
    return False

def get_accepted_format(solver_formats, benchmark_formats):
    _shared = list(set.intersection(set(solver_formats), set(benchmark_formats)))

    if _shared:
        return _shared[0]
    else:
        return None


def _format_benchmark_info(benchmark_info):
    formatted = {}
    for key in benchmark_info.keys():
        formatted[f'benchmark_{key}'] = benchmark_info[key]
    return formatted

def _append_result_directoy_suffix(config: config_handler.Config):
    suffix = 1
    while(True):
        experiment_name =  f'{config.name}_{suffix}'
        result_file_directory = os.path.join(definitions.RESULT_DIRECTORY, experiment_name)
        if not os.path.exists(result_file_directory):
            config.name = experiment_name
            return result_file_directory
        else:
            suffix += 1



def init_result_path(config: config_handler.Config, result_file_directory):

    if os.path.exists(result_file_directory):
        result_file_directory = _append_result_directoy_suffix(config)

    os.makedirs(result_file_directory,exist_ok=True)
    result_file_path = os.path.join(result_file_directory, f'raw.{config.result_format}')
    if os.path.exists(result_file_path):
        suffix = len(os.listdir(result_file_directory))
        result_file_path = os.path.join(result_file_directory, f'raw_{suffix}.{config.result_format}')

    return result_file_path

def load_results_via_name(name):

    experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)

    if name in experiment_index.name.values:
        cfg_path = experiment_index[experiment_index.name.values == name]['config_path'].iloc[0]
        cfg = config_handler.load_config_yaml(cfg_path,as_obj=True)
        return load_experiments_results(cfg)
    else:
        print(f"Experiment with name {name} does not exist.")
        exit()



def _write(format,file,content):
    if format == 'json':
        file.seek(0)
        json.dump(content, file)
    elif format == 'csv':

        header = content.keys()
        dictwriter_object = DictWriter(file, fieldnames=header)
        dictwriter_object.writerow(content)

def _write_initial(format,file,content):
    if format == 'json':
        json.dump(content, file)
    elif format == 'csv':
        header = content.keys()
        dictwriter_object = DictWriter(file, fieldnames=header)
        dictwriter_object.writeheader()
        dictwriter_object.writerow(content)

def _read(format,file):
    if format=='json':
        return json.load(file)
    elif format == 'csv':
        return [ row for row in DictReader(file) ]



def write_result(result, result_path, result_format):

    if not os.path.exists(result_path):
        with open(result_path,'w') as result_file:
            _write_initial(result_format,result_file, result)

    else:
        with open(result_path,'a+') as result_file:
            #result_data = _read(result_format, result_file)
            #result_data.append(result)
            _write(result_format,result_file, result)

def write_experiment_index(config: config_handler.Config, result_directory_path):


    header = ['name','raw_path','config_path']
    with open(definitions.EXPERIMENT_INDEX,'a') as fd:

        writer = csv.writer(fd)
        if os.stat(definitions.EXPERIMENT_INDEX).st_size == 0:
            writer.writerow(header)
        writer.writerow([config.name, config.raw_results_path,os.path.join(result_directory_path, config.yaml_file_name)])



def load_experiments_results(config: config_handler.Config)->pd.DataFrame:
    if os.path.exists(config.raw_results_path):
        if config.result_format == 'json':
            return pd.read_json(config.raw_results_path)
        elif config.result_format == 'csv':
            return pd.read_csv(config.raw_results_path)
    else:
        print(f'Results for experiment {config.name} not found!')



def run_experiment(config: config_handler.Config):
    solver_list = solver_handler.load_solver(config.solver)
    benchmark_list = benchmark_handler.load_benchmark(config.benchmark)
    additional_arguments_lookup= None
    dynamic_files_lookup = None
    experiment_result_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
    result_path = init_result_path(config,experiment_result_directory)
    config.raw_results_path = result_path
    if config.save_to is None:
        config.save_to = os.path.join(os.getcwd(), config.name)
    else:
        config.save_to = os.path.join(config.save_to, config.name)
    cfg_experiment_result_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
    config.dump(cfg_experiment_result_directory)
    write_experiment_index(config, cfg_experiment_result_directory)

    print('========== Experiment Summary ==========')
    config.print()
    print('========== RUNNING EXPERIMENT ==========')
    for task in config.task:
        print(f'+TASK: {task}')
        for benchmark in benchmark_list:
            benchmark_info = _format_benchmark_info(benchmark)
            print(f" +BENCHMARK: {benchmark_info['benchmark_name']}")
            if need_additional_arguments(task):
                additional_arguments_lookup = benchmark_handler.generate_additional_argument_lookup(benchmark)
            if need_dynamic_arguments(task):
                dynamic_files_lookup = benchmark_handler.generate_dynamic_file_lookup(benchmark)

                _check_dynamic_files_lookup(dynamic_files_lookup)
            print(f"  +Solver:")
            for solver in solver_list:
                format = get_accepted_format(solver['format'], benchmark['format'])
                if format is not None:
                    if task in solver['tasks']:
                        instances = benchmark_handler.get_instances(benchmark['path'], format)
                        if config.save_output:
                            solver_output_dir = os.path.join(cfg_experiment_result_directory,solver['name'],task,benchmark_info['benchmark_name'])
                            os.makedirs(solver_output_dir,exist_ok=True)
                        else:
                            solver_output_dir = None

                        for rep in range(1, config.repetitions + 1):
                            desc = f"    {solver['name']}|REP#{rep}"
                            for instance in tqdm(instances,desc=desc):
                                result = solver_handler.run_solver(solver, task, config.timeout, instance, format, additional_arguments_lookup,dynamic_files_lookup,output_file_dir=solver_output_dir,repetition=rep)
                                result.update(benchmark_info)
                                result['repetition'] = rep
                                result['tag'] = config.name
                                write_result(result,result_path,config.result_format)
                else:
                    print(f"    {solver['name']} SKIPPED! No files in supported solver format: {','.join(solver['format'])}")
    print('========== DONE ==========')

def _check_dynamic_files_lookup(dynamic_files_lookup):
    missing = []

    for format, instances in dynamic_files_lookup.items():
        if not instances:
            missing.append(format)

    if missing:
        print(f"No modification files found for instances in format: {','.join(missing)}")
        exit()

def copy_raws(config: config_handler.Config):
    os.makedirs(config.save_to, exist_ok=True)
    shutil.copy(config.raw_results_path, config.save_to)
