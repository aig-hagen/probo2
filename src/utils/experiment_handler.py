
import csv
from genericpath import exists
import json
from csv import DictWriter, DictReader
from src.utils import config_handler, utils
import os
import pandas as pd

from src.utils import solver_handler,benchmark_handler
from src.utils import definitions
from tqdm import tqdm


def need_additional_arguments(task):
    if 'DC-' in task or 'DS-' in task:
        return True
    else:
        return False

def need_dynamic_files_lookup(task):
    return False

def get_accepted_format(solver_formats, benchmark_formats):
    return list(set.intersection(set(solver_formats), set(benchmark_formats)))[0]

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



def init_result_path(config: config_handler.Config):
    result_file_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
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
        cfg_path = experiment_index[experiment_index.name == name]['config_path'].iloc[0]
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
        last = content[-1]
        header = last.keys()
        dictwriter_object = DictWriter(file, fieldnames=header)
        dictwriter_object.writerow(last)

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
        with open(result_path,'r+') as result_file:
            result_data = _read(result_format, result_file)
            result_data.append(result)
            _write(result_format,result_file, result_data)

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
    result_path = init_result_path(config)
    config.raw_results_path = result_path
    cfg_path_result_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
    if config.save_to is None:
        config.save_to = os.path.join(os.getcwd(), config.name)
    else:
        config.save_to = os.path.join(config.save_to, config.name)
    config.dump(cfg_path_result_directory)
    write_experiment_index(config, cfg_path_result_directory)

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
            print(f"  +Solver:")
            for solver in solver_list:
                if task in solver['tasks']:
                    format = get_accepted_format(solver['format'], benchmark['format'])
                    instances = benchmark_handler.get_instances(benchmark['path'], format)
                    for rep in range(1, config.repetitions + 1):
                        desc = f"    {solver['name']}|REP#{rep}"
                        for instance in tqdm(instances,desc=desc):
                            result = solver_handler.run_solver(solver, task, config.timeout, instance, format, additional_arguments_lookup,dynamic_files_lookup)
                            result.update(benchmark_info)
                            result['repetition'] = rep
                            result['tag'] = config.name
                            write_result(result,result_path,config.result_format)
    print('========== DONE ==========')

