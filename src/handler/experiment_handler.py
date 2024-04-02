
import csv
import json
from csv import DictWriter, DictReader
from src.handler import config_handler
import os
import pandas as pd

from src.handler import solver_handler,benchmark_handler
from src.utils import definitions
from tqdm import tqdm
import shutil
import tabulate
from src.utils import Status
from click import echo

from src.functions import plot,statistics,score,printing,table_export
from src.functions import validation,plot_validation,validation_table_export,print_validation
from src.functions import parametric_significance,parametric_post_hoc
from src.functions import non_parametric_significance,non_parametric_post_hoc
from src.functions import plot_significance,post_hoc_table_export,plot_post_hoc,print_significance
import src.functions.register as register
from functools import reduce


def run_pipeline(cfg: config_handler.Config):
    run_experiment(cfg)

    if cfg.copy_raws:
        echo('Copying raw files...',nl=False)
        copy_raws(cfg)
        echo('done!')

    result_df = load_results_via_name(cfg.name)



    saved_file_paths = []
    if cfg.plot is not None:
        saved_plots = plot.create_plots(result_df,cfg)

    to_merge = []
    others =  []
    if cfg.statistics is not None:

        if cfg.statistics =='all' or 'all' in cfg.statistics:
            cfg.statistics = register.stat_dict.keys()
        stats_results = []
        print("========== STATISTICS ==========")
        for stat in cfg.statistics:
            _res = register.stat_dict[stat](result_df)
            stats_results.append(_res)

        for res in stats_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if cfg.score is not None:
        score_results = []
        if cfg.score =='all' or 'all' in cfg.score:
                    cfg.score = register.score_functions_dict.keys()
        for s in cfg.score:
            _res = register.score_functions_dict[s](result_df)
            score_results.append(_res)
        for res in score_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if len(to_merge) >0:
        df_merged = reduce(lambda  left,right: pd.merge(left,right,how='inner'), to_merge)
        register.print_functions_dict[cfg.printing](df_merged,['tag','task','benchmark_name'])
        if cfg.table_export is not None:
            if cfg.table_export == 'all' or 'all' in cfg.table_export:
                cfg.table_export = register.table_export_functions_dict.keys()
            for format in cfg.table_export:
                register.table_export_functions_dict[format](df_merged,cfg,['tag','task','benchmark_name'])

    if cfg.validation['mode']:
        validation_results = validation.validate(result_df, cfg)
        print_validation.print_results(validation_results)
        if cfg.validation['plot']:
            plot_validation.create_plots(validation_results['pairwise'],cfg)
        if 'pairwise' in cfg.validation['mode']:
            if cfg.validation['table_export']:
                if cfg.validation['table_export'] == 'all' or 'all' in cfg.validation['table_export']:
                    cfg.validation['table_export'] = register.validation_table_export_functions_dict.keys()
                for f in cfg.validation['table_export']:
                    register.validation_table_export_functions_dict[f](validation_results['pairwise'],cfg)

    test_results = {}
    post_hoc_results = {}

    if cfg.significance['parametric_test']:
        test_results.update(parametric_significance.test(result_df,cfg))
    if cfg.significance['non_parametric_test']:
        test_results.update(non_parametric_significance.test(result_df,cfg))
    if cfg.significance['parametric_post_hoc']:
        post_hoc_results.update(parametric_post_hoc.test(result_df,cfg))
    if cfg.significance['non_parametric_post_hoc']:
        post_hoc_results.update(non_parametric_post_hoc.test(result_df,cfg))

    if test_results:
        print("========== Significance Analysis Summary ==========")
        for test in test_results.keys():
            print_significance.print_results(test_results[test],test)

    if post_hoc_results:
        print("========== Post-hoc Analysis Summary ==========")
        for test in post_hoc_results.keys():
            print_significance.print_results_post_hoc(post_hoc_results[test],test)

        if cfg.significance['plot']:
            for post_hoc_test in post_hoc_results.keys():
                plot_post_hoc.create_plots(post_hoc_results[post_hoc_test],cfg,post_hoc_test)
        if cfg.significance['table_export']:
            if cfg.significance['table_export'] == 'all' or 'all' in cfg.significance['table_export']:
                cfg.significance['table_export'] = register.post_hoc_table_export_functions_dict.keys()
            for post_hoc_test in post_hoc_results.keys():
                for f in cfg.significance['table_export']:
                    register.post_hoc_table_export_functions_dict[f](post_hoc_results[post_hoc_test],cfg,post_hoc_test)




    if cfg.archive is not None:
        echo('Creating archives...',nl=False)
        for _format in cfg.archive:
            register.archive_functions_dict[_format](cfg.save_to)
        echo('done!')


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

def exclude_task(config: config_handler.Config):
    to_exclude = []
    for current_task in config.task:
        for task in config.exclude_task:
            if task in current_task:
               to_exclude.append(current_task)


    config.task = [ task for task in config.task if task not in to_exclude]


def set_supported_tasks(solver_list, config: config_handler.Config):
    supported_tasks = [ set(solver['tasks']) for solver in solver_list]
    supported_set = set.union(*supported_tasks)
    config.task = list(supported_set)

def run_experiment(config: config_handler.Config):


    solver_list = solver_handler.load_solver(config.solver)
    benchmark_list = benchmark_handler.load_benchmark(config.benchmark)
    if config.task == 'supported':
        set_supported_tasks(solver_list,config)
    if config.exclude_task is not None:
        exclude_task(config)

    additional_arguments_lookup= None
    dynamic_files_lookup = None

    #if config.solver_arguments:
     #  config_handler.create_solver_argument_grid(config.solver_arguments, solver_list)

    experiment_result_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
    result_path = init_result_path(config,experiment_result_directory)
    config.raw_results_path = result_path
    if config.save_to is None:
        config.save_to = os.path.join(os.getcwd(), config.name)
    else:
        config.save_to = os.path.join(config.save_to, config.name)
    cfg_experiment_result_directory = os.path.join(definitions.RESULT_DIRECTORY, config.name)
    status_file_path = os.path.join(cfg_experiment_result_directory,'status.json')
    config.status_file_path = status_file_path
    Status.init_status_file(config)
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
            if need_additional_arguments(task) and len(benchmark_info['benchmark_ext_additional']) == 1:
                additional_arguments_lookup = benchmark_handler.generate_additional_argument_lookup(benchmark)

            if need_dynamic_arguments(task):
                dynamic_files_lookup = benchmark_handler.generate_dynamic_file_lookup(benchmark)

                _check_dynamic_files_lookup(dynamic_files_lookup)
            print(f"  +Solver:")
            for solver in solver_list:
                
                solver_options = config.solver_options.get(solver['id']) if config.solver_options else None
             
                format = get_accepted_format(solver['format'], benchmark['format'])
                if format is not None:
                    if need_additional_arguments(task) and len(benchmark_info['benchmark_ext_additional']) > 1:
                        additional_arguments_lookup = benchmark_handler.generate_additional_argument_lookup(benchmark, format)
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
                                result = solver_handler.run_solver(solver, task, config.timeout, instance, format, additional_arguments_lookup,dynamic_files_lookup,output_file_dir=solver_output_dir,repetition=rep,solver_options=solver_options)
                                result.update(benchmark_info)
                                result['repetition'] = rep
                                result['tag'] = config.name
                                write_result(result,result_path,config.result_format)
                                if rep == 1:
                                    Status.increment_instances_counter(config,task,solver['id'])
                else:
                    print(f"    {solver['name']} SKIPPED! No files in supported solver format: {','.join(solver['format'])}")
        Status.increment_task_counter()
    print('')

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

def get_last_experiment():
    if os.path.exists(definitions.EXPERIMENT_INDEX):
        experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)
        return  experiment_index.iloc[-1]

    else:
       return None

def print_experiment_index(tablefmt=None):
    if os.path.exists(definitions.EXPERIMENT_INDEX):
        experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)
        print(tabulate.tabulate(experiment_index,headers='keys', tablefmt=tablefmt, showindex=False))

    else:
        print("No experiments found.")
