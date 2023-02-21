

import json
import os
import re
import subprocess
import time

from pathlib import Path

import pandas as pd
import tabulate

import src.utils.definitions as definitions
import src.utils.utils as utils

from dataclasses import dataclass
from click import confirm

import logging

logging.basicConfig(filename=str(definitions.LOG_FILE_PATH),format='[%(asctime)s] - [%(levelname)s] : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


@dataclass
class Solver:
    name: str
    version: str
    path: str
    tasks: list
    format: list
    id: int

@dataclass(frozen=False)
class AddSolverOptions():
    name: str
    path: str
    version: str
    fetch: bool
    format: list
    tasks: list
    yes: bool
    no_check: bool


def _parse_fetch_options(parameter: AddSolverOptions):
    try:
        if not parameter.tasks:
            _tasks = fetch_tasks(parameter.path)
        if not parameter.format:
            _format = fetch_format(parameter.path)
        else:
            format = [format]
    except ValueError as e:
            logger.error(f'Error while fetching solver tasks and formats: {e}')
            print(e)
            exit()
    return _tasks,_format


def parse_cli_input(parameter: AddSolverOptions):
    
    if parameter.fetch:
        tasks,format = _parse_fetch_options(parameter)
        
    if not isinstance(format, list):
        format = [format]
    
    solver_info = {'name': parameter.name,'version': parameter.version,
                   'path': parameter.path,'tasks': tasks,'format': format, 'id': None}
    new_solver = Solver(**solver_info)
    if not parameter.no_check:
        is_working = check_interface(new_solver)
    else:
        is_working = True
    
    if not is_working:
        logger.error(f'{new_solver.name}: Solver interface check failed.')
        print('Solver not working.')
        exit()
    if not parameter.yes:
        confirm(
            "Are you sure you want to add this solver?"
            ,abort=True,default=True)
    id = add_solver(new_solver)
    print_summary(new_solver.__dict__)
    print(f"Solver {new_solver.name} added with ID: {new_solver.id}")
    logger.info(f"Solver {new_solver.name} added with ID: {new_solver.id}")
    return new_solver

def _create_solver_obj(parameter: AddSolverOptions):
    if parameter.fetch:
        tasks,format = _parse_fetch_options(parameter)
        
    if not isinstance(format, list):
        format = [format]
    
    solver_info = {'name': parameter.name,'version': parameter.version,
                   'path': parameter.path,'tasks': tasks,'format': format, 'id': None}
    new_solver = Solver(**solver_info)
    return new_solver



def _write_to_empty_solver_file(solver: Solver) -> int:
    with open(definitions.SOLVER_FILE_PATH,'w') as solver_file:
        id = 1
        solver.id = id
        json.dump([solver.__dict__], solver_file,indent=2)
    return id

def _append_to_solver_file(solver: Solver) -> int:
    with open(definitions.SOLVER_FILE_PATH,'r+') as solver_file:
        solver_data = json.load(solver_file)
        _df = pd.DataFrame(solver_data)
        id = int(_df.id.max() + 1)
        solver.id = id
        solver_data.append(solver.__dict__)
        solver_file.seek(0)
        json.dump(solver_data, solver_file,indent=2)
    return id



def add_solver(solver: Solver):
    if not os.path.exists(definitions.SOLVER_FILE_PATH) or (os.stat(definitions.SOLVER_FILE_PATH).st_size == 0):
        return _write_to_empty_solver_file(solver)
    else:
        return _append_to_solver_file(solver)
       


def _add_solver(solver_info: dict):
    if not os.path.exists(definitions.SOLVER_FILE_PATH) or (os.stat(definitions.SOLVER_FILE_PATH).st_size == 0):
        with open(definitions.SOLVER_FILE_PATH,'w') as solver_file:
            id = 1
            solver_info['id'] = id
            json.dump([solver_info], solver_file,indent=2)
        return id
    else:
        with open(definitions.SOLVER_FILE_PATH,'r+') as solver_file:
            solver_data = json.load(solver_file)
            _df = pd.DataFrame(solver_data)
            id = int(_df.id.max() + 1)
            solver_info['id'] = id
            solver_data.append(solver_info)

            solver_file.seek(0)
            json.dump(solver_data, solver_file,indent=2)
        return id

def _init_cmd_params(solver_path):
    if solver_path.endswith('.sh'):
        return ['bash']
    elif solver_path.endswith('.py'):
        return ['python3']
    return []

def fetch(solver_path, to_fetch):
        cmd_params = _init_cmd_params(solver_path)
        cmd_params.append(solver_path)
        cmd_params.append(f"--{to_fetch}")
        solver_dir = os.path.dirname(solver_path)
        try:
            solver_output = utils.run_process(cmd_params,
                                        capture_output=True, check=True,cwd=solver_dir)
            solver_output = re.sub("\s+", " ",
                                solver_output.stdout.decode("utf-8")).strip(" ")

            solver_property = solver_output[solver_output.find("[") + 1:solver_output.find("]")].split(",")
        except subprocess.CalledProcessError as err:
                logger.error(f'Error code: {err.returncode} stdout: {err.output.decode("utf-8")} stderr: {err.stderr.decode("utf-8")}')
                print("Error code: {}\nstdout: {}\nstderr:{}\n".format(err.returncode, err.output.decode("utf-8"), err.stderr.decode("utf-8")))
                exit()
        return solver_property

def fetch_tasks(solver_path)->list:
        """Calls solver with "problems" options.
        Args:
            tasks (list, optional): List of tasks the solver might support. Defaults to None.

        Returns:
            list: List of tasks the solver acutally supports.
        """
        supported_tasks = sorted([ x.strip(" ") for x in fetch(solver_path, "problems")])
        return supported_tasks

def fetch_format(solver_path):
        supported_formats = fetch(solver_path, "formats")
        return supported_formats



def check_interface(solver: Solver) -> bool:
        init_cmd = _init_cmd_params(solver.path)

        solver_format = solver.format[0]
        if 'apx' in solver_format:
            instance = str(definitions.TEST_INSTANCE_APX)
        elif 'tgf' in solver_format:
            instance = str(definitions.TEST_INSTANCE_TGF)



        for test_task in solver.tasks:
            cmd_params = init_cmd.copy()
            cmd_params.extend([solver.path,
                    "-p", test_task,
                    "-f", instance,
                    "-fo", solver_format])

            if 'DS' in test_task or 'DC' in test_task:
                with open(str(definitions.TEST_INSTANCE_ARG),'r') as arg_file:
                    arg = arg_file.read().rstrip('\n')
                    cmd_params.extend(["-a",arg])

            if test_task.endswith('-D'):
                dynamic_file = None
                if 'apx' in solver_format:
                    dynamic_file = str(definitions.TEST_INSTANCE_APXM)
                elif 'tgf' in solver_format:
                    dynamic_file = str(definitions.TEST_INSTANCE_TGFM)
                cmd_params.extend([ "-m", dynamic_file])



            try:
                result = utils.run_process(cmd_params,
                                        capture_output=True, timeout=5, check=True)

            except subprocess.TimeoutExpired as e:
                print(f'Solver interface test timed out.')
                logger.error(f'Solver interface test timed out.')
                return False
            except subprocess.CalledProcessError as err:
                print(f'Something went wrong when testing the interface: {err}\nOutput:{err.output}')
                logger.error(f'Something went wrong when testing the interface: {err}\nOutput:{err.output}')
                return False

        return True


def print_summary(solver_info: dict):
    print()
    print("**********SOLVER SUMMARY**********")
    for key,value in solver_info.items():
        if key == 'tasks':
            print(f"Tasks: {json.dumps(solver_info['tasks'],indent=4)}" )
        elif key == 'format':
            print(f"Format: {json.dumps(solver_info['format'],indent=4)}" )
        else:
            print(f'{str(key).capitalize()}: {value}')
    #print(json.dumps(solver_info, indent=4))
    print()

def print_solvers(extra_columns=None, tablefmt=None):
    columns = ['id','name','version','format']
    if extra_columns:
        columns.extend(extra_columns)
    if tablefmt is None:
        tablefmt = 'pretty'
    if os.path.exists(definitions.SOLVER_FILE_PATH) and os.stat(definitions.SOLVER_FILE_PATH).st_size != 0:
        solvers_df = pd.read_json(definitions.SOLVER_FILE_PATH)
        print(tabulate.tabulate(solvers_df[columns],headers='keys', tablefmt=tablefmt, showindex=False))
    else:
        print("No solvers found.")

def load_solver(identifiers, as_class=False):
    if isinstance(identifiers,str):
        if identifiers == 'all':
            return load_solver_json()
    elif isinstance(identifiers,list):
        if 'all' in identifiers:
            return load_solver_json()
        else:
            return load_solver_by_identifier(identifiers)
    else:
        print(f'Unable to load solvers: {identifiers}')
        logger.error(f'Unable to load solvers: {identifiers}')
        exit()


def load_solver_json():
    with open(definitions.SOLVER_FILE_PATH,'r+') as solver_file:
            solver_data = json.load(solver_file)
    return solver_data

def load_solver_dataframe():
    return pd.read_json(definitions.SOLVER_FILE_PATH)


def load_solver_by_task(task:str)-> list:
    """Load solver by task

    Args:
        task (str): _description_

    Returns:
        list: _description_
    """
    solver_json = load_solver_json()
    support = []
    for solver in solver_json:
        if task in solver['tasks']:
            support.append(solver)
    return support

def load_solver_by_identifier(identifier: list) -> list:
    """Load solvers by name or id

    Args:
        identifier (list): _description_

    Returns:
        list: _description_
    """
    solver_json = load_solver_json()
    solver_list = []


    for solver in solver_json:
        if solver['name'] in identifier:
            solver_list.append(solver)
        elif solver['id'] in identifier or str(solver['id']) in identifier:
            solver_list.append(solver)
    return solver_list

def _update_solver_json_after_delete(solvers: list):
    json_str = json.dumps(solvers, indent=2)

    with open(definitions.SOLVER_FILE_PATH, "w") as f:
        f.write(json_str)

def delete_solver(id) -> bool:
    solvers = load_solver('all')
    deleted = False
    if isinstance(id,str):
        if id.isdigit():
            id = int(id)
    for solver in solvers:
        if solver['id'] == id or solver['name'] == id:
            deleted = True
            solvers.remove(solver)
    if not deleted:
        print(f"Unable to delete solver {id}. Solver not found.")
        logger.error(f"Unable to delete solver {id}. Solver not found.")
        return False
    
    _update_solver_json_after_delete(solvers)
    print(f"Solver {id} deleted")
    logger.info(f"Solver {id} deleted")
    return True




def delete_all_solvers():
    with open(definitions.SOLVER_FILE_PATH,"w") as f:
        f.write("")

def _init_results_dict(solver_info) -> dict:
    results = {}
    for key in solver_info.keys():
        results[f'solver_{key}'] = solver_info[key]
    del results['solver_tasks']
    del results['solver_format']
    return results

@dataclass
class SolverParameters:
    instance_name: str
    repetition: int
    solver_dir: str
    task: str
    additional_argument: str
    timeout: str
    final_params: list
    format: list

def _set_solver_parameters(solver_info,instance,task,format,additional_arguments_lookup,dynamic_files_lookup,repetition,timeout) -> SolverParameters:
    cmd_params = []
    additional_argument = ""
    if solver_info['path'].endswith('.sh'):
        cmd_params.append('bash')
    elif solver_info['path'].endswith('.py'):
        cmd_params.append('python')
    
    instance_name = Path(instance).stem
    params = [solver_info['path'],
          "-p", task,
          "-f", instance,
          "-fo", format]
    if additional_arguments_lookup:
        additional_argument = additional_arguments_lookup[instance_name]
        params.extend(["-a",additional_argument])
    if dynamic_files_lookup:
        dynamic_file = dynamic_files_lookup[format][instance]
        params.extend([ "-m", dynamic_file])
    
    solver_dir = os.path.dirname(solver_info['path'])
    final_params = cmd_params + params

    solver_parameter = SolverParameters(instance_name=instance_name,
                                        repetition=repetition,
                                        solver_dir=solver_dir,
                                        task=task,
                                        additional_argument=additional_argument,
                                        timeout=timeout,
                                        final_params=final_params,
                                        format=format)
    
    return solver_parameter
    
    



def _write_results_to_file(output_file_dir,solver_parameters:SolverParameters):

    if output_file_dir is None:
        logger.error(f"Output destination not found:{output_file_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Output destination not found")

    if not os.path.exists(solver_parameters.solver_dir):
        logger.error(f"Solver path not found:{solver_parameters.solver_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Solver path not found.")

    out_file_path = os.path.join(output_file_dir,f'{solver_parameters.instance_name}_{solver_parameters.repetition}.out')
    with open(out_file_path,'w') as output:
        start_time_current_run = time.perf_counter()
        utils.run_process(solver_parameters.final_params,
                        timeout=solver_parameters.timeout, check=True,cwd=solver_parameters.solver_dir,stdout=output)
        end_time_current_run = time.perf_counter()
        run_time = end_time_current_run - start_time_current_run
    return {'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':False,'additional_argument': solver_parameters.additional_argument, 'runtime': run_time, 'result': out_file_path, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':solver_parameters.timeout}

def _run_solver_with_parameters(solver_parameters: SolverParameters):
    if not os.path.exists(solver_parameters.solver_dir):
        logger.error(f"Solver path not found:{solver_parameters.solver_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Solver path not found.")
    
    start_time_current_run = time.perf_counter()
    utils.run_process(solver_parameters.final_params,
                    capture_output=True, timeout=solver_parameters.timeout, check=True,cwd=solver_parameters.solver_dir)
    end_time_current_run = time.perf_counter()
    run_time = end_time_current_run - start_time_current_run
    return {'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':False,'additional_argument': solver_parameters.additional_argument, 'runtime': run_time, 'result': None, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':solver_parameters.timeout}
            

def run_solver(solver_info,task,timeout,instance,format,additional_arguments_lookup=None,dynamic_files_lookup=None,output_file_dir=None,additional_solver_arguments=None, repetition=None):
  
    results = _init_results_dict(solver_info)
    
    solver_parameters = _set_solver_parameters(solver_info,instance,task,format,additional_arguments_lookup,dynamic_files_lookup,repetition,timeout)
 
    
    try:
        if output_file_dir is not None:
           results.update(_write_results_to_file(output_file_dir,solver_parameters))
        else:
            results.update(_run_solver_with_parameters(solver_parameters))
        return results
    except subprocess.TimeoutExpired as e:
        logger.error(f'Solver {solver_info.get("solver_name")} timed out on instance {solver_parameters.instance_name}')
        results.update({'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':True,'additional_argument': solver_parameters.additional_argument, 'runtime': solver_parameters.timeout, 'result': None, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':solver_parameters.timeout})
        return results
    except subprocess.CalledProcessError as err:
        logger.error(f'Something went wrong running solver {solver_info.get("solver_name")}: {err}')
        print("\nError occured:",err)
        results.update({'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':False,'additional_argument': solver_parameters.additional_argument, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode,'error': err,'cut_off':solver_parameters.timeout})
        return results
