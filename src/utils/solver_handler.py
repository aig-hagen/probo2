import cmd
from copy import copy
import json
import tabulate
from src import solver


import src.utils.definitions as definitions
import os
import pandas as pd
import src.utils.utils as utils
import subprocess
import re
from json import JSONEncoder
import time
from pathlib import Path

def add_solver(solver_info: dict):
    if not os.path.exists(definitions.SOLVER_FILE_PATH):
        with open(definitions.SOLVER_FILE_PATH,'w') as solver_file:
            id = 1
            solver_info['id'] = id
            json.dump([solver_info], solver_file)
        return id
    else:
        with open(definitions.SOLVER_FILE_PATH,'r+') as solver_file:
            solver_data = json.load(solver_file)
            id = len(solver_data) + 1
            solver_info['id'] = id
            solver_data.append(solver_info)

            solver_file.seek(0)
            json.dump(solver_data, solver_file)
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



def check_interface(solver_info) -> bool:
        init_cmd = _init_cmd_params(solver_info['path'])
        solver_format = solver_info['format'][0]
        print(solver_info)
        if 'apx' in solver_format:
            instance = str(definitions.TEST_INSTANCE_APX)
        elif 'tgf' in solver_format:
            instance = str(definitions.TEST_INSTANCE_TGF)



        for test_task in solver_info['tasks']:
            cmd_params = init_cmd.copy()
            cmd_params.extend([solver_info['path'],
                    "-p", test_task,
                    "-f", instance,
                    "-fo", solver_format])

            if 'DS' in test_task or 'DC' in test_task:
                with open(str(definitions.TEST_INSTANCE_ARG),'r') as arg_file:
                    arg = arg_file.read().rstrip('\n')
                    cmd_params.extend(["-a",arg])

            try:
                result = utils.run_process(cmd_params,
                                        capture_output=True, timeout=5, check=True)
            except subprocess.TimeoutExpired as e:
                print(f'Solver interface test timed out.')
                return False
            except subprocess.CalledProcessError as err:
                print(f'Something went wrong when testing the interface: {err}\nOutput:{err.output}')
                return False

        return True


def print_summary(solver_info):
    print("**********SOLVER SUMMARY**********")
    print(json.dumps(solver_info, indent=4))

def print_solvers(extra_columns=None, tablefmt=None):
    columns = ['id','name','version','format']
    if extra_columns:
        columns.extend(extra_columns)
    if tablefmt is None:
        tablefmt = 'pretty'

    solvers_df = pd.read_json(definitions.SOLVER_FILE_PATH)
    print(tabulate.tabulate(solvers_df[columns],headers='keys', tablefmt=tablefmt, showindex=False))

class Solver(object):
    def __init__(self, name, version, path, format, tasks, id):
        self.name = name
        self.version =version
        self.path  = path
        self.format = format
        self.tasks = tasks
        self.id = id

    def to_json(self):
        return json.dumps(self,cls=SolverEncoder, indent=4)

    def run(self,task,timeout,instances, additional_arguments_lookup=None,dynamic_files_lookup=None,additional_solver_arguments=None):
        results = {}
        cmd_params = []
        additional_argument = ""
        solver_dir = os.path.dirname(self.solver_path)
        if self.path.endswith('.sh'):
            cmd_params.append('bash')
        elif self.path.endswith('.py'):
            cmd_params.append('python')

        for instance in instances:
            instance_name = Path(instance).stem
            params = [self.solver_path,
                  "-p", task.symbol,
                  "-f", instance,
                  "-fo", self.solver_format]
            if additional_arguments_lookup:
                additional_argument = additional_arguments_lookup[instance_name]
                params.extend(["-a",additional_argument])
            final_param = cmd_params + params
            try:
                start_time_current_run = time.perf_counter()
                result = utils.run_process(final_param,
                                   capture_output=True, timeout=timeout, check=True,cwd=solver_dir)
                end_time_current_run = time.perf_counter()
                run_time = end_time_current_run - start_time_current_run
                solver_output = re.sub("\s+", "",
                                   result.stdout.decode("utf-8"))
                results[instance_name] = {'timed_out':False,'additional_argument': additional_argument, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None}
            except subprocess.TimeoutExpired as e:
                results[instance_name] = {'timed_out':True,'additional_argument': additional_argument, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None}
            except subprocess.CalledProcessError as err:
                #logging.exception(f'Something went wrong running solver {self.solver_name}')
                print("\nError occured:",err)
                results[instance_name] = {'timed_out':False,'additional_argument': additional_argument, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode}





class SolverEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def load_solver(identifiers):
    if identifiers == 'all':
        return load_solver_json()
    else:
        return load_solver_by_identifier(identifiers)

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
        if solver['name'] in identifier or solver['id'] in identifier:
            solver_list.append(solver)
    return solver_list


def run_solver(solver_info,task,timeout,instance,format,additional_arguments_lookup=None,dynamic_files_lookup=None,additional_solver_arguments=None):

        cmd_params = []
        additional_argument = ""
        solver_dir = os.path.dirname(solver_info['path'])
        if solver_info['path'].endswith('.sh'):
            cmd_params.append('bash')
        elif solver_info['path'].endswith('.py'):
            cmd_params.append('python')

        results = {}
        for key in solver_info.keys():
            results[f'solver_{key}'] = solver_info[key]
        del results['solver_tasks']
        del results['solver_format']


        instance_name = Path(instance).stem
        params = [solver_info['path'],
              "-p", task,
              "-f", instance,
              "-fo", format]
        if additional_arguments_lookup:
            additional_argument = additional_arguments_lookup[instance_name]
            params.extend(["-a",additional_argument])
        if dynamic_files_lookup:
            pass

        final_param = cmd_params + params
        try:
            start_time_current_run = time.perf_counter()
            result = utils.run_process(final_param,
                               capture_output=True, timeout=timeout, check=True,cwd=solver_dir)
            end_time_current_run = time.perf_counter()
            run_time = end_time_current_run - start_time_current_run
            solver_output = re.sub("\s+", "",
                               result.stdout.decode("utf-8"))
            results.update({'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': additional_argument, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None})
            return results
        except subprocess.TimeoutExpired as e:
            results.update({'instance': instance_name,'format':format,'task': task,'timed_out':True,'additional_argument': additional_argument, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None})
            return results
        except subprocess.CalledProcessError as err:
            #logging.exception(f'Something went wrong running solver {self.solver_name}')
            print("\nError occured:",err)
            results.update({'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': additional_argument, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode})
            return results