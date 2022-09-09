

import json
from sys import stdout
import tabulate



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
    if os.path.exists(definitions.SOLVER_FILE_PATH) and os.stat(definitions.SOLVER_FILE_PATH).st_size != 0:
        solvers_df = pd.read_json(definitions.SOLVER_FILE_PATH)
        print(tabulate.tabulate(solvers_df[columns],headers='keys', tablefmt=tablefmt, showindex=False))
    else:
        print("No solvers found.")

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
    if isinstance(identifiers,str):
        if identifiers == 'all':
            return load_solver_json()
    elif isinstance(identifiers,list):
        if 'all' in identifiers:
            return load_solver_json()
        else:
            return load_solver_by_identifier(identifiers)
    else:
        print('Unable to load solvers.')
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
        if solver['name'] in identifier or solver['id'] in identifier:
            solver_list.append(solver)
    return solver_list

def _update_solver_json(solvers: list):
    json_str = json.dumps(solvers, indent=2)

    with open(definitions.SOLVER_FILE_PATH, "w") as f:
        f.write(json_str)

def delete_solver(id):
    solvers = load_solver('all')
    deleted = False
    if id.isdigit():
        id = int(id)
    for solver in solvers:
        if solver['id'] == id or solver['name'] == id:
            deleted = True
            solvers.remove(solver)
    if deleted:
        _update_solver_json(solvers)
        print(f"Solver {id} deleted")
    else:
        print("Solver not found.")




def delete_all_solvers():
    with open(definitions.SOLVER_FILE_PATH,"w") as f:
        f.write("")


def run_solver(solver_info,task,timeout,instance,format,additional_arguments_lookup=None,dynamic_files_lookup=None,output_file_dir=None,additional_solver_arguments=None, repetition=None):

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
            dynamic_file = dynamic_files_lookup[format][instance]
            params.extend([ "-m", dynamic_file])


        final_param = cmd_params + params
        try:
            if output_file_dir is not None:
                out_file_path = os.path.join(output_file_dir,f'{instance_name}_{repetition}.out')
                with open(out_file_path,'w') as output:
                    start_time_current_run = time.perf_counter()
                    if os.path.exists(solver_dir):
                        utils.run_process(final_param,
                                        timeout=timeout, check=True,cwd=solver_dir,stdout=output)
                        end_time_current_run = time.perf_counter()
                        run_time = end_time_current_run - start_time_current_run
                        results.update({'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': additional_argument, 'runtime': run_time, 'result': out_file_path, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':timeout})
                    else:
                        raise subprocess.CalledProcessError(returncode=-1,cmd=final_param,output="Solver path not found.")

            else:
                if os.path.exists(solver_dir):
                    start_time_current_run = time.perf_counter()
                    result = utils.run_process(final_param,
                                    capture_output=True, timeout=timeout, check=True,cwd=solver_dir)
                    end_time_current_run = time.perf_counter()
                    run_time = end_time_current_run - start_time_current_run
                    # solver_output = re.sub("\s+", "",
                    #                 result.stdout.decode("utf-8"))
                    results.update({'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': additional_argument, 'runtime': run_time, 'result': None, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':timeout})
                else:
                    raise subprocess.CalledProcessError(returncode=-1,cmd=final_param,output="Solver path not found.")
            return results
        except subprocess.TimeoutExpired as e:
            results.update({'instance': instance_name,'format':format,'task': task,'timed_out':True,'additional_argument': additional_argument, 'runtime': timeout, 'result': None, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':timeout})
            return results
        except subprocess.CalledProcessError as err:
            #logging.exception(f'Something went wrong running solver {self.solver_name}')
            print("\nError occured:",err)
            results.update({'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': additional_argument, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode,'error': err,'cut_off':timeout})
            return results