

import json
import os
import re
import subprocess
import time
import io

from pathlib import Path

import pandas as pd
import tabulate

import src.utils.definitions as definitions
import src.utils.utils as utils

from dataclasses import dataclass
from click import confirm

import logging
from tqdm import tqdm
from src.utils.options.CommandOptions import AddSolverOptions, EditSolverOptions

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




def _parse_fetch_options(parameter: AddSolverOptions):
    try:
        if not parameter.tasks:
            _tasks = fetch_tasks(parameter.path)
        else:
            _tasks = parameter.tasks
        if not parameter.format:
            _format = fetch_format(parameter.path)
        else:
            _format = parameter.format

    except ValueError as e:
            logger.error(f'Error while fetching solver tasks and formats: {e}')
            print(e)
            exit()
    return _tasks,_format


def add_solver(options: AddSolverOptions):

    if options.fetch:
        options.tasks,options.format = _parse_fetch_options(options)

    if not isinstance(options.format, list):
        options.format = [options.format]

    solver_info = {'name': options.name,'version': options.version,
                   'path': options.path,'tasks': options.tasks,'format': options.format, 'id': None}
    new_solver = Solver(**solver_info)
    if not options.no_check:
        is_working = check_interface(new_solver)
    else:
        is_working = True

    if not is_working:
        logger.error(f'{new_solver.name}: Solver interface check failed.')
        print('Solver not working properly.')
        confirm(
            "Do you want to add this solver anyways?"
            ,abort=True,default=True)
        id = _add_solver_to_database(new_solver)
        print(f"Solver {new_solver.name} added with ID: {new_solver.id}")
        logger.info(f"Solver {new_solver.name} added with ID: {new_solver.id}")
        return new_solver

    print_summary(new_solver.__dict__)
    if not options.yes:
        confirm(
            "Are you sure you want to add this solver?"
            ,abort=True,default=True)
    id = _add_solver_to_database(new_solver)
    print(f"Solver {new_solver.name} added with ID: {new_solver.id}")
    logger.info(f"Solver {new_solver.name} added with ID: {new_solver.id}")
    return new_solver

def edit_solver(options: EditSolverOptions):
    solver_infos = load_solver_by_identifier(options.id)
    if not solver_infos or solver_infos is None:
        print(f'No solver with id {options.id} found. Please run solvers command to get a list of solvers in database.')
        exit()
    solver_infos = solver_infos[0] # load_solver returns a list,
    print('Changing values:')
    for attribute, value in options.__dict__.items():
        if attribute in solver_infos.keys() and value is not None:
            if attribute != 'id':
                print(f'+ {attribute}: {solver_infos[attribute]} -> {value} ')
            solver_infos[attribute] = value
    confirm('Apply changes?',abort=True,default=True)
    update_solver(solver_infos)
    print(f'Updated solver!')






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



def _add_solver_to_database(solver: Solver):
    if not os.path.exists(definitions.SOLVER_FILE_PATH) or (os.stat(definitions.SOLVER_FILE_PATH).st_size == 0):
        return _write_to_empty_solver_file(solver)
    else:
        return _append_to_solver_file(solver)




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
                print('Somethin went wrong fetching informations from solver:')
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



test_instance_paths = { 'apx': definitions.TEST_INSTANCE_APX,
                       'tgf': definitions.TEST_INSTANCE_TGF,
                       'i23': definitions.TEST_INSTANCE_I23}

def check_interface(solver: Solver) -> bool:
    init_cmd = _init_cmd_params(solver.path)

    for solver_format in solver.format:
        print(f'Testing interface of {solver.name} for format {solver_format}.')
        #solver_format = solver.format[0]
        if "23" in solver_format:
            instance = test_instance_paths['i23']
        else:
            instance = test_instance_paths[solver_format]
        for test_task in tqdm(solver.tasks,desc='Testing supported tasks'):
            cmd_params = init_cmd.copy()

            if solver_format == 'i23' or '23' in solver_format:
                cmd_params.extend([solver.path,
                    "-p", test_task,
                    "-f", instance])
            else:
                cmd_params.extend([solver.path,
                    "-p", test_task,
                    "-f", instance,
                    "-fo", solver_format])


            if 'DS' in test_task or 'DC' in test_task:
                if solver_format == 'i23' or '23' in solver_format:
                    arg_path = definitions.TEST_INSTANCE_ARG_I23
                else:
                    arg_path = definitions.TEST_INSTANCE_ARG
                with open(str(arg_path),'r') as arg_file:
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

def load_solver_by_identifier(identifier) -> list:
    """Load solvers by name or id

    Args:
        identifier (list): _description_

    Returns:
        list: _description_
    """
    solver_json = load_solver_json()
    solver_list = []

    if not isinstance(identifier,list):
        identifier = [identifier]


    for solver in solver_json:
        if solver['name'] in identifier:
            solver_list.append(solver)
        elif solver['id'] in identifier or str(solver['id']) in identifier:
            solver_list.append(solver)
    return solver_list

def _update_solver_json(solvers: list):
    json_str = json.dumps(solvers, indent=2)
    with open(definitions.SOLVER_FILE_PATH, "w") as f:
        f.write(json_str)

def update_solver(solver_infos):
    solvers = load_solver('all')
    for i,solver in enumerate(solvers):
        if solver['id'] == solver_infos['id']:
            solvers[i] = solver_infos
            break

    _update_solver_json(solvers)



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
            break
    if not deleted:
        print(f"Unable to delete solver {id}. Solver not found.")
        logger.error(f"Unable to delete solver {id}. Solver not found.")
        return False

    _update_solver_json(solvers)
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
    time_measurement: str

def _set_solver_parameters(solver_info,instance,task,format,additional_arguments_lookup,dynamic_files_lookup,repetition,timeout,time_measurement='default',interface_mode=None) -> SolverParameters:
    cmd_params = []
    additional_argument = ""
    if time_measurement == 'default':
        cmd_params.append('time')
    if solver_info['path'].endswith('.sh'):
        cmd_params.append('bash')
    elif solver_info['path'].endswith('.py'):
        cmd_params.append('python')

    instance_name = Path(instance).stem

    if interface_mode == 'i23':
        params = [solver_info['path'],
          "-p", task,
          "-f", instance]
    else:
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
                                        format=format,
                                        time_measurement=time_measurement)

    return solver_parameter





def _run_solver_write_results_to_file(output_file_dir,solver_parameters:SolverParameters):

    if output_file_dir is None:
        logger.error(f"Output destination not found:{output_file_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Output destination not found")

    if not os.path.exists(solver_parameters.solver_dir):
        logger.error(f"Solver path not found:{solver_parameters.solver_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Solver path not found.")

    out_file_path = os.path.join(output_file_dir,f'{solver_parameters.instance_name}_{solver_parameters.repetition}.out')
    with open(out_file_path,'w') as output:
        out_time = open('temp.time','w+')
        start_time_current_run = time.perf_counter()
        utils.run_process(solver_parameters.final_params,
                        timeout=solver_parameters.timeout, check=True,cwd=solver_parameters.solver_dir,stdout=output,stderr=out_time)
        end_time_current_run = time.perf_counter()
        out_time.close()
        if solver_parameters.time_measurement == 'default':
            out_time = open('temp.time','r')
            out_str = out_time.read()
            out_time.close()
            os.remove("temp.time")
            # out_str = out_time.stdout.decode("utf-8")
            time_list = out_str.split(" ")
            if not 'user' in time_list[0] and not 'system' in time_list[1]:

                raise subprocess.CalledProcessError(-1,None,"Time could not be measured.")
            time_user = float(time_list[0].split('user')[0])
            time_system = float(time_list[1].split('system')[0])
            run_time = time_user + time_system
        else:
            run_time = end_time_current_run - start_time_current_run
    return {'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':False,'additional_argument': solver_parameters.additional_argument, 'runtime': run_time, 'result': out_file_path, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':solver_parameters.timeout}

def _run_solver(solver_parameters: SolverParameters):
    if not os.path.exists(solver_parameters.solver_dir):
        logger.error(f"Solver path not found:{solver_parameters.solver_dir}")
        raise subprocess.CalledProcessError(returncode=-1,cmd=solver_parameters.final_params,output="Solver path not found.")

    start_time_current_run = time.perf_counter()
    out = utils.run_process(solver_parameters.final_params,
                    capture_output=True, timeout=solver_parameters.timeout, check=True,cwd=solver_parameters.solver_dir)
    end_time_current_run = time.perf_counter()
    if solver_parameters.time_measurement == 'default':
        time_list = out.stderr.decode("utf-8").split(" ")
        time_user = float(time_list[0].split('user')[0])
        time_system = float(time_list[1].split('system')[0])
        run_time = time_user + time_system
    else:
        run_time = end_time_current_run - start_time_current_run
    return {'instance': solver_parameters.instance_name,'format':solver_parameters.format,'task': solver_parameters.task,'timed_out':False,'additional_argument': solver_parameters.additional_argument, 'runtime': run_time, 'result': None, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off':solver_parameters.timeout}


def run_solver(solver_info,task,timeout,instance,format,additional_arguments_lookup=None,dynamic_files_lookup=None,output_file_dir=None,additional_solver_arguments=None, repetition=None):

    results = _init_results_dict(solver_info)

    if  format == 'i23' or '23' in format:
        interface_mode = 'i23'
    else:
        interface_mode = None

    solver_parameters = _set_solver_parameters(solver_info,instance,task,format,additional_arguments_lookup,dynamic_files_lookup,repetition,timeout,interface_mode=interface_mode)


    try:
        if output_file_dir is not None:
           results.update(_run_solver_write_results_to_file(output_file_dir,solver_parameters))
        else:
            results.update(_run_solver(solver_parameters))
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


def run_solver_accaptence(solver,task,instance,arg,timeout,format):
    cmd_params = []
    if solver['path'].endswith('.sh'):
        cmd_params.append('bash')
    elif solver['path'].endswith('.py'):
        cmd_params.append('python')

    instance_name = Path(instance).stem
    params = [solver['path'],
          "-p", task,
          "-f", instance,
          "-fo", format,
          "-a", arg]

    solver_dir = os.path.dirname(solver['path'])
    final_params = cmd_params + params
    try:
        start_time_current_run = time.perf_counter()
        output = utils.run_process(final_params,
                        capture_output=True, timeout=timeout, check=True,cwd=solver_dir)
        end_time_current_run = time.perf_counter()
        run_time = end_time_current_run - start_time_current_run
        result = output.stdout.decode("utf-8")

        accepted = True if 'YES' in result else False

        return {'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': arg, 'runtime': run_time, 'result': result,'accepted': accepted, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off': timeout}
    except subprocess.TimeoutExpired as e:
        #logger.error(f'Solver {solver_info.get("solver_name")} timed out on instance {solver_parameters.instance_name}')
        return {'instance': instance_name,'format':format,'task': task,'timed_out':True,'additional_argument': arg, 'runtime': timeout, 'result': result,'accepted': accepted, 'exit_with_error': False, 'error_code': None,'error': None,'cut_off': timeout}

    except subprocess.CalledProcessError as err:
        logger.error(f'Something went wrong running solver {solver.get("solver_name")}: {err}')
        print("\nError occured:",err)
        return {'instance': instance_name,'format':format,'task': task,'timed_out':False,'additional_argument': arg, 'runtime': None, 'result': result,'accepted': accepted, 'exit_with_error': True, 'error_code': err.returncode,'error': err,'cut_off':timeout}


def dry_run(solver,task,instance,arg,format,timeout):
    cmd_params = []
    if solver['path'].endswith('.sh'):
        cmd_params.append('bash')
    elif solver['path'].endswith('.py'):
        cmd_params.append('python')

    if arg:
        params = [solver['path'],
          "-p", task,
          "-f", instance,
          "-fo", format,
          "-a", arg]
    else:
        params = [solver['path'],
          "-p", task,
          "-f", instance,
          "-fo", format]

    solver_dir = os.path.dirname(solver['path'])
    final_params = cmd_params + params
    print(f'Running solver {solver["name"]}:')
    try:
        start_time_current_run = time.perf_counter()
        output = utils.run_process(final_params,
                        capture_output=True, timeout=timeout, check=True,cwd=solver_dir)
        end_time_current_run = time.perf_counter()
        run_time = end_time_current_run - start_time_current_run
        result = output.stdout.decode("utf-8")
        print(f'Runtime: {run_time}\nResult:{result}')

    except subprocess.TimeoutExpired as e:
       print(f'Solver {solver["name"]} timeout.')

    except subprocess.CalledProcessError as err:
        print("\nError occured:",err)

