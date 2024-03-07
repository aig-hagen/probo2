import json
import os
from src.handler import benchmark_handler, config_handler , solver_handler
from src.utils import definitions


def init_status_file(cfg: config_handler.Config):
    """
    Initializes the status file with the given configuration.

    Args:
        cfg (config_handler.Config): The configuration object.

    Returns:
        None
    """

    name = cfg.name
    tasks = cfg.task
    benchmarks = benchmark_handler.load_benchmark(cfg.benchmark)
    solvers = solver_handler.load_solver(cfg.solver)
    status_dict = {'name': name, 'total_tasks': len(tasks), 'finished_tasks': 0, 'tasks': {}}
    for task in tasks:

        status_dict['tasks'][task] = {}
        status_dict['tasks'][task]['solvers'] = dict()
        for solver in solvers:
            if task in solver['tasks']:
                total_num_instances = 0
                for benchmark in benchmarks:

                    file_count = benchmark_handler.get_instances_count(benchmark['path'],solver['format'][0])
                    total_num_instances += file_count

                status_dict['tasks'][task]['solvers'][solver['id']] = {'name': solver['name'], 'version': solver['version'],
                                                                        'solved': 0, 'total': total_num_instances}
    if os.path.exists(str(definitions.STATUS_FILE_DIR)):
        os.remove(str(definitions.STATUS_FILE_DIR))
    with open(str(definitions.STATUS_FILE_DIR), 'w') as outfile:
        json.dump(status_dict, outfile)


    with open(cfg.status_file_path,'w') as f:
        json.dump(status_dict, f)


import os
import json

def print_status_summary(path=None):
    """
    Prints a summary of the status data.

    Args:
        path (str): Optional. The path to the status JSON file. If not provided, the default status file path is used.

    Raises:
        FileNotFoundError: If the status JSON file does not exist.

    """
    if path is None or not os.path.exists(path):
        with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
            status_data = json.load(status_json_file)
    else:
        with open(path) as status_json_file:
            status_data = json.load(status_json_file)

    print("========== Status Summary ==========")
    print("Tag: ", status_data['name'])
    print("Tasks finished: {} / {}".format(status_data['finished_tasks'], status_data['total_tasks']))
    print("---------------------------------")
    for task in status_data['tasks'].keys():
        print(f'+TASK: {task}')
        print(f" +Solver:")
        for solver_id, solver_info in status_data['tasks'][task]['solvers'].items():
            print("   {}_{} : {} / {} ".format(solver_info['name'], solver_info['version'],
                                                  solver_info['solved'], solver_info['total']), end='')
            if solver_info['solved'] == solver_info['total']:
                print("--- FINISHED")
            else:
                print('')
        print("---------------------------------")


def increment_task_counter():
    """
    Increments the counter for finished tasks in the status file.
    """
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['finished_tasks'] += 1

    with open(str(definitions.STATUS_FILE_DIR), 'w') as outfile:
        json.dump(status_data, outfile)


def increment_instances_counter(config: config_handler.Config, task, solver_id):
    """
    Increments the counter for the number of instances solved by a specific solver for a given task.

    Args:
        config (config_handler.Config): The configuration object.
        task (str): The task for which the solver solved an instance.
        solver_id (int): The ID of the solver.

    Raises:
        FileNotFoundError: If the status file is not found.

    """
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['tasks'][task]['solvers'][str(solver_id)]['solved'] += 1
    with open(config.status_file_path) as _status_json_file:
        _status_data = json.load(_status_json_file)
        _status_data['tasks'][task]['solvers'][str(solver_id)]['solved'] += 1

    with open(str(definitions.STATUS_FILE_DIR), 'w') as outfile:
        json.dump(status_data, outfile)
    with open(config.status_file_path, 'w') as _outfile:
        json.dump(_status_data, _outfile)

def get_total_number_instances_per_solver(path=None):
    """
    Retrieves the total number of instances per solver from the status data.

    Args:
        path (str, optional): The path to the status JSON file. If not provided, the default path is used.

    Returns:
        dict: A dictionary containing the total number of instances per solver.
            The dictionary has the following structure:
            {
                'task1': {
                    'solver1_name_version': {
                        'solved': <number_of_instances_solved>,
                        'total': <total_number_of_instances>
                    },
                    'solver2_name_version': {
                        'solved': <number_of_instances_solved>,
                        'total': <total_number_of_instances>
                    },
                    ...
                },
                'task2': {
                    ...
                },
                ...
            }
    """
    if path is None:
        with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
            status_data = json.load(status_json_file)
    else:
        with open(path) as status_json_file:
            status_data = json.load(status_json_file)

    info = {}
    for task in status_data['tasks'].keys():
        info[task] = {}
        for solver_id, solver_info in status_data['tasks'][task]['solvers'].items():
            info[task][f'{solver_info["name"]}_{solver_info["version"]}'] = {
                'solved': solver_info['solved'],
                'total': solver_info['total']
            }

    return info
