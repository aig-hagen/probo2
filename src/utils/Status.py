import json
import os
from src.utils import benchmark_handler, definitions, solver_handler
from src.utils import config_handler
from src.utils import solver_handler

def init_status_file(cfg: config_handler.Config):
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


def print_status_summary():
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
        print("========== Satus Summary ==========")
        print("Tag: ", status_data['name'])
        print("Tasks finished: {} / {}".format(status_data['finished_tasks'], status_data['total_tasks']))
        print("---------------------------------")
        for task in status_data['tasks'].keys():
            #total_instances = status_data['tasks'][task]['total_instances']
            print(f'+TASK: {task}')
            print(f" +Solver:")
            #print("Total instances: ", total_instances)
            for solver_id, solver_info in status_data['tasks'][task]['solvers'].items():
                print("   {}_{} : {} / {} ".format(solver_info['name'], solver_info['version'],
                                                      solver_info['solved'], solver_info['total']), end='')
                if solver_info['solved'] == solver_info['total']:
                    print("--- FINISHED")
                else:
                    print('')
            print("---------------------------------")


def increment_task_counter():
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['finished_tasks'] += 1

    with open(str(definitions.STATUS_FILE_DIR), 'w') as outfile:
        json.dump(status_data, outfile)


def increment_instances_counter(task, solver_id):
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['tasks'][task]['solvers'][str(solver_id)]['solved'] += 1
    with open(str(definitions.STATUS_FILE_DIR), 'w') as outfile:
        json.dump(status_data, outfile)
