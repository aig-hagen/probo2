import json
import os
import glob

from src.utils import definitions


def init_status_file(tasks, benchmarks, tag, solvers):
    status_dict = {'tag': tag, 'total_tasks': len(tasks), 'finished_tasks': 0, 'tasks': {}}
    for task in tasks:

        status_dict['tasks'][task.symbol] = {}
        status_dict['tasks'][task.symbol]['solvers'] = dict()
        intersection_solvers = set.intersection(set(solvers), set(task.solvers))
        for solver in intersection_solvers:
            total_num_instances = 0
            for benchmark in benchmarks:
                file_count = len(benchmark.get_instances(solver.solver_format))
                total_num_instances += file_count

            status_dict['tasks'][task.symbol]['solvers'][solver.solver_id] = {'name': solver.solver_name, 'version': solver.solver_version,
                                                                        'solved': 0, 'total': total_num_instances}
    if os.path.exists(definitions.STATUS_FILE_DIR):
        os.remove(definitions.STATUS_FILE_DIR)
    with open(definitions.STATUS_FILE_DIR, 'w') as outfile:
        json.dump(status_dict, outfile)


def print_status_summary():
    print(definitions.STATUS_FILE_DIR)
    with open(definitions.STATUS_FILE_DIR) as status_json_file:
        status_data = json.load(status_json_file)
        print("**********STATUS SUMMARY*********")
        print("Tag: ", status_data['tag'])
        print("Tasks finished: {} / {}".format(status_data['finished_tasks'], status_data['total_tasks']))
        print("---------------------------------")
        for task in status_data['tasks'].keys():
            #total_instances = status_data['tasks'][task]['total_instances']
            print("Task: ", task)
            #print("Total instances: ", total_instances)
            print('')
            for solver_id, solver_info in status_data['tasks'][task]['solvers'].items():
                print("Name: {}_{} : {} / {} ".format(solver_info['name'], solver_info['version'],
                                                      solver_info['solved'], solver_info['total']), end='')
                if solver_info['solved'] == solver_info['total']:
                    print("--- FINISHED")
                else:
                    print('')
            print("---------------------------------")


def increment_task_counter():
    with open(definitions.STATUS_FILE_DIR) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['finished_tasks'] += 1

    with open(definitions.STATUS_FILE_DIR, 'w') as outfile:
        json.dump(status_data, outfile)


def increment_instances_counter(task, solver_id):
    with open(definitions.STATUS_FILE_DIR) as status_json_file:
        status_data = json.load(status_json_file)
        status_data['tasks'][task]['solvers'][str(solver_id)]['solved'] += 1
    with open(definitions.STATUS_FILE_DIR, 'w') as outfile:
        json.dump(status_data, outfile)
