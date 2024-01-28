"""Contains all utility functions for the benchmark generators"""
import json
from math import floor
import random

from src.utils.utils import dispatch_on_value

import os
from pathlib import Path
from click import progressbar
import numpy as np
import tempfile
import subprocess
import time
from networkx import DiGraph
#from progressbar import progressbar
def solve(solver,task,instance_content,timeout):


        cmd = [
            solver.solver_path,
            "-fo",
            solver.solver_format,
            "-p",
            task.symbol,
            "-f",
        ]

        # write APX file to RAM
        with tempfile.NamedTemporaryFile(mode="w+") as tmp_input:
            tmp_input.write(instance_content)
            tmp_input.seek(0)
            cmd.append(tmp_input.name)

            # write output file to RAM and solve
            with tempfile.TemporaryFile(mode="w+") as tmp_output:
                try:
                    start_time_current_run = time.perf_counter()
                    p = subprocess.run(cmd, stdout=tmp_output,timeout=timeout,stderr=subprocess.PIPE)
                    end_time_current_run = time.perf_counter()
                    run_time = end_time_current_run - start_time_current_run
                    print(run_time)
                except subprocess.TimeoutExpired as e:
                    return timeout + 1
                if p.stderr:
                    raise Exception(p.stderr)
                tmp_output.seek(0)

                res = tmp_output.read()
                print(f'{res=}')
        return run_time
def _ensure_runtime(graphs, configs, solver, task):
    timeout = float(configs['general']['timeout'])
    timemin = float(configs['general']['timemin'])
    accpeted_graphs = []
    too_hard = 0
    too_easy = 0
    for graph in graphs:
        instance_content = parse_graph(solver.solver_format, graph)
        runtime = solve(solver,task,instance_content,timeout)
        if timemin<= runtime <= timeout:

            accpeted_graphs.append(graph)
            print('Instance added')
            print(len(accpeted_graphs))
        elif runtime < timemin:
            too_easy += 1
        elif runtime > timeout:
            too_hard += 1
    return accpeted_graphs, too_easy,too_hard

def _ensure_runtime_single_instance(graph, configs, solver, task):
    timeout = float(configs['general']['timeout'])
    timemin = float(configs['general']['timemin'])
    accpeted_graphs = []

    instance_content = parse_graph(solver.solver_format, graph)
    runtime = solve(solver,task,instance_content,timeout)
    if timemin<= runtime <= timeout:
            return True
    else:
           return False

def get_batch_size(num, batch_size):
    if num % batch_size == 0:
        return [batch_size] * int((num / batch_size))
    else:
        remainder = num % batch_size
        sizes = [batch_size] * int(floor(num / batch_size))
        sizes.append(remainder)
        return sizes

@dispatch_on_value
def _modify_configs(generator,configs,type):
    pass



def tweak_configs(generator,configs,too_easy,too_hard):
    timeout = float(configs['general']['timeout'])
    timemin = float(configs['general']['timemin'])
    if too_easy >= too_hard:
        _modify_configs(generator,configs,'easy')
    else:
        _modify_configs(generator,configs,'hard')



@_modify_configs.register('scc')
def _modify_configs_scc(generator,configs,type):
    step_size_max_scc = configs['scc']['step_size_max_scc']
    step_size_prob = configs['scc']['step_size_prob']
    step_size_num_args = configs['general']['step_size_num_args']
    if type == 'easy':
        print("Instance to easy.")
        random_size = configs['general']['random_size']
        configs['general']['random_size'] = [ random_size[0] + step_size_num_args, random_size[1] + step_size_num_args]
        #configs['scc']['max_scc'] += step_size_max_scc
        configs['scc']['inner_attack_prob'] -= step_size_prob
        #configs['scc']['outer_attack_prob'] += step_size_prob
    else:
        print("Instance to hard.")
        random_size = configs['general']['random_size']
        configs['general']['random_size'] = [ random_size[0] - step_size_num_args, random_size[1] - step_size_num_args]
        #configs['scc']['step_size_max_scc'] += step_size_max_scc
        configs['scc']['inner_attack_prob'] += step_size_prob
        configs['scc']['outer_attack_prob'] -= step_size_prob
    if configs['scc']['inner_attack_prob'] < 0.1:
        configs['scc']['inner_attack_prob'] = 0.1
    elif configs['scc']['inner_attack_prob'] > 0.95:
        configs['scc']['inner_attack_prob'] = 0.95
    if configs['scc']['outer_attack_prob'] < 0.1:
        configs['scc']['outer_attack_prob'] = 0.1
    elif configs['scc']['outer_attack_prob'] > 0.95:
        configs['scc']['outer_attack_prob'] = 0.95
    print(configs['scc'])
    print(configs['general'])

def update_configs(user_options, default_options,generator_type):
    """Updates the default configurations with user specified options.

    Args:
        user_options (dict): Options specified via cli
        default_options (dict): Default configurations from generator_config.json

    """
    for key, value in user_options.params.items():
        if value is not None:
            if key in default_options['general'].keys():
                default_options['general'][key] = value
            elif key in default_options[generator_type].keys():
                default_options[generator_type][key] = value

def read_default_config(path: str) -> dict:
    with open(path,'rb') as fp:
        config =json.load(fp)
    return config

def save_instances(instances, general_configs, generator_type,offset=0):
    save_directory  = os.path.join(general_configs['save_to'],general_configs['name'])
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    with progressbar(enumerate(instances), label='Saving instances') as i_instances:
        for index,instance in i_instances:
            instance_name = f'{generator_type}_{index+1+offset}'
            for f in general_configs['format']:
                instance_content = parse_graph(f,instance)
                full_path_instance = os.path.join(save_directory,f'{instance_name}.{f}')
                with open(full_path_instance,'w') as file:
                    file.write(instance_content)
    return save_directory



@dispatch_on_value
def parse_graph(format,graph):
    pass

@parse_graph.register('i23')
def _nx_to_i23(format,graph: DiGraph):
    num_arguments = graph.number_of_nodes()
    arg_to_id_map = {a:i for i,a in enumerate(graph.nodes,start=1)}
    file_comment = f'# This file was created by the probo2 kwt-generator.\n'
    file_header = f'p af {num_arguments}\n'
    attacks_content = ""
    for attack in graph.edges:
        attacks_content += f'{arg_to_id_map[attack[0]]} {arg_to_id_map[attack[1]]}\n'
    
    return file_comment + file_header + attacks_content

@parse_graph.register('apx')
def _nx_to_apx(format,graph) -> str:
    """Parse networkx graph to apx format

    Args:
        graph (networkx.DiGraph): Directed nx graph

    Returns:
        str: Graph in apx format
    """

    arguments_str = "".join([f'arg({a}).\n' for a in graph.nodes])
    attacks__str = "".join([f'att({a},{b}).\n' for a,b in graph.edges]).rstrip()
    return arguments_str + attacks__str

@parse_graph.register('tgf')
def _nx_to_tgf(format,graph) -> str:
    """Parse networkx graph to tgf format

    Args:
        graph (networkx.DiGraph): Directed nx graph

    Returns:
        str: Graph in tgf format
    """

    arguments_str = "".join([f'{a}\n' for a in graph.nodes])
    attacks__str = "".join([f'{a} {b}\n' for a,b in graph.edges]).rstrip()
    return arguments_str+ "#\n" + attacks__str

def get_num_args(general_configs, num_instances):

    if general_configs['random_size']:
        range_args = list(general_configs['random_size'])

        return np.random.choice(range(range_args[0], range_args[1]+1), num_instances, replace=True)
    else:
        return [general_configs['num_args']] * num_instances