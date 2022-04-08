import src.generators.generator_utils as gen_utils
import numpy as np
import networkx as nx
from click import progressbar



def generate_instances(configs):
    general_configs = configs['general']
    stable_configs = configs['stable']
    save = general_configs['save']
    # init sampling parameters
    num_instances = general_configs['num']
    num_args = gen_utils.get_num_args(general_configs,num_instances)
    e_params,s_params,g_params = _init_sampling_parameters(stable_configs,num_instances)

    instances = []
    p = zip(num_args,e_params,s_params,g_params)
    with progressbar(p, label='Generating instances') as parameters:
        for a,e,s,g in parameters:
            instance = nx.DiGraph()
            instance.add_nodes_from(np.arange(1,a+1))
            ground = _isolate_arguments(instance,a,g)
            sample = np.random.choice(list(instance.nodes),s, replace=True)
            for argument in instance.nodes:
                if not(argument in sample) and not(argument in ground):
                    instance.add_edge(np.random.choice(sample,1)[0], argument)
            instances.append(instance)
    if save:
        return gen_utils.save_instances(instances,general_configs,'stable')
    else:
        return instances

def _init_sampling_parameters(stable_configs, num_instances):
    min_num_extensions, max_num_extension = tuple(stable_configs['num_extension'])
    min_size_extensions, max_size_extensions = tuple(stable_configs['size_extension'])
    min_size_grounded, max_size_grounded = tuple(stable_configs['size_grounded'])

    e =  np.random.choice(range(0, max_num_extension-min_num_extensions+1),num_instances, replace=True) + min_num_extensions
    s =  np.random.choice(range(0, max_size_extensions-min_size_extensions+1),num_instances, replace=True) + min_size_extensions
    g =  np.random.choice(range(0, max_size_grounded-min_size_grounded+1),num_instances, replace=True) + min_size_grounded
    return e,s,g

def _isolate_arguments(graph,a: int, g:int):
    ground = []
    for j in range(0,g):
        ground.append(list(graph.nodes)[np.random.randint(a)])
        for k in range(0,j):
            if np.random.random() < 0.2:
                graph.add_edge(ground[j],ground[k])

    return ground