import random
import src.generators.generator_utils as gen_utils
import numpy as np
import networkx as nx


def generate_instances(configs):
    general_configs = configs['general']
    grounded_configs = configs['grounded']
    num_instances = general_configs['num']
    num_args = gen_utils.get_num_args(general_configs,num_instances)
    instances = []
    for instance in range(0, num_instances):

        unconnected = []
        curr_instance = nx.DiGraph()
        curr_instance.add_nodes_from(np.arange(0,num_args[instance]))

        for i in range(0,num_args[instance]):
            unconnected.append(i)

            for j in range (0, i):
                if np.random.random() <= grounded_configs['attack_prob']:
                    curr_instance.add_edge(i,j)

                    if i in unconnected:
                        unconnected.remove(i)

        for arg in unconnected:
            k = random.choice(list(curr_instance.nodes))
            if bool(random.getrandbits(1)):
                curr_instance.add_edge(arg,k)
            else:
                curr_instance.add_edge(k, arg)
        instances.append(curr_instance)
    return gen_utils.save_instances(instances,general_configs,'grounded')











