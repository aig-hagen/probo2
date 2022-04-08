import src.generators.generator_utils as gen_utils
import numpy as np
import networkx as nx
def generate_instances(configs):
    general_configs = configs['general']
    scc_configs = configs['scc']
    num_instances = general_configs['num']
    save = general_configs['save']
    num_args = gen_utils.get_num_args(general_configs,num_instances)
    num_scc = np.random.choice(range(1,scc_configs['max_scc']+1),num_instances)
    instances = []
    for instance in range(0, num_instances):
        arguments =np.arange(1,num_args[instance]+1)
        curr_instance = nx.DiGraph()
        curr_instance.add_nodes_from(arguments)

        if num_args[instance] <= num_scc[instance]:
            num_scc[instance] = num_args[instance] -1
        sccs = arguments.copy()
        np.random.shuffle(sccs)
        sccs = np.array_split(sccs,num_scc[instance])
        _generate_inner_attacks(sccs,scc_configs['inner_attack_prob'],curr_instance)
        _generate_outer_attacks(sccs,scc_configs['outer_attack_prob'],curr_instance)
        instances.append(curr_instance)
    if save:
        return gen_utils.save_instances(instances,general_configs,'scc')
    else:
        return instances




def _generate_inner_attacks(sccs, inner_attack_prob, graph):
    if inner_attack_prob < 0.1:
        inner_attack_prob = 0.1
    if inner_attack_prob > 0.95:
        inner_attack_prob = 0.95
    for scc in sccs:
        for i in range(0,scc.size):
            for j in range(0,scc.size):
                if np.random.random() < inner_attack_prob:
                    graph.add_edge(scc[i],scc[j])

def _generate_outer_attacks(sccs, outer_attack_prob, graph):
    if outer_attack_prob < 0.1:
        outer_attack_prob = 0.1
    if outer_attack_prob > 0.95:
        outer_attack_prob = 0.95
    for i in range(0, len(sccs) - 1):
        for j in range(0, len(sccs)):
            if np.random.random() < 0.3:
                scc_1 = sccs[i]
                scc_2 = sccs[j]
                for arg_1 in range(0,scc_1.size):
                    for arg_2 in range(0,scc_2.size):
                        if np.random.random() < outer_attack_prob:
                            graph.add_edge(scc_1[arg_1],scc_2[arg_2])


