"""Module to calculate node features and embeddings
"""
import src.functions.register as register
from tqdm import tqdm
import pandas as pd

import networkx as nx
import numpy as np
from src.utils import benchmark_handler
import karateclub.node_embedding.neighbourhood as node_embeddings
import karateclub.node_embedding.structural as structural_node_embeddings
import karateclub.node_embedding.meta as meta_node_embeddings
import os
import csv

def tgf_to_nx(tgf_file_content) -> nx.DiGraph: 
    args_attacks = tgf_file_content.split('#\n')
    args = args_attacks[0].split()
    attacks = args_attacks[1].rstrip().split('\n')
    attacks = [tuple(att.split()) for att in attacks]
   
    
    G = nx.DiGraph()
    ids_args =  dict([(int(x),y) for x,y in enumerate(args)])
    args_ids = {v: k for k, v in ids_args.items()}
    attacks = [(args_ids[att[0]],args_ids[att[1]]) for att in attacks]
    G.add_nodes_from(ids_args.keys())
    G.add_edges_from(attacks)
    return G,args_ids,ids_args

def apx_to_nx(apx_file_content) -> nx.DiGraph:
    args_attacks = apx_file_content.split('att')
    args = args_attacks[0].split()
    args = [ s[s.find("(")+1:s.find(")")] for s in args]
    attacks = [ tuple(att.replace("(","").replace(")","").replace("."," ").rstrip().split(",")) for att in args_attacks[1:]]
    G = nx.DiGraph()
    ids_args =  dict([(int(x),y) for x,y in enumerate(args)])
    args_ids = {v: k for k, v in ids_args.items()}
    G.add_nodes_from(args_ids.keys())
    G.add_edges_from(attacks)
    return G, args_ids,ids_args

def i23_to_nx(file_content) -> nx.DiGraph:
    pass

parse_function_dict = {'tgf': tgf_to_nx,'apx': apx_to_nx}

def build_graph(instance_path, format):
    with open(instance_path,'r') as f:
        f_content = f.read()
    
    return parse_function_dict[format](f_content)


def calculate_features(benchmark_info: dict,feature,embedding,save_to):
    if not os.path.exists(save_to):
        print(f'Path {save_to} not found')
        exit()
    
    save_to = os.path.join(save_to,f"{benchmark_info['name']}_features")
    embeddings_save_to = os.path.join(save_to,'embeddings')
    features_save_to = os.path.join(save_to,'features')
    mappings_save_to = os.path.join(save_to,'mappings')
    os.makedirs(embeddings_save_to,exist_ok=True)
    os.makedirs(features_save_to,exist_ok=True)
    os.makedirs(mappings_save_to,exist_ok=True)

    parse_format = 'tgf' if 'tgf' in benchmark_info['format'] else benchmark_info['format'][0]
    instances = benchmark_handler.get_instances(benchmark_info['path'], parse_format,full_path=True)
    all_instances_index = []
    for instance_path in tqdm(instances,desc='Calculating features'):
        instance_graph,args_ids, ids_args = build_graph(instance_path,parse_format)
     
        instance_features = {calculate: register.feature_calculation_functions_dict[calculate](instance_graph) for calculate in feature}
        print(instance_features)
        instance_embeddings = {calculate: register.embeddings_calculation_functions_dict[calculate](instance_graph) for calculate in embedding}
        instance_name = os.path.basename(instance_path)[:-4]
        
        instance_embeddings_save_to = os.path.join(embeddings_save_to,f'{instance_name}_embeddings.npz')
        instance_features_save_to = os.path.join(features_save_to,f'{instance_name}_features.csv')
        args_ids_save_to = os.path.join(mappings_save_to,f'{instance_name}_args_to_ids.csv')
        ids_args_save_to = os.path.join(mappings_save_to,f'{instance_name}_ids_to_args.csv')
        with open(args_ids_save_to, 'w') as csvfile:
            fieldnames = ['arg', 'id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key in args_ids:
                writer.writerow({'arg': key, 'id': args_ids[key]})
        with open(ids_args_save_to, 'w') as csvfile:
            fieldnames = ['id', 'arg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key in ids_args:
                writer.writerow({'id': key, 'arg': ids_args[key]})
        np.savez(os.path.join(embeddings_save_to,f'{instance_name}_embeddings.npz'),**instance_embeddings)
        
        instance_features_df = pd.DataFrame(instance_features)
        print(instance_features_df)
        instance_features_df['id'] = instance_features_df.index
        instance_features_df['arg'] = instance_features_df['id'].map(ids_args)
        all_instances_index.append({'instance_name': instance_name,
                                    'format':parse_format,
                                    'embeddings_path': os.path.relpath(instance_embeddings_save_to,save_to),
                                     'features_path': os.path.relpath(instance_features_save_to,save_to),
                                     'args_ids_map': os.path.relpath(args_ids_save_to,save_to),
                                     'ids_args_map': os.path.relpath(ids_args_save_to,save_to)})
    
    pd.DataFrame(all_instances_index).to_csv(os.path.join(save_to,f"{benchmark_info['name']}_index.csv"))

        
        
        
    
#========== networkx features ==========
def degree_centrality(graph: nx.DiGraph):
    return nx.degree_centrality(graph)

def in_degree_centrality(graph: nx.DiGraph):
    return nx.in_degree_centrality(graph)

def out_degree_centrality(graph: nx.DiGraph):
    return nx.out_degree_centrality(graph)


def eigenvector_centrality(graph: nx.DiGraph):
    return nx.eigenvector_centrality(graph)

def katz_centrality(graph: nx.DiGraph):
    return nx.katz_centrality(graph)

def closeness_centrality(graph: nx.DiGraph):
    return nx.closeness_centrality(graph)

def test(graph: nx.DiGraph):
    return nx.global_reaching_centrality(graph)

def harmonic_centrality(graph: nx.DiGraph):
    return nx.harmonic_centrality(graph)


def betweenness_centrality(graph: nx.DiGraph):
    return nx.betweenness_centrality(graph)









#=========== Node Embeddings =========== 
def deep_walk(graph: nx.DiGraph):
    model = node_embeddings.DeepWalk()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def test_embedding(graph: nx.DiGraph):
    _m = node_embeddings.Node2Vec()
    model = meta_node_embeddings.NEU()

    model.fit(graph,_m)
    embeddings = model.get_embedding()
    print(embeddings.shape)
    return embeddings
def NEU(graph: nx.DiGraph):
    _m = node_embeddings.Node2Vec()
    model = meta_node_embeddings.NEU()

    model.fit(graph,_m)
    embeddings = model.get_embedding()
    return embeddings
def Role2Vec(graph: nx.DiGraph):
    model = structural_node_embeddings.Role2Vec()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings
def NMFADMM(graph: nx.DiGraph):
    model = node_embeddings.NMFADMM()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def Node2Vec(graph: nx.DiGraph):
    model = node_embeddings.Node2Vec()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def GraRep(graph: nx.DiGraph):
    model = node_embeddings.GraRep()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def Walklets(graph: nx.DiGraph):
    model = node_embeddings.Walklets()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def BoostNE(graph: nx.DiGraph):
    model = node_embeddings.BoostNE()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def NetMF(graph: nx.DiGraph):
    model = node_embeddings.NetMF()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def NodeSketch(graph: nx.DiGraph):
    model = node_embeddings.NodeSketch()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def Diff2Vec(graph: nx.DiGraph):
    model = node_embeddings.Diff2Vec()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings

def RandNE(graph: nx.DiGraph):
    model = node_embeddings.RandNE()

    model.fit(graph)
    embeddings = model.get_embedding()
    return embeddings



    


register.feature_calculation_register('test', test)
register.feature_calculation_register('degree_centrality', degree_centrality)
register.feature_calculation_register('out_degree_centrality', out_degree_centrality)
register.feature_calculation_register('in_degree_centrality', in_degree_centrality)
register.feature_calculation_register('eigenvector_centrality', eigenvector_centrality)
register.feature_calculation_register('katz_centrality', katz_centrality)
register.feature_calculation_register('closeness_centrality', closeness_centrality)
register.feature_calculation_register('betweenness_centrality', betweenness_centrality)
register.feature_calculation_register('harmonic_centrality', harmonic_centrality)


register.embeddings_calculation_register('DeepWalk',deep_walk)
register.embeddings_calculation_register('RandNE',RandNE)
register.embeddings_calculation_register('Diff2Vec',Diff2Vec)
register.embeddings_calculation_register('NodeSketch',NodeSketch)
register.embeddings_calculation_register('NetMF',NetMF)
register.embeddings_calculation_register('BoostNE',BoostNE)
register.embeddings_calculation_register('Walklets',Walklets)
register.embeddings_calculation_register('GraRep',GraRep)
register.embeddings_calculation_register('Node2Vec',Node2Vec)
register.embeddings_calculation_register('NMFADMM',NMFADMM)
register.embeddings_calculation_register('Role2Vec',Role2Vec)
register.embeddings_calculation_register('NEU',NEU)

   


