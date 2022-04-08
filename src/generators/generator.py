from dataclasses import replace
from re import I
import src.generators.generator_utils as gen_utils
from src.utils.utils import dispatch_on_value

import numpy as np
import networkx as nx

import src.generators.stable_generator as st_gen
import src.generators.scc_generator as scc_gen
import src.generators.grounded_generator as grounded_gen

@dispatch_on_value
def generate(generator,configs):
    pass

@generate.register("stable")
def stable_generator(generator,configs):
    return st_gen.generate_instances(configs)

@generate.register("scc")
def scc_generator(generator, configs):
    return scc_gen.generate_instances(configs)

@generate.register("grounded")
def grounded_generator(generator, configs):
    return grounded_gen.generate_instances(configs)