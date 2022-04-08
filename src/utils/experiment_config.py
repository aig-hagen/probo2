import json
from json import JSONEncoder
import os
from src.utils import definitions
import yaml
class Config(object):
    def __init__(self,name, task, benchmark, solver, timeout, repetitions, result_format):
        self.task = task
        self.benchmark = benchmark
        self.solver = solver
        self.timeout = timeout
        self.repetitions = repetitions
        self.name = name
        self.result_format = result_format


    def write_config(self):
        save_to = os.path.join(definitions.CONFIGS_DIRECTORY,self.name)
        with open(save_to, 'w') as config_file:
            config_json = json.dumps(self,cls=ConfigDecoder, indent=4)

def load_config(path):
    with open(path,'r') as config_file:
        if path.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)

    return Config(**configs)



class ConfigDecoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


