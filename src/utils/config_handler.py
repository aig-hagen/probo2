from ftplib import error_perm
import json
from json import JSONEncoder
import os
from src.utils import definitions
import yaml
class Config(object):
    def __init__(self,name, task, benchmark, solver, timeout, repetitions, result_format,plot,yaml_file_name,raw_results_path=None):
        self.task = task
        self.benchmark = benchmark
        self.solver = solver
        self.timeout = timeout
        self.repetitions = repetitions
        self.name = name
        self.result_format = result_format
        self.plot = plot
        self.yaml_file_name = yaml_file_name
        self.raw_results_path = raw_results_path


    def write_config(self):
        save_to = os.path.join(definitions.CONFIGS_DIRECTORY,self.name)
        with open(save_to, 'w') as config_file:
            config_json = json.dumps(self,cls=ConfigDecoder, indent=4)

    def merge_user_input(self, cfg_to_merge):
        for key,value in cfg_to_merge.items():
            if key in self.__dict__.keys():
                if value is not None:
                    self.__dict__[key] = value

    def print(self):
        print(yaml.dump(self.__dict__))

    def dump(self,path):
        with open(os.path.join(path,self.yaml_file_name),'w') as cfg_file:
            yaml.dump(self.__dict__,cfg_file)

    def check(self):
        """Ensures that all configurations have valid values

        Returns:
            _type_: _description_
        """
        error = False
        msg_errors = ''
        if self.task is None:
            error = True
            msg_errors +=f"- No computational tasks found. Please specify tasks via --task option or in {self.yaml_file_name}.\n"
        if self.benchmark is None:
            error = True
            msg_errors +=f"- No benchmark found. Please specify benchmark via --benchmark option or in {self.yaml_file_name}.\n"
        if self.solver is None:
            error = True
            msg_errors +=f"- No solver found. Please specify benchmark via --solver option or in {self.yaml_file_name}.\n"
        if self.repetitions is None or self.repetitions < 1:
            error = True
            msg_errors +=f"- Invalid number of repetitions. Please specify benchmark via --repetitions option or in {self.yaml_file_name}.\n"
        if self.timeout is None or self.timeout < 0:
            error = True
            msg_errors +=f"- Invalid timeout. Please specify benchmark via --timeout option or in {self.yaml_file_name}.\n"
        if error:
            print('Bad configuration found:')
            print(msg_errors)
            print('Please refer to the documentation for additional pieces of information.\n')

            print('========== Experiment Summary ==========')
            self.print()

            exit()


class ConfigDecoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def load_config_yaml(path: str) -> Config:
    with open(path,'r') as config_file:
        if path.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            configs['yaml_file_name'] = os.path.basename(path)
    return configs

def load_default_config() -> Config:
    with open(definitions.DEFAULT_CONFIG_PATH,'r') as config_file:
        if definitions.DEFAULT_CONFIG_PATH.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            configs['yaml_file_name'] = os.path.basename(definitions.DEFAULT_CONFIG_PATH)

    return Config(**configs)








