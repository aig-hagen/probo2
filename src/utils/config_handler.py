from ftplib import error_perm
import json
from json import JSONEncoder
import os
from src.functions import archive, statistics
from src.utils import definitions
import src.functions.register as register
import yaml
class Config(object):
    def __init__(self,name, task, benchmark, solver, timeout, repetitions, result_format,save_to,yaml_file_name,archive=None,table_export=None,copy_raws=None,printing=None,plot=None,grouping=None,statistics=None,raw_results_path=None):
        self.task = task
        self.benchmark = benchmark
        self.solver = solver
        self.timeout = timeout
        self.repetitions = repetitions
        self.name = name
        self.result_format = result_format
        self.plot = plot
        self.grouping = grouping
        self.yaml_file_name = yaml_file_name
        self.raw_results_path = raw_results_path
        self.save_to = save_to
        self.statistics = statistics
        self.printing = printing
        self.copy_raws = copy_raws
        self.table_export = table_export
        self.archive = archive

    def write_config(self):
        save_to = os.path.join(definitions.CONFIGS_DIRECTORY,self.name)
        with open(save_to, 'w') as config_file:
            config_json = json.dumps(self,cls=ConfigDecoder, indent=4)

    def merge_user_input(self, cfg_to_merge):
        for key,value in cfg_to_merge.items():
            if key in self.__dict__.keys():
                if value is not None or value:
                    if not value:
                        pass
                    else:
                        if isinstance(value,tuple):
                            value = list(value)
                        if isinstance(self.__dict__[key],list):

                            if  isinstance(value,list):
                                _intersection = set.intersection(set(self.__dict__[key]), set(value))
                                _to_add = list(set.difference(set(value),_intersection ))
                                self.__dict__[key].extend(_to_add)
                            else:
                                if value not in self.__dict__[key]:
                                    self.__dict__[key].append(value)
                        else:
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

        if self.plot is not None:
            _invalid = []
            if isinstance(self.plot, list):
                for p in self.plot:
                    if p not in register.plot_dict.keys():
                        _invalid.append(p)
                        error=True
            else:
                if self.plot not in register.plot_dict.keys() and (self.plot != 'all'):
                    _invalid.append(self.plot)
                    error = True

            if error:
                 msg_errors +=f"- Invalid plot type: {','.join(_invalid)}. Please choose from following options: {','.join(register.plot_dict.keys())}\n"

        if self.statistics is not None:
            _invalid = []
            if isinstance(self.statistics, list):
                for stat in self.statistics:
                    if stat not in register.stat_dict.keys():
                        _invalid.append(stat)
                        error=True
            else:
                if self.statistics not in register.stat_dict.keys() and (self.statistics != 'all'):
                    _invalid.append(self.statistics)
                    error = True

            if error:
                 msg_errors +=f"- Invalid statistics : {','.join(_invalid)}. Please choose from following options: {','.join(register.stat_dict.keys())}\n"
        if self.archive is not None:
            _invalid = []
            if isinstance(self.archive, list):
                for _format in self.archive:
                    if _format not in register.archive_functions_dict.keys():
                        _invalid.append(stat)
                        error=True
            else:
                if self.archive not in register.archive_functions_dict.keys() and (self.archive != 'all'):
                    _invalid.append(self.archive)
                    error = True

            if error:
                 msg_errors +=f"- Invalid archive format : {','.join(_invalid)}. Please choose from following options: {','.join(register.archive_functions_dict.keys())}\n"


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

def load_config_yaml(path: str, as_obj=False) -> Config:
    with open(path,'r') as config_file:
        if path.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            configs['yaml_file_name'] = os.path.basename(path)
    if not as_obj:
        return configs
    else:
        return Config(**configs)


def load_default_config() -> Config:
    with open(definitions.DEFAULT_CONFIG_PATH,'r') as config_file:
        if definitions.DEFAULT_CONFIG_PATH.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            configs['yaml_file_name'] = os.path.basename(definitions.DEFAULT_CONFIG_PATH)
    return Config(**configs)


def load_all_configs(directory: str) -> list:
    configs = []
    files_in_directory = os.listdir(directory)
    for f in files_in_directory:
        if f.endswith('.yaml'):
            f_path = os.path.join(directory,f)
            configs.append(load_config_yaml(f_path))
    return configs











