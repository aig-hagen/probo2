
import json
from json import JSONEncoder
import os


from src.functions import archive, statistics
from src.handler import benchmark_handler
from src.utils import definitions
import src.functions.register as register
import yaml
import pandas as pd
class Config(object):
    def __init__(self,name, task, benchmark, solver, timeout, repetitions, result_format,save_to,yaml_file_name,status_file_path=None,save_output=None,archive_output=None,archive=None,table_export=None,copy_raws=None,printing=None,plot=None,grouping=None,statistics=None,score=None,validation=None,significance=None,raw_results_path=None,exclude_task=None,solver_arguments=None,time_measurement=None):
        self.task = task
        self.exclude_task = exclude_task
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
        self.score = score
        self.printing = printing
        self.copy_raws = copy_raws
        self.table_export = table_export
        self.archive = archive
        self.save_output = save_output
        self.archive_output = archive_output
        self.validation = validation
        self.significance = significance
        self.solver_arguments = solver_arguments
        self.status_file_path = status_file_path
        self.time_measurement = time_measurement

    def write_config(self):
        save_to = os.path.join(definitions.CONFIGS_DIRECTORY,self.name)
        with open(save_to, 'w') as config_file:
            config_json = json.dumps(self,cls=ConfigDecoder, indent=4)

    def merge_user_input(self, cfg_to_merge):
        for key,value in cfg_to_merge.items():
            if key in self.__dict__.keys():
                if value is not None or value:
                    if not value and not isinstance(value,bool):
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

    def get_summary_as_string(self):
        return yaml.dump(self.__dict__)

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
        else:
            benchmarks = benchmark_handler.load_benchmark(self.benchmark)
            for b in benchmarks:
                if not os.path.exists(b["path"]):
                    error = True
                    msg_errors += f"- Path for benchmark {b['name']} not found."
                else:
                    if len(os.listdir(b['path'])) == 0:
                        error = True
                        msg_errors +=f"- No instances found for benchmark {b['name']} at path {b['path']}."
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
            plot_error = False
            if isinstance(self.plot, list):
                for p in self.plot:
                    if p not in register.plot_dict.keys() and p != 'all':
                        _invalid.append(p)
                        error=True
                        plot_error = True
            else:
                if self.plot not in register.plot_dict.keys() and (self.plot != 'all' or 'all'  not in self.plot):
                    _invalid.append(self.plot)
                    error = True
                    plot_error = True

            if plot_error:
                 msg_errors +=f"- Invalid plot type: {','.join(_invalid)}. Please choose from following options: {','.join(register.plot_dict.keys())}\n"

        if self.statistics is not None:
            _invalid = []
            stat_error = False
            if isinstance(self.statistics, list):
                for stat in self.statistics:
                    print(stat)
                    if stat not in register.stat_dict.keys() and stat != 'all':
                        _invalid.append(stat)
                        error=True
                        stat_error=True
            else:
                if self.statistics not in register.stat_dict.keys() and (self.statistics != 'all' or 'all' not in self.statistics):
                    _invalid.append(self.statistics)
                    error = True
                    stat_error = True

            if stat_error:
                 msg_errors +=f"- Invalid statistics : {','.join(_invalid)}. Please choose from following options: {','.join(register.stat_dict.keys())}\n"
        if self.archive is not None:
            _invalid = []
            arch_error = False
            if isinstance(self.archive, list):
                for _format in self.archive:
                    if _format not in register.archive_functions_dict.keys():
                        _invalid.append(stat)
                        error=True
                        arch_error = True
            else:
                if self.archive not in register.archive_functions_dict.keys() and (self.archive != 'all'):
                    _invalid.append(self.archive)
                    error = True
                    arch_error = True

            if arch_error:
                 msg_errors +=f"- Invalid archive format : {','.join(_invalid)}. Please choose from following options: {','.join(register.archive_functions_dict.keys())}\n"

        if error:
            print('Bad configuration found:')
            print(msg_errors)
            print('Please refer to the documentation for additional pieces of information.\n')

            print('========== Experiment Summary ==========')
            self.print()

            return False
        return True


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

def load_config_via_name(name):
    experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)
    if name in experiment_index.name.values:
        cfg_path = experiment_index[experiment_index.name.values == name]['config_path'].iloc[0]
        cfg = load_config_yaml(cfg_path,as_obj=True)
        return cfg
    else:
        print(f"No Config with name {name} found.")
        exit()


def create_solver_argument_grid(grid_file_path, solver_list: list) -> list:
    import itertools
    with open(grid_file_path,'r') as config_file:
        if grid_file_path.endswith('yaml'):
            argument_grids = yaml.load(config_file, Loader=yaml.FullLoader)


    for solver, argument_configs in argument_grids.items():

        arguments_list = argument_configs['arguments']
        argument_prefix = argument_configs['argument_prefix']
        argument_values_lists = []
        arg_names = []
        for argument in arguments_list:
            print(f"{argument=}")
            for arg_name, arg_values in argument.items():
                arg_names.append(arg_name)
                print(arg_name)
                print(arg_values)
                step = arg_values['step']
                start = arg_values['range'][0]
                end = arg_values['range'][1] + arg_values['step']

                argument_values = list(range(start,end,step))
                argument_values_lists.append(argument_values)
        argument_combinations = list(itertools.product(*argument_values_lists))
        prompts = _construct_command_line_prompt(argument_combinations, arg_names, argument_prefix)
        print(prompts)


def _construct_command_line_prompt(argument_combination_list,argument_names,argument_prefix):
    # NOTE: argument_names and tuples in argument_values list must be the same order !
    print(argument_combination_list)
    print(argument_names)
    print(argument_prefix)

    argument_prompt = [f"{argument_prefix}{n}" for n in argument_names]
    print(argument_prompt)
    all_prompts = []
    for combination in argument_combination_list:
        _combs= list(zip(argument_prompt,combination))
        prompt = list(sum(_combs,())) # hack to flatten the tuples in the list to a list
        all_prompts.append(prompt)
    return all_prompts

    # TODO: Create solver with prompts





















