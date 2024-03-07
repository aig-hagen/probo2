from src.utils.configs import CommandConfigInterface
import yaml
import os
from src.functions import register


class ValidationConfig(CommandConfigInterface):
    def __init__(self, mode:list, plot: list, table_export, experiment_id) -> None:
        super().__init__()
        self.mode = mode
        self.plot = plot
        self.table_export = table_export
        self.experiment_id = experiment_id
    
    def print(self):
        print(yaml.dump(self.__dict__))
    
    def check(self):
        pass
    
    def dump(self,path):
        with open(os.path.join(path,self.yaml_file_name),'w') as cfg_file:
            yaml.dump(self.__dict__,cfg_file)




class SignificanceTestConfig(CommandConfigInterface):
    def __init__(self, parametric_test: list, 
                parametric_post_hoc: list,
                non_parametric_test: list,
                non_parametric_post_hoc: list,
                p_adjust: str,
                plot: list,
                table_export
                 ) -> None:
        super().__init__()
        self.parametric_test = parametric_test
        self.non_parametric_test = non_parametric_test
        self.parametric_post_hoc = parametric_post_hoc
        self.non_parametric_post_hoc = non_parametric_post_hoc
        self.p_adjust = p_adjust
        self.plot = plot
        self.table_export = table_export
    
    def print(self):
        print(self.to_string())
    
    def to_string(self):
        return yaml.dump(self.__dict__)
    
    def check(self):
        pass
    
    def dump(self,path):
        with open(os.path.join(path,self.yaml_file_name),'w') as cfg_file:
            yaml.dump(self.__dict__,cfg_file)

class SolverArgumentConfig(CommandConfigInterface):
     pass

class ExperimentConfig(CommandConfigInterface):
    def __init__(self,
                 name: str, 
                 task: list, 
                 benchmark: list, 
                 solver: list, 
                 timeout: int, 
                 repetitions: int, 
                 result_format,
                 save_to: str,
                 yaml_file_name: str,
                 status_file_path: str,
                 save_output: bool,
                 archive_output: bool,
                 archive: list,
                 table_export,
                 copy_raws: bool,
                 printing: str,
                 plot: list,
                 grouping: list,
                 statistics: list,
                 score: list,
                 validation: ValidationConfig ,
                 significance: SignificanceTestConfig,
                 solver_arguments: SolverArgumentConfig,
                 raw_results_path: str,
                 exclude_task: list):
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
        
        
    
    def print(self):
        print(self.to_string())
    
    def to_string(self):
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
            print(self.statistics)
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

