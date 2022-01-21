import logging
import re
import subprocess
import os
from timeit import default_timer as timer
import time

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import column_property, relationship
from pathlib import Path
from sqlalchemy.sql.expression import null, update
from src.data import test
from src.utils import utils

from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Result import Result
from src.database_models import DatabaseHandler
from src.utils import Status
import src.analysis.validation as validation
import src.utils.definitions as definitions

class TaskNotSupported(ValueError):
    pass
class FormatNotSupported(ValueError):
    pass


def _init_cmd_params(solver_path):
    if solver_path.endswith('.sh'):
        return ['bash']
    elif solver_path.endswith('.py'):
        return ['python3']
    return []

class Solver(Base):
    __tablename__ = "solvers"
    solver_id = Column(Integer, primary_key=True)
    solver_name = Column(String, nullable=False)
    solver_path = Column(String, nullable=False)
    solver_format = Column(String, nullable=False)
    supported_tasks = relationship("Task", secondary=Supported_Tasks, back_populates="solvers")
    solver_version = Column(String, nullable=False)
    solver_results = relationship('Result')
    solver_full_name = column_property(solver_name + "_" + solver_version)




    def check_interface(self,test_task) -> bool:
        cmd_params = _init_cmd_params(self.solver_path)

        if self.solver_format == 'apx':
            instance = definitions.TEST_INSTANCE_APX
        elif self.solver_format == 'tgf':
            instance = definitions.TEST_INSTANCE_TGF
        else:
            raise FormatNotSupported(f'Format {format} is not supported for solver {self.solver_name}')



        cmd_params.extend([self.solver_path,
                  "-p", test_task,
                  "-f", instance,
                  "-fo", self.solver_format])

        if 'DS' or 'DC' in test_task:
            with open(definitions.TEST_INSTANCE_ARG,'r') as arg_file:
                arg = arg_file.read().rstrip('\n')
                cmd_params.extend(["-a",arg])

        try:
             result = utils.run_process(cmd_params,
                                       capture_output=True, timeout=5, check=True)
        except subprocess.TimeoutExpired as e:
            print(f'Solver interface test timed out.')
            return False
        except subprocess.CalledProcessError as err:
            print(f'Something went wrong when testing the interface: {err}')
            return False

        return True





    def fetch(self, prop):
        cmd_params = []
        if self.solver_path.endswith('.sh'):
            cmd_params.append('bash')
        elif self.solver_path.endswith('.py'):
            cmd_params.append('python3')
        cmd_params.append(self.solver_path)
        cmd_params.append("--{}".format(prop))
        solver_dir = os.path.dirname(self.solver_path)
        try:
            solver_output = utils.run_process(cmd_params,
                                        capture_output=True, check=True,cwd=solver_dir)
            solver_output = re.sub("\s+", " ",
                                solver_output.stdout.decode("utf-8")).strip(" ")

            solver_property = solver_output[solver_output.find("[") + 1:solver_output.find("]")].split(",")
        except subprocess.CalledProcessError as err:
                print("Error code: {}\nstdout: {}\nstderr:{}\n".format(err.returncode, err.output.decode("utf-8"), err.stderr.decode("utf-8")))
                exit()
        return solver_property

    def fetch_tasks(self,tasks=None)->list:
        """Calls solver with "problems" options.

        If tasks is specified it is checked wether the solver really supports these formats.
        If not a list of the actual supported formats is returned.

        Args:
            tasks (list, optional): List of tasks the solver might support. Defaults to None.

        Returns:
            list: List of tasks the solver acutally supports.
        """
        supported_tasks = sorted([ x.strip(" ") for x in self.fetch("problems")])
        if not tasks:
            return supported_tasks
        else:
            intersection_supported = set.intersection(set(supported_tasks),set(tasks))
            if intersection_supported:
                return list(intersection_supported)
            else:
                raise TaskNotSupported(f'None of the tasks {tasks} are supported for solver {self.solver_name}')
    def fetch_format(self,format=None):
        supported_formats = self.fetch("formats")
        if format:
            if format in supported_formats:
                return format
            else:
                raise FormatNotSupported(f'Format {format} is not supported for solver {self.solver_name}\nSupported by solver: {supported_formats}')
        else:
           return self.fetch("formats")[0]  # select first supported format

    def get_supported_tasks(self):
        supported = []
        for task in self.supported_tasks:
            supported.append(task.symbol)
        return supported


    def print_summary(self):
        print("**********SOLVER SUMMARY**********")
        print("Name: {} \nVersion: {} \nsolver_path: {} \nFormat: {} \nProblems: {}".format(self.solver_name,self.solver_version, self.solver_path, self.solver_format,
                                                                                                                    self.get_supported_tasks()))

    def run(self,task,benchmark,timeout, save_db=True, tag=None, session=None, update_status=True,n=1,first_n_instances=None):
        results = {}
        cmd_params = []
        arg_lookup = {}
        arg = ""
        solver_dir = os.path.dirname(self.solver_path)
        if self.solver_path.endswith('.sh'):

            # cmd_params.append('cd')
            # cmd_params.append(solver_dir)
            # cmd_params.append('&&')
            cmd_params.append('bash')
        elif self.solver_path.endswith('.py'):
            cmd_params.append('python')

        instances = benchmark.get_instances(self.solver_format)

        if first_n_instances is not None:
            instances = instances[0:first_n_instances]

        if "DS" in task.symbol or "DC" in task.symbol:
            arg_lookup = benchmark.generate_additional_argument_lookup(self.solver_format)

        for instance in instances:
            instance_name = Path(instance).stem
            params = [self.solver_path,
                  "-p", task.symbol,
                  "-f", instance,
                  "-fo", self.solver_format]
            if arg_lookup:
                arg = arg_lookup[instance_name]
                params.extend(["-a",arg])
            final_param = cmd_params + params
            try:
                total_run_time = 0
                for i in range(1,n+1):

                    start_time_current_run = time.perf_counter()
                    result = utils.run_process(final_param,
                                       capture_output=True, timeout=timeout, check=True,cwd=solver_dir)


                    end_time_current_run = time.perf_counter()
                    run_time_current_run = end_time_current_run - start_time_current_run
                    total_run_time+=run_time_current_run


                run_time = total_run_time / n

                solver_output = re.sub("\s+", "",
                                   result.stdout.decode("utf-8"))
                results[instance_name] = {'timed_out':False,'additional_argument': arg, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None}
            except subprocess.TimeoutExpired as e:
                results[instance_name] = {'timed_out':True,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None}
            except subprocess.CalledProcessError as err:
                logging.exception(f'Something went wrong running solver {self.solver_name}')
                print("\nError occured:",err)
                results[instance_name] = {'timed_out':False,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode}
            if save_db:
                data = results[instance_name]
                result = Result(tag=tag,solver_id=self.solver_id,benchmark_id = benchmark.id,task_id = task.id,
                            instance=instance_name,cut_off=timeout, timed_out = data['timed_out'],
                            runtime=data['runtime'], result=data['result'], additional_argument = data['additional_argument'],
                            benchmark=benchmark, solver=self, task=task, exit_with_error=data['exit_with_error'], error_code=data['error_code'])
                session.add(result)
                session.commit()
                del data
                del results[instance_name]
            if update_status:
                Status.increment_instances_counter(task.symbol, self.solver_id)
        return results
