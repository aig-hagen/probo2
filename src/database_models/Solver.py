import re
import subprocess
import os
from timeit import default_timer as timer

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import column_property, relationship
from pathlib import Path
from sqlalchemy.sql.expression import null

from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Result import Result
from src import DatabaseHandler, Status

class Solver(Base):
    __tablename__ = "solvers"
    id = Column(Integer, primary_key=True)
    solver_name = Column(String, nullable=False)
    solver_path = Column(String, nullable=False)
    solver_format = Column(String, nullable=False)
    supported_tasks = relationship("Task", secondary=Supported_Tasks, back_populates="solvers")
    solver_version = Column(String, nullable=False)
    solver_competition = Column(String, nullable=True)
    solver_author = Column(String, nullable=True)
    results = relationship('Result')
    fullname = column_property(solver_name + "_" + solver_version)

    def check_solver(self,tasks):
        print(tasks)

    def guess(self, prop):        
        cmd_params = []
        if self.solver_path.endswith('.sh'):
            cmd_params.append('bash')
        elif self.solver_path.endswith('.py'):
            cmd_params.append('python')
        cmd_params.append(self.solver_path)
        cmd_params.append("--{}".format(prop))
        try:
            solver_output = subprocess.run(cmd_params,
                                        capture_output=True, check=True)
            solver_output = re.sub("\s+", " ",
                                solver_output.stdout.decode("utf-8")).strip(" ")
            solver_property = solver_output[solver_output.find("[") + 1:solver_output.find("]")].split(",")

        except subprocess.CalledProcessError as err:
                print("Error code: {}\nstdout: {}\nstderr:{}\n".format(err.returncode, err.output.decode("utf-8"), err.stderr.decode("utf-8")))
                exit()
        return solver_property
    
    def get_supported_tasks(self):
        supported = []
        for task in self.supported_tasks:
            supported.append(task.symbol)
        return supported


    def print_summary(self):
        print("**********SOLVER SUMMARY**********")
        print("Name: {} \nVersion: {} \nsolver_path: {} \nFormat: {} \nProblems: {} \nCompetition: {} \nAuthor: {}".format(self.solver_name,
                                                                                                                    self.solver_version, self.solver_path, self.solver_format, 
                                                                                                                    self.get_supported_tasks(),self.solver_competition,self.solver_author))

    def run(self,task,benchmark,timeout, save_db=True, tag=None, session=None):
        results = {}
        cmd_params = []
        arg_lookup = {}
        arg = ""
        if self.solver_path.endswith('.sh'):
            cmd_params.append('bash')
        elif self.solver_path.endswith('.py'):
            cmd_params.append('python')
        
        instances = benchmark.get_instances(self.solver_format)

        if "DS" in task.symbol or "DC" in task.symbol:
            arg_lookup = benchmark.generate_additional_argument_lookup(self.solver_format)
        
        for instance in instances:
            instance_name = Path(instance).stem
            params = [self.solver_path,
                  "-p", task.symbol,
                  "-f", os.path.join(benchmark.benchmark_path, instance),
                  "-fo", self.solver_format]
            if arg_lookup:
                arg = arg_lookup[instance_name]
                params.extend(["-a",arg])
            final_param = cmd_params + params
            try:
                
                start_time = timer()
                result = subprocess.run(final_param,
                                        stdout=subprocess.PIPE, timeout=timeout, check=True)
                end_time = timer()
                run_time = end_time - start_time
                solver_output = re.sub("\s+", " ",
                                   result.stdout.decode("utf-8")).strip(" ")
                results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None}
            except subprocess.TimeoutExpired:
                results[instance] = {'timed_out':True,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None}
            except subprocess.CalledProcessError as err:
                 results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode}
            if save_db:
                data = results[instance]
                result = Result(tag=tag,solver_id=self.id,benchmark_id = benchmark.id,task_id = task.id,
                            instance=instance,cut_off=timeout, timed_out = data['timed_out'],
                            runtime=data['runtime'], result=data['result'], additional_argument = data['additional_argument'],
                            benchmark=benchmark, solver=self, task=task, exit_with_error=data['exit_with_error'], error_code=data['error_code'])
                session.add(result)
                session.commit()
                del data
                del results[instance]
            Status.increment_instances_counter(task.symbol,self.id)
        return results
