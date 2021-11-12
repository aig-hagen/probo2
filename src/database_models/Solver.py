import re
import subprocess
import os
from timeit import default_timer as timer

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import column_property, relationship
from pathlib import Path
from sqlalchemy.sql.expression import null, update

from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Result import Result
from src.database_models import DatabaseHandler
from src.utils import Status
import src.analysis.validation as validation
import src.utils.definitions as definitions


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

    def check_solver(self,tasks):
        engine = DatabaseHandler.get_engine()
        session = DatabaseHandler.create_session(engine)
        benchmark = DatabaseHandler.get_benchmark(session, 4)
        task = DatabaseHandler.get_task(session, "EE-CO")


        if 'EE-CO' in tasks:
            task = DatabaseHandler.get_task(session, "EE-CO")
            solver_output = self.run(task,benchmark,600,save_db=False)
            instance = list(solver_output.keys())[0]
            data = solver_output[instance]
            extensions_solver = validation.multiple_extensions_string_to_list(data['result'])
            extension_reference = validation.multiple_extensions_string_to_list(validation.get_reference_result_enumeration(definitions.TEST_INSTANCES_REF_PATH,instance,'EE-CO'))
            val_result = validation.compare_results_enumeration(extensions_solver,extension_reference)
            if val_result == 'correct':
                return True
            else:
                return False
        elif 'SE-CO' in tasks:
            task = DatabaseHandler.get_task(session, "SE-CO")
            solver_output = self.run(task,benchmark,600,save_db=False)
            instance = list(solver_output.keys())[0]
            data = solver_output[instance]
            extensions_solver = validation.single_extension_string_to_list(data['result'])
            extension_reference = validation.single_extension_string_to_list(validation.get_reference_result_enumeration(definitions.TEST_INSTANCES_REF_PATH,instance,'SE-CO'))
            val_result = validation.compare_results_enumeration(extensions_solver,extension_reference)
            if val_result == 'correct':
                return True
            else:
                return False

        return False
    def guess(self, prop):
        cmd_params = []
        if self.solver_path.endswith('.sh'):
            cmd_params.append('bash')
        elif self.solver_path.endswith('.py'):
            cmd_params.append('python3')
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
        print("Name: {} \nVersion: {} \nsolver_path: {} \nFormat: {} \nProblems: {}".format(self.solver_name,self.solver_version, self.solver_path, self.solver_format,
                                                                                                                    self.get_supported_tasks()))

    def run(self,task,benchmark,timeout, save_db=True, tag=None, session=None, update_status=True,n=1):
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
                total_run_time = 0
                for i in range(1,n+1):

                    start_time_current_run = timer()
                    result = subprocess.run(final_param,
                                       capture_output=True, timeout=timeout, check=True)

                    end_time_current_run = timer()
                    run_time_current_run = end_time_current_run - start_time_current_run
                    total_run_time+=run_time_current_run


                run_time = total_run_time / n


                solver_output = re.sub("\s+", "",
                                   result.stdout.decode("utf-8"))
                results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None}
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                results[instance] = {'timed_out':True,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None}
            except subprocess.CalledProcessError as err:
                print("Error occured")
                results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode}
            if save_db:
                data = results[instance]
                result = Result(tag=tag,solver_id=self.solver_id,benchmark_id = benchmark.id,task_id = task.id,
                            instance=instance,cut_off=timeout, timed_out = data['timed_out'],
                            runtime=data['runtime'], result=data['result'], additional_argument = data['additional_argument'],
                            benchmark=benchmark, solver=self, task=task, exit_with_error=data['exit_with_error'], error_code=data['error_code'])
                session.add(result)
                session.commit()
                del data
                del results[instance]
            # else:
            #     print("")
            #     print( re.sub("\s+", "",
            #                        result.stdout.decode("utf-8")))
            #     print("--------------------------------")
            if update_status:
                Status.increment_instances_counter(task.symbol, self.solver_id)
        return results
