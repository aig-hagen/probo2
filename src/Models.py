import re
import subprocess
import os
import click
import random
from timeit import default_timer as timer

from sqlalchemy import Column, ForeignKey, Integer, String, Table, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import column_property, relationship
from pathlib import Path
from sqlalchemy.sql.expression import null

from sqlalchemy.sql.sqltypes import Float

from src.database_models.Base import Base


from sqlalchemy.ext.declarative import declarative_base



# Supported_Tasks = Table(
#   "Supported_Tasks",
#   Base.metadata,
#   Column("solver_id", Integer, ForeignKey("solvers.id")),
#   Column("problem_id", Integer, ForeignKey("tasks.id")),
# )
# class Solver(Base):
#     __tablename__ = "solvers"
#     id = Column(Integer, primary_key=True)
#     solver_name = Column(String, nullable=False)
#     solver_path = Column(String, nullable=False)
#     solver_format = Column(String, nullable=False)
#     supported_tasks = relationship("Task", secondary=Supported_Tasks, back_populates="solvers")
#     solver_version = Column(String, nullable=False)
#     solver_competition = Column(String, nullable=True)
#     solver_author = Column(String, nullable=True)
#     results = relationship('Result')
#     fullname = column_property(solver_name + "_" + solver_version)

#     def check_solver(self,tasks):
#         print(tasks)

#     def guess(self, prop):        
#         cmd_params = []
#         if self.solver_path.endswith('.sh'):
#             cmd_params.append('bash')
#         elif self.solver_path.endswith('.py'):
#             cmd_params.append('python')
#         cmd_params.append(self.solver_path)
#         cmd_params.append("--{}".format(prop))
#         try:
#             solver_output = subprocess.run(cmd_params,
#                                         capture_output=True, check=True)
#             solver_output = re.sub("\s+", " ",
#                                 solver_output.stdout.decode("utf-8")).strip(" ")
#             solver_property = solver_output[solver_output.find("[") + 1:solver_output.find("]")].split(",")
#             print(solver_property)
#         except subprocess.CalledProcessError as err:
#                 print("Error code: {}\nstdout: {}\nstderr:{}\n".format(err.returncode, err.output.decode("utf-8"), err.stderr.decode("utf-8")))
#                 exit()
#         return solver_property
    
#     def get_supported_tasks(self):
#         supported = []
#         for task in self.supported_tasks:
#             supported.append(task.symbol)
#         return supported


#     def print_summary(self):
#         print("**********SOLVER SUMMARY**********")
#         print("Name: {} \nVersion: {} \nsolver_path: {} \nFormat: {} \nProblems: {} \nCompetition: {} \nAuthor: {}".format(self.solver_name,
#                                                                                                                     self.solver_version, self.solver_path, self.solver_format, 
#                                                                                                                     self.get_supported_tasks(),self.solver_competition,self.solver_author))

#     def run(self,task,benchmark,timeout):
#         results = {}
#         cmd_params = []
#         arg_lookup = {}
#         arg = ""
#         if self.solver_path.endswith('.sh'):
#             cmd_params.append('bash')
#         elif self.solver_path.endswith('.py'):
#             cmd_params.append('python')
        
#         instances = benchmark.get_instances(self.solver_format)

#         if "DS" in task.symbol or "DC" in task.symbol:
#             arg_lookup = benchmark.generate_additional_argument_lookup(self.solver_format)
        
#         for instance in instances:
#             instance_name = Path(instance).stem
#             params = [self.solver_path,
#                   "-p", task.symbol,
#                   "-f", os.path.join(benchmark.benchmark_path, instance),
#                   "-fo", self.solver_format]
#             if arg_lookup:
#                 arg = arg_lookup[instance_name]
#                 params.extend(["-a",arg])
#             final_param = cmd_params + params
#             try:
                
#                 start_time = timer()
#                 result = subprocess.run(final_param,
#                                         stdout=subprocess.PIPE, timeout=timeout, check=True)
#                 end_time = timer()
#                 run_time = end_time - start_time
#                 solver_output = re.sub("\s+", " ",
#                                    result.stdout.decode("utf-8")).strip(" ")
#                 results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': run_time, 'result': solver_output, 'exit_with_error': False, 'error_code': None}
#             except subprocess.TimeoutExpired:
#                 results[instance] = {'timed_out':True,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': False, 'error_code': None}
#             except subprocess.CalledProcessError as err:
#                  results[instance] = {'timed_out':False,'additional_argument': arg, 'runtime': None, 'result': None, 'exit_with_error': True, 'error_code': err.returncode}
#         return results

            
            
# class Task(Base):
#     __tablename__ = "tasks"
#     id = Column(Integer, primary_key=True)
#     symbol = Column(String, nullable=False)
#     solvers = relationship("Solver", secondary=Supported_Tasks, back_populates="supported_tasks")
#     results = relationship('Result')

# class Benchmark(Base):
#     __tablename__ = "benchmarks"
#     id = Column(Integer, primary_key=True)
#     benchmark_name = Column(String, nullable=False)
#     benchmark_path = Column(String, nullable=False)
#     graph_type = Column(String, nullable=True)
#     format_instances = Column(String, nullable=False)
#     hardness = Column(String, nullable=True)
#     extension_arg_files = Column(String, nullable=False)
#     benchmark_competition = Column(String, nullable=True)
#     results = relationship("Result")

#     def get_instances(self,extension,without_extension=False,full_path=False):
#         instances = []
#         for instance in os.listdir(self.benchmark_path):
#             if instance.endswith(extension):
#                 if without_extension:
#                     instances.append(Path(instance).stem)
#                 elif full_path:
#                     instances.append(os.path.join(self.benchmark_path,instance))
#                 else:
#                     instances.append(instance)
#         return sorted(instances)
    
#     def get_argument_files(self):
#         return self.get_instances(self.extension_arg_files)
    
#     def generate_additional_argument_lookup(self, format):
#         lookup = {}
#         argument_files = self.get_argument_files()
#         for file in argument_files:
#             argument_file_path = os.path.join(self.benchmark_path, file)

#             try:
#                 with open(argument_file_path, 'r') as af:
#                     argument_param = af.read().replace('\n', '')
#             except IOError as err:
#                 print(err)
            
#             instance_name = Path(file).stem
#             lookup[instance_name] = argument_param


#         instances = self.get_instances(format,without_extension=True)

#         return {k: lookup[k] for k in instances} # Select only the argument files of instances that are present in the solver format



#     def get_formats(self):
#         return self.format_instances.split(',')

#     def generate_files(self, generate_format, present_format=''):
#         """Generate files in specified format

#         Args:
#           generate_format: Format of files to generate
#           present_format: Format of corresponding files in fileset

#         Returns:
#           None

#         """
#         if not present_format:
#             present_format = self.get_formats()[0]

#         if present_format not in self.get_formats():
#             raise ValueError("Format {} not supported!".format(present_format))

#         if generate_format.upper() not in ['APX', 'TGF']:
#             raise ValueError("Can not generate {} instances".format(generate_format))

#         if generate_format not in self.get_formats():
#             self.format_instances += "," + generate_format
#         num_generated = 0
        
#         with click.progressbar(self.get_instances(present_format),
#                                label="Generating {} files:".format(generate_format)) as instance_progress:
#             for instance in instance_progress:
#                 instance_name = instance.split('.')[0]
#                 generate_instance_name = instance_name + "." + generate_format
#                 if generate_instance_name not in self.get_instances(generate_format):
#                     self.generate_single_file(instance_name, generate_format, present_format)
#                     num_generated += 1

#         print("{} files generated.".format(num_generated))
    
#     def generate_single_file(self, instance_name, generate_format, present_format):
#         """Generates a single instance in the specified format.
#         Args:
#           instance_name: Name of instance to generate.
#           generate_format: Format of files to generate.
#           present_format: Format of corresponding files in fileset.

#         Returns:
#           None
#         """

#         if generate_format.upper() not in ['APX', 'TGF']:
#             raise ValueError("Can not generate {} instances".format(generate_format))
#         if present_format not in self.get_formats():
#             raise ValueError("Format {} not supported!".format(present_format))


#         if generate_format not in self.get_formats():
#             self.supported_formats += "," + generate_format
        
#         present_instance_path = os.path.join(self.benchmark_path, "{}.{}".format(instance_name, present_format))
#         with open(present_instance_path) as present_file:
#             present_file_content = present_file.read()
#         if generate_format.upper() == 'APX':
#             generate_file_content = self.__parse_apx_from_tgf(present_file_content)
#         elif generate_format.upper() == 'TGF':
#             generate_file_content = self.__parse_tgf_from_apx(present_file_content)

#         generate_file_name = "{}.{}".format(instance_name, generate_format)
#         generate_file = open(os.path.join(self.benchmark_path, generate_file_name), 'a')
#         generate_file.write(generate_file_content)
#         generate_file.close()

    

#     @staticmethod
#     def __parse_apx_from_tgf(file_content):
#         """Parse tgf to apx format.

#         Args:
#           file_content: File content in tgf format.

#         Returns:
#           String: File content in apx format.

#         """
#         arg_attacks = file_content.split("#\n")
#         arguments = arg_attacks[0].split('\n')
#         attacks = arg_attacks[1].split('\n')

#         apx_args = ''
#         apx_attacks = ''

#         for arg in arguments:
#             if arg:
#                 apx_args += 'arg({}).\n'.format(arg)

#         for attack in attacks:
#             if attack:
#                 apx_attacks += 'att({},{}).\n'.format(*attack.split(" "))

#         apx_instance_string = apx_args + apx_attacks

#         return apx_instance_string.rstrip()

#     @staticmethod
#     def __parse_tgf_from_apx(file_content):
#         """Parse apx to tgf format.

#         Args:
#           file_content: file content in apx format.

#         Returns:
#           String: file content in tgf format.
#         """
#         tgf_arguments = ''
#         tgf_attacks = ''

#         for line in file_content.splitlines():
#             if 'arg' in line:
#                 tgf_arguments += line[line.find("(") + 1:line.find(")")] + '\n'
#             if 'att' in line:
#                 tgf_attacks += line[line.find("(") + 1:line.find(")")].replace(',', ' ') + '\n'

#         tgf_file_content = tgf_arguments + '#\n' + tgf_attacks

#         return tgf_file_content.rstrip()

#     @staticmethod
#     def __get_random_argument(file_content, format):
#         """Return random argument from a file

#         Args:
#           file_content: Content of file
#           format: Format of the file

#         Returns:
#           String: Random argument from the file content
#         """
#         arguments = list()
#         if format.upper() == 'APX':
#             for line in file_content.splitlines():
#                 if 'arg' in line:
#                     arguments.append(line[line.find("(") + 1:line.find(")")])
#             return random.choice(arguments)

#         if format.upper() == 'TGF':
#             arg_attacks = file_content.split("#\n")
#             arguments = arg_attacks[0].split('\n')
#             return random.choice(arguments)

#     def generate_single_argument_file(self, instance_name, present_format, extension="arg"):
#         """Creates a single argument file with a random argument.
#         Args:
#           instance_name: Name of file to generate.
#           present_format: Format of existing file.
#           extension: Extension of argument file.
#         Returns:

#         """

#         if present_format not in self.get_formats():
#             raise ValueError("Format {} not supported!".format(present_format))

#         present_instance_path = os.path.join(self.benchmark_path, "{}.{}".format(instance_name, present_format))
#         with open(present_instance_path) as present_file:
#             present_file_content = present_file.read()

#         random_argument = self.__get_random_argument(present_file_content, present_format)

#         argument_file_name = "{}.{}".format(instance_name, extension)
#         argument_file = open(os.path.join(self.benchmark_path, argument_file_name), 'a')
#         argument_file.write(random_argument)
#         argument_file.close()
    
#     def generate_argument_files(self, extension="arg"):
#         """Generate argument file with random arguments from existing files.
#             Args:
#             extension: Extension of argument file.

#             Returns:

#         """
#         num_generated_files = 0
#         with click.progressbar(self.get_instances(self.get_formats()[0]),
#                                 label="Generating argument files:") as files:

#              for file in files:
#                 name_extension = file.split(".")
#                 if "{}.{}".format(name_extension[0], extension) not in os.listdir(self.benchmark_path):
#                     self.generate_single_argument_file(name_extension[0], name_extension[1], extension)
#                     num_generated_files += 1
#         print("{} argument files generated.".format(num_generated_files))

#     def is_complete(self):
#         """Check the fileset for completeness.
#         Returns:
#           bool: True if the fileset is complete, false otherwise.
#         """
#         num_formats = len(self.get_formats())
#         if num_formats > 1:
#             for i in range(0, num_formats):
#                 current_format = self.get_formats()[i]
#                 current_format_instances = self.get_instances(current_format)
#                 for j in range(i + 1, num_formats):
#                     compare_format = self.get_formats()[j]
#                     compare_format_instances = self.get_instances(compare_format)
#                     if not (len(current_format_instances) == len(compare_format_instances)):
#                         print("Number of files with different formats is not equal.")
#                         return False
#                     else:
#                         str_current_format_instances = ",".join(current_format_instances).replace("." + current_format,
#                                                                                                   "")
#                         str_compare_format_instances = ",".join(compare_format_instances).replace("." + compare_format,
#                                                                                                   "")
#                         if str_compare_format_instances != str_current_format_instances:
#                             print("Files of different formats differ")
#                             return False
#             return True
#         else:
#             True  # FileSet with just one supported format is complete

#     def get_missing_files(self):
#         """Return a dictionary of missing instances for each format.
#         Returns:
#           dict: Mapping {format:[instances]}
#         """
#         missing_instances = {}
#         num_formats = len(self.get_formats())
#         for i in range(0, num_formats):
#             current_format = self.get_formats()[i]
#             missing_instances[current_format] = set()
#             current_format_instances = set(","
#                                            .join(self.get_instances(current_format))
#                                            .replace("." + current_format, "")
#                                            .split(","))
#             for j in range(0, num_formats):
#                 compare_format = self.get_formats()[j]
#                 compare_format_instances = set(","
#                                                .join(self.get_instances(compare_format))
#                                                .replace("." + compare_format, "")
#                                                .split(","))
#                 print(compare_format_instances)
#                 current_missing = compare_format_instances.difference(current_format_instances)
#                 missing_instances[current_format].update(current_missing)
#         return missing_instances
    
#     def generate_missing_files(self):
#         """Generate all missing files for each format.
#         Returns:
#         """
#         missing_files = self.get_missing_files()
#         num_generated = 0
#         with click.progressbar(missing_files.items(),
#                                label="Generating missing files") as missing_files_items:
#             for missing_format, missing in missing_files_items:
#                 for instance in missing:
#                     for present_format in self.get_formats():
#                         if "{}.{}".format(instance, present_format) in self.get_instances(present_format):
#                             self.generate_single_file(instance, missing_format, present_format)
#                             num_generated += 1
#                             break  # Instance was created, so we don't have to check the other formats
#         print("{} files generated.".format(num_generated))


# class Result(Base):
#     __tablename__ = "results"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     tag = Column(String, nullable=False)
#     solver_id = Column(Integer, ForeignKey('solvers.id'))
#     benchmark_id = Column(Integer, ForeignKey('benchmarks.id'))
#     task_id = Column(Integer,ForeignKey('tasks.id'))
#     instance = Column(String,nullable=False)
#     cut_off = Column(Integer, nullable=False)
#     timed_out = Column(Boolean, nullable=False)
#     exit_with_error = Column(Boolean, nullable=False)
#     error_code = Column(Integer,nullable=True )
#     runtime = Column(Float,nullable=True)
#     result = Column(String,nullable=True)
#     additional_argument= Column(String, nullable=True)
#     benchmark = relationship("Benchmark", back_populates="results")
#     solver = relationship("Solver", back_populates="results")
#     task = relationship("Task", back_populates="results")
