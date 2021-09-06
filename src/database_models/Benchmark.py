import os
import click
import random

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pathlib import Path
from sqlalchemy.sql.expression import null


from src.database_models.Base import Base, Supported_Tasks

class Benchmark(Base):
    __tablename__ = "benchmarks"
    id = Column(Integer, primary_key=True)
    benchmark_name = Column(String, nullable=False)
    benchmark_path = Column(String, nullable=False)
    graph_type = Column(String, nullable=True)
    format_instances = Column(String, nullable=False)
    hardness = Column(String, nullable=True)
    extension_arg_files = Column(String, nullable=False)
    benchmark_competition = Column(String, nullable=True)
    results = relationship("Result")

    def get_instances(self,extension,without_extension=False,full_path=False):
        instances = []
        for instance in os.listdir(self.benchmark_path):
            if instance.endswith(extension):
                if without_extension:
                    instances.append(Path(instance).stem)
                elif full_path:
                    instances.append(os.path.join(self.benchmark_path,instance))
                else:
                    instances.append(instance)
        return sorted(instances)
    
    def get_argument_files(self):
        return self.get_instances(self.extension_arg_files)
    
    def generate_additional_argument_lookup(self, format):
        lookup = {}
        argument_files = self.get_argument_files()
        for file in argument_files:
            argument_file_path = os.path.join(self.benchmark_path, file)

            try:
                with open(argument_file_path, 'r') as af:
                    argument_param = af.read().replace('\n', '')
            except IOError as err:
                print(err)
            
            instance_name = Path(file).stem
            lookup[instance_name] = argument_param


        instances = self.get_instances(format,without_extension=True)

        return {k: lookup[k] for k in instances} # Select only the argument files of instances that are present in the solver format



    def get_formats(self):
        return self.format_instances.split(',')

    def generate_files(self, generate_format, present_format=''):
        """Generate files in specified format

        Args:
          generate_format: Format of files to generate
          present_format: Format of corresponding files in fileset

        Returns:
          None

        """
        if not present_format:
            present_format = self.get_formats()[0]

        if present_format not in self.get_formats():
            raise ValueError("Format {} not supported!".format(present_format))

        if generate_format.upper() not in ['APX', 'TGF']:
            raise ValueError("Can not generate {} instances".format(generate_format))

        if generate_format not in self.get_formats():
            self.format_instances += "," + generate_format
        num_generated = 0
        
        with click.progressbar(self.get_instances(present_format),
                               label="Generating {} files:".format(generate_format)) as instance_progress:
            for instance in instance_progress:
                instance_name = Path(instance).stem
                generate_instance_name = instance_name + "." + generate_format
                if generate_instance_name not in self.get_instances(generate_format):
                    self.generate_single_file(instance_name, generate_format, present_format)
                    num_generated += 1

        print("{} files generated.".format(num_generated))
    
    def generate_single_file(self, instance_name, generate_format, present_format):
        """Generates a single instance in the specified format.
        Args:
          instance_name: Name of instance to generate.
          generate_format: Format of files to generate.
          present_format: Format of corresponding files in fileset.

        Returns:
          None
        """

        if generate_format.upper() not in ['APX', 'TGF']:
            raise ValueError("Can not generate {} instances".format(generate_format))
        if present_format not in self.get_formats():
            raise ValueError("Format {} not supported!".format(present_format))


        if generate_format not in self.get_formats():
            self.supported_formats += "," + generate_format
        
        present_instance_path = os.path.join(self.benchmark_path, "{}.{}".format(instance_name, present_format))
        with open(present_instance_path) as present_file:
            present_file_content = present_file.read()
        if generate_format.upper() == 'APX':
            generate_file_content = self.__parse_apx_from_tgf(present_file_content)
        elif generate_format.upper() == 'TGF':
            generate_file_content = self.__parse_tgf_from_apx(present_file_content)

        generate_file_name = "{}.{}".format(instance_name, generate_format)
        generate_file = open(os.path.join(self.benchmark_path, generate_file_name), 'a')
        generate_file.write(generate_file_content)
        generate_file.close()

    

    @staticmethod
    def __parse_apx_from_tgf(file_content):
        """Parse tgf to apx format.

        Args:
          file_content: File content in tgf format.

        Returns:
          String: File content in apx format.

        """
        arg_attacks = file_content.split("#\n")
        arguments = arg_attacks[0].split('\n')
        attacks = arg_attacks[1].split('\n')

        apx_args = ''
        apx_attacks = ''

        for arg in arguments:
            if arg:
                apx_args += 'arg({}).\n'.format(arg)

        for attack in attacks:
            if attack:
                apx_attacks += 'att({},{}).\n'.format(*attack.split(" "))

        apx_instance_string = apx_args + apx_attacks

        return apx_instance_string.rstrip()

    @staticmethod
    def __parse_tgf_from_apx(file_content):
        """Parse apx to tgf format.

        Args:
          file_content: file content in apx format.

        Returns:
          String: file content in tgf format.
        """
        tgf_arguments = ''
        tgf_attacks = ''

        for line in file_content.splitlines():
            if 'arg' in line:
                tgf_arguments += line[line.find("(") + 1:line.find(")")] + '\n'
            if 'att' in line:
                tgf_attacks += line[line.find("(") + 1:line.find(")")].replace(',', ' ') + '\n'

        tgf_file_content = tgf_arguments + '#\n' + tgf_attacks

        return tgf_file_content.rstrip()

    @staticmethod
    def __get_random_argument(file_content, format):
        """Return random argument from a file

        Args:
          file_content: Content of file
          format: Format of the file

        Returns:
          String: Random argument from the file content
        """
        arguments = list()
        if format.upper() == 'APX':
            for line in file_content.splitlines():
                if 'arg' in line:
                    arguments.append(line[line.find("(") + 1:line.find(")")])
            return random.choice(arguments)

        if format.upper() == 'TGF':
            arg_attacks = file_content.split("#\n")
            arguments = arg_attacks[0].split('\n')
            return random.choice(arguments)

    def generate_single_argument_file(self, instance_name, present_format, extension="arg"):
        """Creates a single argument file with a random argument.
        Args:
          instance_name: Name of file to generate.
          present_format: Format of existing file.
          extension: Extension of argument file.
        Returns:

        """

        if present_format not in self.get_formats():
            raise ValueError("Format {} not supported!".format(present_format))

        present_instance_path = os.path.join(self.benchmark_path, "{}.{}".format(instance_name, present_format))
        with open(present_instance_path) as present_file:
            present_file_content = present_file.read()

        random_argument = self.__get_random_argument(present_file_content, present_format)

        argument_file_name = "{}.{}".format(instance_name, extension)
        argument_file = open(os.path.join(self.benchmark_path, argument_file_name), 'a')
        argument_file.write(random_argument)
        argument_file.close()
    
    def generate_argument_files(self, extension="arg"):
        """Generate argument file with random arguments from existing files.
            Args:
            extension: Extension of argument file.

            Returns:

        """
        num_generated_files = 0
        with click.progressbar(self.get_instances(self.get_formats()[0]),
                                label="Generating argument files:") as files:

             for file in files:
                #0file_name = Path(file).stem
                #print(file_name)
                name_extension = os.path.splitext(file)
                if "{}.{}".format(name_extension[0], extension) not in os.listdir(self.benchmark_path):
                    self.generate_single_argument_file(name_extension[0], name_extension[1].strip("."), extension)
                    num_generated_files += 1
        print("{} argument files generated.".format(num_generated_files))

    def is_complete(self):
        """Check the fileset for completeness.
        Returns:
          bool: True if the fileset is complete, false otherwise.
        """
        num_formats = len(self.get_formats())
        if num_formats > 1:
            for i in range(0, num_formats):
                current_format = self.get_formats()[i]
                current_format_instances = self.get_instances(current_format)
                for j in range(i + 1, num_formats):
                    compare_format = self.get_formats()[j]
                    compare_format_instances = self.get_instances(compare_format)
                    if not (len(current_format_instances) == len(compare_format_instances)):
                        print("Number of files with different formats is not equal.")
                        return False
                    else:
                        str_current_format_instances = ",".join(current_format_instances).replace("." + current_format,
                                                                                                  "")
                        str_compare_format_instances = ",".join(compare_format_instances).replace("." + compare_format,
                                                                                                  "")
                        if str_compare_format_instances != str_current_format_instances:
                            print("Files of different formats differ")
                            return False
            return True
        else:
            True  # FileSet with just one supported format is complete

    def get_missing_files(self):
        """Return a dictionary of missing instances for each format.
        Returns:
          dict: Mapping {format:[instances]}
        """
        missing_instances = {}
        num_formats = len(self.get_formats())
        for i in range(0, num_formats):
            current_format = self.get_formats()[i]
            missing_instances[current_format] = set()
            current_format_instances = set(","
                                           .join(self.get_instances(current_format))
                                           .replace("." + current_format, "")
                                           .split(","))
            for j in range(0, num_formats):
                compare_format = self.get_formats()[j]
                compare_format_instances = set(","
                                               .join(self.get_instances(compare_format))
                                               .replace("." + compare_format, "")
                                               .split(","))
                print(compare_format_instances)
                current_missing = compare_format_instances.difference(current_format_instances)
                missing_instances[current_format].update(current_missing)
        return missing_instances
    
    def generate_missing_files(self):
        """Generate all missing files for each format.
        Returns:
        """
        missing_files = self.get_missing_files()
        num_generated = 0
        with click.progressbar(missing_files.items(),
                               label="Generating missing files") as missing_files_items:
            for missing_format, missing in missing_files_items:
                for instance in missing:
                    for present_format in self.get_formats():
                        if "{}.{}".format(instance, present_format) in self.get_instances(present_format):
                            self.generate_single_file(instance, missing_format, present_format)
                            num_generated += 1
                            break  # Instance was created, so we don't have to check the other formats
        print("{} files generated.".format(num_generated))
