import os
import click
import random
import numpy as np
from glob import glob
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pathlib import Path
from sqlalchemy.sql.expression import null
from itertools import chain
from src.database_models.Base import Base, Supported_Tasks
from functools import reduce

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
        instances = (chain.from_iterable(glob(os.path.join(x[0], f'*.{extension}')) for x in os.walk(self.benchmark_path)))
        return sorted(list(instances))

    def get_argument_files(self):
        return self.get_instances(self.extension_arg_files)

    def generate_additional_argument_lookup(self, format: str) -> dict:
        """[summary]

        Args:
            format ([type]): [description]

        Returns:
            [type]: [description]
        """
        lookup = {}
        argument_files = self.get_argument_files()
        for file in argument_files:
            try:
                with open(file, 'r') as af:
                    argument_param = af.read().replace('\n', '')
            except IOError as err:
                print(err)

            suffix_length = len(self.extension_arg_files) + 1 # +1 for dot
            instance_name = os.path.basename(file)[:-suffix_length]
            lookup[instance_name] = argument_param


        instances_paths = self.get_instances(format)
        instance_name = [Path(x).stem for x in instances_paths]
        return {k: lookup[k] for k in instance_name} # Select only the argument files of instances that are present in the solver format



    def get_formats(self) -> list:
        return self.format_instances.split(',')

    def generate_instances(self, generate_format, present_format=''):
        """Generate files in specified format

        Args:
          generate_format: Format of files to generate
          present_format: Format of corresponding files in fileset

        Returns:
          None

        """
        if not present_format:
            present_formats= self.get_formats()
            for form in present_formats:
                if form != generate_format:
                    present_format = form
                    break

        if not present_format:
            print("Please use other format to generate instances.")
            exit()

        if present_format not in self.get_formats():
            raise ValueError(f"Format {present_format} not supported!")

        if generate_format.upper() not in ['APX', 'TGF']:
            raise ValueError(f"Can not generate {generate_format} instances")

        _present_instances = self.get_instances(present_format)
        num_generated = 0
        with click.progressbar(_present_instances,
                               label=f"Generating {generate_format} files:") as present_instances:
            for present_instance in present_instances:
                file_name, file_extension = os.path.splitext(present_instance)
                generate_instance_path = f"{file_name}.{generate_format}"
                if not os.path.isfile(generate_instance_path):
                    self.gen_single_instance(present_instance, generate_instance_path, generate_format)
                    num_generated += 1
        print(f'{num_generated} .{generate_format} instances generated.')
        self.format_instances += f',{generate_format}'

    def gen_single_instance(self,present_instance_path, generate_instance_path, generate_format):
        with open(present_instance_path) as present_file:
            present_file_content = present_file.read()
        if generate_format.upper() == 'APX':
            generate_file_content = self.__parse_apx_from_tgf(present_file_content)
        elif generate_format.upper() == 'TGF':
            generate_file_content = self.__parse_tgf_from_apx(present_file_content)

        with open(generate_instance_path,'w') as generate_instance_file:
            generate_instance_file.write(generate_file_content)
        generate_instance_file.close()

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
                apx_args += f'arg({arg}).\n'

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
        if format.upper() == '.APX':
            for line in file_content.splitlines():
                if 'arg' in line:
                    arguments.append(line[line.find("(") + 1:line.find(")")])
            return random.choice(arguments)

        if format.upper() == '.TGF':
            arg_attacks = file_content.split("#\n")
            arguments = arg_attacks[0].split('\n')
            return random.choice(arguments)

    def generate_single_argument_file(self, present_instance_path, generate_instance_path,present_format):
        """Creates a single argument file with a random argument.
        Args:
          instance_name: Name of file to generate.
          present_format: Format of existing file.
          extension: Extension of argument file.
        Returns:

        """
        with open(present_instance_path) as present_file:
            present_file_content = present_file.read()
        random_argument = self.__get_random_argument(present_file_content, present_format)

        with open(generate_instance_path,'w') as argument_file:
            argument_file.write(random_argument)

        argument_file.close()

    def generate_argument_files(self, extension=None,to_generate=None):
        """Generate argument file with random arguments from existing files.
            Args:
            extension: Extension of argument file.

            Returns:

        """
        if not extension:
            extension = 'arg'
        if to_generate:
            present_format = self.get_formats()[0]
            _present_instances = [ f'{x}.{present_format}' for x in to_generate]
            arg_files_present = to_generate
        else:
            _present_instances = self.get_instances(self.get_formats()[0])
            arg_files_present = self.get_instances(extension)
        num_generated_files = 0
        with click.progressbar(_present_instances,
                                label="Generating argument files:") as present_instances:

             for instance in present_instances:
                present_file_name, present_file_extension = os.path.splitext(instance)
                generate_instance_path = f"{present_file_name}.{extension}"
                if generate_instance_path not in arg_files_present:
                    self.generate_single_argument_file(instance,generate_instance_path, present_file_extension)
                    num_generated_files += 1
        print(f"{num_generated_files} .{extension} files generated.")

    def _strip_extension_arg_files(self, instances):
        suffix_length = len(self.extension_arg_files) + 1 # +1 for dot
        return  [ instance[:-suffix_length] for instance in instances]

        #return [ instance.removesuffix(f'.{self.extension_arg_files}') for instance in instances]
    def strip_extension(self,instances):
        extensions_stripped = list()
        for instance in instances:
            instance_file_name, instance_file_extension = os.path.splitext(instance)
            extensions_stripped.append(instance_file_name)
        return sorted(extensions_stripped)

    def is_complete(self):
        num_formats = len(self.get_formats())
        if num_formats > 1:
            apx_instances = self.get_instances('apx')
            tgf_instances = self.get_instances('tgf')
            arg_instances = self.get_instances(self.extension_arg_files)

            apx_instances_names = np.array(self.strip_extension(apx_instances))
            tgf_instances_names = np.array(self.strip_extension(tgf_instances))
            arg_instances_names = np.array(self._strip_extension_arg_files(arg_instances))


            if apx_instances_names.size == tgf_instances_names.size == arg_instances_names.size:

                return np.logical_and( (apx_instances_names==tgf_instances_names).all(), (tgf_instances_names==arg_instances_names).all() )
            else:

                return False
        else:
            preset_format = self.get_formats()[0]

            present_instances = self.get_instances(preset_format)
            arg_instances = self.get_instances(self.extension_arg_files)
            present_instances_names =  np.array(self.strip_extension(present_instances))
            arg_instances_names = np.array(self._strip_extension_arg_files(arg_instances))

            if present_instances_names.size == arg_instances_names.size:

                return (present_instances_names==arg_instances_names).all()
            else:
                return False

    def get_missing_files_per_format(self):
        instance_formats = self.get_formats()
        missing_instances = dict()
        if len(instance_formats) > 1:
            apx_instances_path = set(self.strip_extension(self.get_instances('apx')))
            tgf_instances_path = set(self.strip_extension(self.get_instances('tgf')))
            apx_missing_path = tgf_instances_path.difference(apx_instances_path)
            tgf_missing_path = apx_instances_path.difference(tgf_instances_path)

            apx_missing_names = [(os.path.basename(x) + '.apx') for x in apx_missing_path]
            tgf_missing_names = [(os.path.basename(x) + '.tgf') for x in tgf_missing_path]
            missing_instances['apx'] = {'paths':apx_missing_path, 'names':apx_missing_names}
            missing_instances['tgf'] = {'paths':tgf_missing_path, 'names': tgf_missing_names}

            arg_instances_names = set(self._strip_extension_arg_files(self.get_instances(self.extension_arg_files)))
            present_instances = set.union(apx_instances_path,tgf_instances_path)
            arg_missing_path = present_instances.difference(arg_instances_names)
            arg_missing_names = [(os.path.basename(x) + f'.{self.extension_arg_files}') for x in arg_missing_path]
            missing_instances[self.extension_arg_files] = {'paths':arg_missing_path, 'names': arg_missing_names}
            return missing_instances
        else:
            arg_instances_names = set(self._strip_extension_arg_files(self.get_instances(self.extension_arg_files)))
            present_instances =  set(self.strip_extension(self.get_instances(instance_formats[0])))
            arg_missing_path = present_instances.difference(arg_instances_names)
            arg_missing_names = [(os.path.basename(x) + f'.{self.extension_arg_files}') for x in arg_missing_path]
            missing_instances[self.extension_arg_files] = {'paths':arg_missing_path, 'names': arg_missing_names}
            return missing_instances

    def generate_missing_files_with_format(self,missing_format,missing_paths,present_format):

        if missing_paths:
            num_generated = 0
            for instance in missing_paths:
                self.gen_single_instance(f'{instance}.{present_format}',f'{instance}.{missing_format}',missing_format)
                num_generated += 1
            print(f'{num_generated} .{missing_format} instances generated.')




    def generate_missing_files(self,missing_files: dict):
        """Generate all missing files for each format.
        Returns:
        """
        missing_formats = missing_files.keys()
        if 'apx' in missing_formats:
            apx_missing = missing_files['apx']['paths']
            self.generate_missing_files_with_format('apx',apx_missing,'tgf')
        if 'tgf' in missing_formats:
            tgf_missing = missing_files['tgf']['paths']
            self.generate_missing_files_with_format('tgf',tgf_missing,'apx')
        if self.extension_arg_files in missing_formats:
            arg_missing = missing_files[self.extension_arg_files]['paths']
            if arg_missing:
                self.generate_argument_files(extension=self.extension_arg_files,to_generate=arg_missing)

    def _print_missing(self,missing_files):
        print("The following files are missing:")
        for formats,missing_instances in missing_files.items():

            if missing_instances:
                num_missing = len(missing_instances['names'])


                print(f"Format: {formats}\n#Missing: {num_missing}\nInstances: {missing_instances['names']}\n\n")
            else:
                continue

    def check(self):
        if not self.is_complete():
            missing_files = self.get_missing_files_per_format()
            self._print_missing(missing_files)

            if click.confirm("Do you want to create the missing files?"):
                self.generate_missing_files(missing_files)
                if not self.is_complete():
                    exit("Something went wrong when generating the missing instances.")
            else:
                print("Missing files not created.")

