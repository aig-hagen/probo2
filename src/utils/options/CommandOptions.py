from src.utils.options.CommandOptionInterface import CommandOptionInterface
from src.utils.definitions import DefaultInstanceFormats, DefaultQueryFormats
from os import listdir
from pathlib import Path
import yaml
from tqdm import tqdm
from src.utils import benchmark_handler
import click


class AddSolverOptions(CommandOptionInterface):

    def __init__(self, name: str, path: str, version: str, fetch: bool,
                 format: list, tasks: list, yes: bool, no_check: bool) -> None:
        super().__init__()
        self.name = name
        self.path = path
        self.version = version
        self.fetch = fetch
        self.format = format
        self.tasks = tasks
        self.yes = yes
        self.no_check = yes

    def check(self):

        if self.format is None:
            print(f'No format found. Format fetching enabled.')
            self.fetch = True

        if self.tasks is None:
            print(f'No tasks found. Task fetching enabled.')
            self.fetch = True

        if self.name is None:
            from os.path import basename
            from pathlib import Path
            # Handle also extensions like .py or .sh
            self.name = basename(Path(self.path).with_suffix(''))
            print(f'No name found. Name {self.name} derived from path.')

        if self.version is None:
            print(f'No version found. Setting default version to 1.0')
            self.version = 1.0

    def print(self):
        print(yaml.dump(self.__dict__))


class AddBenchmarkOptions(CommandOptionInterface):

    def __init__(self, name: str, path: str, format: list,
                 additional_extension: list, no_check: bool, generate: list,
                 random_arguments: bool, dynamic_files: bool, function: list,
                 yes: bool) -> None:
        super().__init__()
        self.name = name
        self.path = path
        self.format = format
        self.additional_extension = additional_extension
        self.no_check = no_check
        self.generate = generate
        self.random_arguments = random_arguments
        self.dynamic_files = dynamic_files
        self.function = function
        self.yes = yes

    def check(self):

        self._check_benchmark_name()
        existing_instance_formats = self._check_instance_formats()
        self._check_query_formats(existing_instance_formats=existing_instance_formats)
    
    def _check_benchmark_name(self):
        from os.path import basename
        from pathlib import Path
            # Handle also extensions like .py or .sh
        self.name = basename(Path(self.path).with_suffix(''))
        print(f'**No name found. Name {self.name} derived from path.**\n')


    def _check_instance_formats(self):
        """ Checks the instance formats. 
            If no format is specified, instances are scanned for the formats in definition.DefaultInstanceFormats.
            Formats are set to the intesection beetween found formats and default formats.
        """
        if not self.format or self.format is None:
            print(
                f'**No instance formats specified. Searching for instance formats {DefaultInstanceFormats.as_list()}**\n'
            )

            existing_instance_formats = benchmark_handler.get_unique_formats_from_path(
                self.path)

            supported_formats = existing_instance_formats.intersection(
                set(DefaultInstanceFormats.as_list()))
            if not supported_formats:
                print('**No supported formats found. Please specify the instance format via the --format option or modify the default formats in src/utils/definitions.DefaultInstanceFormats**\n')
            print(f'**Setting instance formats to {list(supported_formats)}**\n')
            self.format = list(supported_formats)
            return existing_instance_formats
        else:
            return None
    
    def _check_query_formats(self,existing_instance_formats=None):
        if not self.additional_extension or self.additional_extension is None:
            print(
                f'**No extensions for query arguments specified. Searching for extensions {DefaultQueryFormats.as_list()}**\n'
            )
            if not existing_instance_formats:
                existing_instance_formats = benchmark_handler.get_unique_formats_from_path(
                    self.path)

            supported_query_formats = existing_instance_formats.intersection(
                set(DefaultQueryFormats.as_list()))
            
            if not supported_query_formats:
                print(f'**No query files with { DefaultQueryFormats.as_list()} found.**\n')
                self.random_arguments = click.confirm(text=f'Do you want to create random query arguments with extension {DefaultQueryFormats.ARG.value}?',default=True)
                self.additional_extension = DefaultQueryFormats.ARG.value
            else:
                # Handels just a single extension for query arguments
                print(f'**Setting query format to {list(supported_query_formats)[0]}**\n') 
                self.additional_extension = list(supported_query_formats)[0]
        else:
            if isinstance(self.additional_extension, tuple):
                if len(self.additional_extension) > 1:
                    self.additional_extension = list(self.additional_extension)
                else:
                    self.additional_extension = self.additional_extension[0]
            elif not isinstance(self.additional_extension, str):
                print(
                f"**Type {type(self.additional_extension)} of additional extension is not valid.**\n"
                )
                exit()


      

    def print(self):
        print(yaml.dump(self.__dict__))
