from src.utils.options.CommandOptionInterface import CommandOptionInterface
from src.utils.definitions import DefaultInstanceFormats, DefaultQueryFormats, DefaultReferenceExtensions
from os import listdir
from pathlib import Path
import yaml
from tqdm import tqdm
from src.handler import benchmark_handler
import click
import re
from pathlib import Path

class EditSolverOptions(CommandOptionInterface):
    def __init__(self,id, name, version, path, tasks) -> None:
        super().__init__()
        self.id = id
        self.name = name
        self.version = version
        self.path = path
        self.tasks = tasks

        self.solver_infos = None

    def check(self):
        pass



    def print(self):
        print(yaml.dump(self.__dict__))

class AddSolverOptions(CommandOptionInterface):

    def __init__(self, name: str, path: str, version: str, fetch: bool,
                 format: tuple, tasks: list, yes: bool, no_check: bool) -> None:
        super().__init__()
        self.name = name
        self.path = path
        self.version = version
        self.fetch = fetch
        self.format = list(format) # click returns multiple options as a tuple, so it needs to be parsed
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


class ConvertBenchmarkOptions(CommandOptionInterface):
    """
    Represents options for converting a benchmark.
    """

    def __init__(self, id:str, benchmark_name: str, save_to: str, formats: list, extension_query_argument: str, add:bool,convert_query_files) -> None:
        """
        Initialize a CommandOptions object.

        Args:
            id (int): The ID of the benchmark to convert.
            benchmark_name (str): The name of the new generated benchmark.
            save_to: The location to save the new generated benchmark.
            formats: Formats to convert benchmark to
            extension_query_argument: The extension of the query argument files.
            add: Add benchmark to database.

        Returns:
            None
        """
        super().__init__()
        self.id = id
        self.benchmark_name = benchmark_name
        self.save_to = save_to
        self.formats = formats
        self.extension_query_argument = extension_query_argument
        self.add = add
        self.convert_query_files = convert_query_files

    def check(self):
        """
        Placeholder method for checking the options.
        """
        pass

    def print(self):
        """
        Prints the options as YAML.
        """
        print(yaml.dump(self.__dict__))


class EditBenchmarkOptions(CommandOptionInterface):
    def __init__(self,id, name, format, path, ext_additional,dynamic_files) -> None:
        super().__init__()
        self.id = id
        self.name = name
        self.ext_additional = ext_additional
        self.path = path
        self.format = format
        self.dynamic_files = dynamic_files

    def check(self):
        pass



    def print(self):
        print(yaml.dump(self.__dict__))


class AddBenchmarkOptions(CommandOptionInterface):

    def __init__(self, name: str, path: str, format: list,
                 additional_extension: list, no_check: bool, generate: list,
                 random_arguments: bool, dynamic_files: bool, function: list,
                 yes: bool,references_path: str,extension_references: str, has_references=False) -> None:
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
        self.references_path = references_path
        self.extension_references = extension_references
        self.has_references = has_references

    def check(self):
        self._check_benchmark_name()
        existing_instance_formats = self._check_instance_formats()
        self._check_query_formats(existing_instance_formats=existing_instance_formats)
        self._check_references()
        self._check_references_for_naming_convention()

    def _check_references_for_naming_convention(self):

        if self.has_references and self.references_path is not None and self.extension_references is not None:
            references_files_list = []
            total_matched_file_names = []
            format_instance_lookup = dict()
            for instance_format in self.format:
                format_instance_lookup[instance_format] = list()
            for file in listdir(self.references_path):
                if file.endswith(f".{self.extension_references}"):
                    references_files_list.append(file)

            for instance in listdir(self.path):
                file_extension = Path(instance).suffix[1:]
                if file_extension in self.format:
                    format_instance_lookup[file_extension].append(instance)





            for instance_format in tqdm(self.format,desc='Checking naming convention of reference files'):
                regex = f'.*\..*\.{instance_format}\.{self.extension_references}'
                r = re.compile(regex)
                matched_file_names = list(filter(r.match, references_files_list)) # Read Note below
                total_matched_file_names.extend(matched_file_names)


            #TODO: dump file paths of reference files to lookup, check if for all files references are present

            not_matched_file_names = set(references_files_list).difference(total_matched_file_names)

            if not_matched_file_names:
                print(f'Some references files do not follow the naming convention <instance_name>.<task>.<instance_format>.<reference_extension>:\n{not_matched_file_names}')
                print('**NOTE: Files that do not follow the naming convention are ignored in the validation process!**')
                click.confirm('Continue?',default=True,abort=True)


    def _check_references(self):
        if not self.references_path:
            print('**No path to reference results specified. Searching for references in benchmark instances directory.**')
            existing_extensions_reference_path = benchmark_handler.get_unique_formats_from_path(self.path)
            if not self.extension_references:
                print(f'**No extension for references files found. Searching for default reference extensions: {DefaultReferenceExtensions.as_list()}**')
                supported_reference_extensions = existing_extensions_reference_path.intersection(
                    set(DefaultReferenceExtensions.as_list()))
                if not supported_reference_extensions:
                    print('**No supported reference extensions found. Please specify the reference extension via the --format option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                    print('**Validation via reference results disabled.**')
                    self.has_references = False
                else:
                    print(f'**Setting reference extension to <{list(supported_reference_extensions)[0]}>**\n')
                    self.has_references = True
                    self.references_path = self.path
                    self.extension_references = list(supported_reference_extensions)[0]
            else:
                if not self.extension_references in existing_extensions_reference_path:
                    print(f'**No reference files with specified extension {self.extension_references} found.**')
                    input_reference_extension = click.confirm('Do you want to specify a different extension for the reference files?',default=True)
                    if input_reference_extension:
                        extension = click.prompt('Please enter a valid extension', type=str)
                        if not extension in existing_extensions_reference_path:
                            print('**No supported reference extensions found. Please specify the reference extension via the --format option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                            print('**Validation via reference results disabled.**')
                            self.has_references = False
                            self.extension_references = None
                            self.references_path = None
                        else:
                            print(f'**Setting reference extension to <{extension}>**\n')
                            self.has_references = True
                            self.extension_references = extension
                            self.references_path = self.path
                    else:
                        print('**No supported reference extensions found. Please specify the reference extension via the -refext option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                        print('**Validation via reference results disabled.**')
                        self.has_references = False
                        self.extension_references = None
                        self.references_path = None
                else:
                    print(f'**Found references with extension <{self.extension_references}>. Setting references path to <{self.path}>')
                    self.has_references = True
                    self.references_path = self.path
        else:
            if not self.extension_references:
                print(f'**No extension for references files found. Searching for default reference extensions: {DefaultReferenceExtensions.as_list()}**')

                existing_extensions_reference_path = benchmark_handler.get_unique_formats_from_path(self.references_path)

                supported_reference_extensions = existing_extensions_reference_path.intersection(
                    set(DefaultReferenceExtensions.as_list()))
                if not supported_reference_extensions:
                    print('**No supported reference extensions found. Please specify the reference extension via the --format option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                    print('**Validation via reference results disabled.**')
                    self.has_references = False
                    self.references_path = None
                else:
                    print(f'**Setting reference extension to {list(supported_reference_extensions)[0]}**\n')
                    self.has_references = True
                    self.extension_references = list(supported_reference_extensions)[0]
            else:

                existing_extensions_reference_path = benchmark_handler.get_unique_formats_from_path(self.references_path)
                if not self.extension_references in existing_extensions_reference_path:
                    print(f'**No reference files with specified extension <{self.extension_references}> in direcotry <{self.references_path}> found.**')
                    input_reference_extension = click.confirm('Do you want to specify a different extension for the reference files?',default=True)
                    if input_reference_extension:
                        extension = click.prompt('Please enter a valid extension', type=str)
                        if not extension in existing_extensions_reference_path:
                            print('**No supported reference extensions found. Please specify the reference extension via the -refext option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                            print('**Validation via reference results disabled.**')
                            self.has_references = False
                            self.extension_references = None
                            self.references_path = None
                        else:
                            print(f'**Setting reference extension to <{extension}>**\n')
                            self.has_references = True
                            self.extension_references = extension
                    else:
                        print('**No supported reference extensions found. Please specify the reference extension via the -refext option or modify the default formats in src/utils/definitions.DefaultReferenceExtensions**')
                        print('**Validation via reference results disabled.**')
                        self.has_references = False
                        self.extension_references = None
                        self.references_path = None

                else:
                    print(f'**Found references with extension <{self.extension_references}>**')
                    self.has_references = True


    def _check_benchmark_name(self):
        from os.path import basename
        from pathlib import Path
        # Handle also extensions like .py or .sh
        if not self.name or self.name is None:
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
