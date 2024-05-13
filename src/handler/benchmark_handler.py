
import json
import shutil

import tabulate
import src.utils.definitions as definitions
import os
import pandas as pd
from itertools import chain
from glob import glob
import click
import numpy as np
import random
from pathlib import Path
from src.utils.options.CommandOptions import AddBenchmarkOptions, EditBenchmarkOptions, ConvertBenchmarkOptions
from tqdm import tqdm
from click import confirm

from dataclasses import dataclass

@dataclass
class Benchmark:
    name: str
    format: list
    ext_additional: str
    path: str
    dynamic_files: bool
    id: int
    has_references: bool
    references_path: str
    extension_references: str
    meta_data: dict

def edit_benchmark(options: EditBenchmarkOptions ):
    benchmark_infos = load_benchmark_by_identifier(options.id)
    if not benchmark_infos or benchmark_infos is None:
        print(f'No solver with id {options.id} found. Please run solvers command to get a list of solvers in database.')
        exit()
    benchmark_infos = benchmark_infos[0] # load_benchmarks returns a list,
    print('Changing values:')
    for attribute, value in options.__dict__.items():
        if attribute in benchmark_infos.keys() and value is not None:
            if attribute != 'id':
                print(f'+ {attribute}: {benchmark_infos[attribute]} -> {value} ')
            benchmark_infos[attribute] = value
    confirm('Apply changes?',abort=True,default=True)
    update_benchmark(benchmark_infos)
    print(f'Updated benchmark!')


def get_unique_formats_from_path(path: str):
    existing_formats = set()
    for instance in tqdm(os.listdir(path),desc='Scanning instance formats'):
        extension = Path(instance).suffix[1:]
        existing_formats.add(extension)
    return existing_formats



def add_benchmark(options: AddBenchmarkOptions):
    benchmark_info = {'name': options.name,
                      'path': options.path,
                      'format': list(options.format),
                      'ext_additional': options.additional_extension,
                      'references_path': options.references_path,
                      'extension_references': options.extension_references,
                      'has_references': options.has_references,
                      'dynamic_files': options.dynamic_files,
                      'meta_data': {},
                      'id': None,
                      }
    new_benchmark = Benchmark(**benchmark_info)
    if options.generate:
        generate_instances(new_benchmark, options.generate)
    if options.random_arguments:
        generate_argument_files(new_benchmark,extension=options.additional_extension)
    if not options.no_check:
        check(new_benchmark)

    if options.dynamic_files:
        check_dynamic(new_benchmark)

    if options.function:
        from src.functions import register
        from src.functions import benchmark
        for fun in options.function:
            new_benchmark.meta_data.update(register.benchmark_functions_dict[fun](new_benchmark))

    print_summary(new_benchmark)
    if not options.yes:
        click.confirm(
            "Are you sure you want to add this benchmark to the database?",
            abort=True,default=True)
    new_benchmark.id = _add_benchmark_to_database(new_benchmark)
    print(f"Benchmark {new_benchmark.name} added to database with ID: {new_benchmark.id}.")
    #logging.info(f"Benchmark {new_benchmark['name']} added to database with ID: {new_benchmark.id}.")

def update_benchmark(benchmark_infos):
    benchmarks = load_benchmark('all')
    for i,solver in enumerate(benchmarks):
        if solver['id'] == benchmark_infos['id']:
            benchmarks[i] = benchmark_infos
            break

    _update_benchmark_json(benchmarks)


def _update_benchmark_json(benchmarks: list):
    json_str = json.dumps(benchmarks, indent=2)
    with open(definitions.BENCHMARK_FILE_PATH, "w") as f:
        f.write(json_str)



def get_file_size(file, unit='MB'):
    if type(file) is list:
        total_size = 0
        for f in file:
            total_size += os.path.getsize(f)

    else:
        total_size =  os.path.getsize(file)
    if unit == 'MB':
        return  round(total_size  / float(1<<20))

def _add_benchmark_to_database(benchmark: Benchmark):
    if not os.path.exists(definitions.BENCHMARK_FILE_PATH) or (os.stat(definitions.BENCHMARK_FILE_PATH).st_size == 0):
        with open(definitions.BENCHMARK_FILE_PATH,'w') as benchmark_file:
            id = 1
            benchmark.id = id

            json.dump([benchmark.__dict__], benchmark_file,indent=2)
        return id
    else:
        with open(definitions.BENCHMARK_FILE_PATH,'r+') as benchmark_file:
            benchmark_data = json.load(benchmark_file)
            _df = pd.DataFrame(benchmark_data)
            id = int(_df.id.max() + 1)

            benchmark.id = id
            benchmark_data.append(benchmark.__dict__)

            benchmark_file.seek(0)
            json.dump(benchmark_data, benchmark_file,indent=2)
        return id

def print_summary(benchmark: Benchmark):
    print()
    print("**********BENCHMARK SUMMARY**********")
    for key,value in benchmark.__dict__.items():

        if key == 'format':
            print(f"Format: {json.dumps(benchmark.format,indent=4)}" )
        else:
            print(f'{str(key).capitalize()}: {value}')

    print(f'#Instances: {get_instances_count(benchmark.path,benchmark.format[0])}')
    print()

def print_benchmarks(extra_columns=None, tablefmt=None):
    columns = ['id','name','format']
    if extra_columns:
        columns.extend(extra_columns)
    if tablefmt is None:
        tablefmt = 'pretty'
    if os.path.exists(definitions.BENCHMARK_FILE_PATH)  and   os.stat(definitions.BENCHMARK_FILE_PATH).st_size != 0:
        benchmarks_df = pd.read_json(definitions.BENCHMARK_FILE_PATH)
        print(tabulate.tabulate(benchmarks_df[columns],headers='keys', tablefmt=tablefmt, showindex=False))
    else:
        print("No benchmarks found.")

def get_instances(benchmark_path,extension,without_extension=False,full_path=False):
    instances = (chain.from_iterable(glob(os.path.join(x[0], f'*.{extension}')) for x in os.walk(benchmark_path)))
    return sorted(list(instances))

def get_instances_count(benchmark_path,extension):
    return len(get_instances(benchmark_path=benchmark_path, extension=extension))

def get_argument_files(benchmark_info, arg_files_format=None):
    if arg_files_format is None:
        return get_instances(benchmark_info['path'], benchmark_info['ext_additional'])
    else:
        return get_instances(benchmark_info['path'], arg_files_format)

def get_dynamic_instances(benchmark: Benchmark,format):
    return get_instances(benchmark.path, f'{format}m')

def check_dynamic(benchmark: Benchmark):
    missing_files = {}
    is_missing = False
    for format in benchmark.format:

        _dynamic_instances = get_dynamic_instances(benchmark,format )
        _instances = get_instances(benchmark_path=benchmark.path, extension=format)
        _instances = set([ os.path.basename(i).replace(f'.{format}','') for i in _instances])
        if _dynamic_instances:

            print(_instances)
             # get file name and remove extension
            _dynamic_instances = set( [os.path.basename(i).replace(f'.{format}m','') for i in _dynamic_instances])

            print(_instances)
            print(_dynamic_instances)
            _diff = list(set.difference(_instances,_dynamic_instances))
            if _diff:
                missing_files[f'{format}m'] = _diff
                is_missing = True

        else:
            is_missing = True
            missing_files[f'{format}m'] = _instances
    print(missing_files)
    print(is_missing)

def generate_dynamic_file_lookup(benchmark: Benchmark):
    lookup = {}
    for _format in benchmark.format:
        lookup[_format] = {}
        dynamic_instances = get_dynamic_instances(benchmark, _format)
        dynamic_instances_names = [Path(x).stem for x in dynamic_instances]
        d_names_path = dict(zip(dynamic_instances_names,dynamic_instances))
        instances_paths = get_instances(benchmark.path, _format)
        instances_names = [Path(x).stem for x in instances_paths]
        names_path = dict(zip(instances_names,instances_paths))
        for instance_name, instances_path in names_path.items():
            if instance_name in d_names_path.keys():
                lookup[_format][instances_path] = d_names_path[instance_name]

    return lookup

def generate_additional_argument_lookup(benchmark_info, solver_supported_format=None) -> dict:
    """[summary]
    Args:
        format ([type]): [description]
    Returns:
        [type]: [description]
    """
    lookup = {}
    arg_files_format = None

    if isinstance(benchmark_info['ext_additional'],list) and len(benchmark_info['ext_additional']) > 1 and solver_supported_format is not None:

        for ext in benchmark_info['ext_additional']:
            if solver_supported_format in ext:
                arg_files_format = ext
                break
        if arg_files_format:
            argument_files = get_argument_files(benchmark_info,arg_files_format=arg_files_format)
        else:
            print(f"No matching argument files for format {format} found!")
            exit()


    else:
        argument_files = get_argument_files(benchmark_info)

    for file in argument_files:
        try:
            with open(file, 'r') as af:
                argument_param = af.read().replace('\n', '')
        except IOError as err:
            print(err)
        if arg_files_format is None:
            suffix_length = len(benchmark_info['ext_additional']) + 1 # +1 for dot
        else:
            suffix_length = len(arg_files_format) + 1 # +1 for dot

        instance_name = os.path.basename(file)[:-suffix_length]
        lookup[instance_name] = argument_param
    return lookup

def generate_instances(benchmark: Benchmark, generate_format, present_format='',save_to=None):
    """Generate files in specified format
    Args:
      generate_format: Format of files to generate
      present_format: Format of corresponding files in fileset
    Returns:
      None
    """
    if not present_format:
        present_formats=benchmark.format # BUG: Falls benchmark.format keine Liste ist
        for form in present_formats:
            if form != generate_format:
                present_format = form
                break
    if not present_format:
        print("Please use other format to generate instances.")
        exit()
    if present_format not in benchmark.format:
        raise ValueError(f"Format {present_format} not supported!")
    if generate_format.lower() not in definitions.DefaultInstanceFormats.as_list():
        raise ValueError(f"Can not generate {generate_format} instances")
    _present_instances = get_instances(benchmark.path,present_format)
    num_generated = 0
    with click.progressbar(_present_instances,
                           label=f"Generating {generate_format} files:") as present_instances:
        for present_instance in present_instances:
            if save_to is not None:
                # get filename from path without file extension
                base_name = os.path.basename(present_instance)
                file_name, _ = os.path.splitext(base_name)  # Split the base name into name and extension
                generate_instance_path = os.path.join(save_to,f'{file_name}.{generate_format}')
            else:
                file_path, file_extension = os.path.splitext(present_instance)
                generate_instance_path = f"{file_path}.{generate_format}"
            
            #print(generate_instance_path)
            if not os.path.isfile(generate_instance_path):
                gen_single_instance(present_instance, generate_instance_path, generate_format)
                num_generated += 1
    print(f'{num_generated} .{generate_format} instances generated.')
    benchmark.format.append(generate_format)



def gen_single_instance(present_instance_path, generate_instance_path, generate_format):
    present_file_extension = Path(present_instance_path).suffix[1:].upper()

    with open(present_instance_path) as present_file:
        present_file_content = present_file.read()
        if not present_file_extension in parse_functions_dict[generate_format.upper()].keys():
            print(f'Parsing from {present_file_extension} to {generate_format} not supported')
            exit()
    generate_file_content = parse_functions_dict[generate_format.upper()][present_file_extension](present_file_content)
    with open(generate_instance_path,'w') as generate_instance_file:
        generate_instance_file.write(generate_file_content)
    generate_instance_file.close()
    return generate_file_content


def generate_mappings_argument_to_integer_mapping(benchmark: Benchmark, save_to=None):
    """Generate integer mappings for all instance in the benchmark"""

    if save_to is None:
        save_to = benchmark.path
    
    
    
    # Select tgf format when present in benchmark.format, if not use apx
    if 'tgf' in benchmark.format:
        present_format = 'tgf'
    elif 'apx' in benchmark.format:
        present_format = 'apx'
    

    # Get all instances in the benchmark and iterate over them
    instances = get_instances(benchmark.path, present_format)

    generated_mappings = dict()
    
    for instance in instances:
   
        # Extracting the filename
        file_name = os.path.splitext(os.path.basename(instance))[0]

        # Read file content
        with open(instance, 'r') as f:
            file_content = f.read()
        
        if present_format == 'tgf':
            arg_attacks = file_content.split("#\n")
            arguments = arg_attacks[0].rstrip().split('\n')
            attacks = arg_attacks[1].rstrip().split('\n')
            arg_to_id_map = {arg:str(i) for i,arg in enumerate(arguments,start=1)}
            
        elif present_format == 'apx':
            lines = file_content.rstrip().split('\n')
            num_arguments = int(lines[0].split(" ")[2])
            arg_to_id_map = {str(i): arg for i, arg in enumerate(lines[1:num_arguments+1], start=1)}
            # Save dictionary arg_to_id_map to file

        mapping_save_path = os.path.join(save_to, f'{file_name}.argmap')
        with open(os.path.join(save_to, f'{file_name}.argmap'), 'w') as f:
            json.dump(arg_to_id_map, f)
        generated_mappings[file_name] = mapping_save_path
    return generated_mappings
            
            
            
def __parse_i23_from_tgf(file_content):
    """Parse .tgf to .i23 format
    """
    arg_attacks = file_content.split("#\n")
    arguments = arg_attacks[0].rstrip().split('\n')
    attacks = arg_attacks[1].rstrip().split('\n')
    arg_to_id_map = {arg:str(i) for i,arg in enumerate(arguments,start=1)}
    file_header = f"p af {len(arg_to_id_map.keys())}\n"
    mapped_attacks = ""
    for att in attacks:
        splitted = att.split(" ")
        _mapped = (arg_to_id_map[splitted[0]],arg_to_id_map[splitted[1]] )
        mapped_attacks += f"{_mapped[0]} {_mapped[1]}\n"

    return file_header + mapped_attacks


def __parse_apx_from_i23(file_content):
    lines = file_content.rstrip().split('\n')
    num_arguments = int(lines[0].split(" ")[2])
    arguments = ""
    for i in range(1,num_arguments+1):
        arguments += f"arg({i}).\n"

    attacks = ""
    for att in lines[1:]:
        if "#" in att:
            continue

        splitted = att.split(" ")
        if len(splitted) == 2:
            attacks += f"att({splitted[0]},{splitted[1]}).\n"
        else:
            print(f"Line {att} skipped")
    return arguments + attacks

def __parse_tgf_from_i23(file_content):
    lines = file_content.rstrip().split('\n')
    num_arguments = int(lines[0].split(" ")[2])
    arguments = ""
    for i in range(1,num_arguments+1):
        arguments += f"{i}\n"
    arguments = arguments.rstrip() # remove trailing newline
    attacks = ""
    for att in lines[1:]:
        if "#" in att:
            continue


        attacks += f"{att}\n"


    return f"{arguments}\n#\n{attacks}"




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
def generate_single_argument_file(present_instance_path, generate_instance_path,present_format):
    """Creates a single argument file with a random argument.
    Args:
      instance_name: Name of file to generate.
      present_format: Format of existing file.
      extension: Extension of argument file.
    Returns:
    """
    with open(present_instance_path) as present_file:
        present_file_content = present_file.read()
    present_file.close()
    random_argument = __get_random_argument(present_file_content, present_format)
    if random_argument is None or random_argument == "":
        #print(f'{present_instance_path=}\n{random_argument=}')
        random_argument = __get_random_argument(present_file_content, present_format)
    with open(generate_instance_path,'w') as argument_file:
        argument_file.write(random_argument)
    argument_file.close()
def generate_argument_files(benchmark: Benchmark, extension=None,to_generate=None):
    """Generate argument file with random arguments from existing files.
        Args:
        extension: Extension of argument file.
        Returns:
    """
    if not extension:
        extension = 'arg'
    if to_generate:
        present_format = benchmark.format[0]
        _present_instances = [ f'{x}.{present_format}' for x in to_generate]
        arg_files_present = to_generate
    else:
        _present_instances = get_instances(benchmark.path, benchmark.format[0])
        arg_files_present = get_instances(benchmark.path, extension)
    num_generated_files = 0
    with click.progressbar(_present_instances,
                            label="Generating argument files:") as present_instances:
         for instance in present_instances:
            present_file_name, present_file_extension = os.path.splitext(instance)
            generate_instance_path = f"{present_file_name}.{extension}"
            if generate_instance_path not in arg_files_present:
                generate_single_argument_file(instance,generate_instance_path, present_file_extension)
                num_generated_files += 1
    print(f"{num_generated_files} .{extension} files generated.")
def _strip_extension_arg_files(benchmark: Benchmark,instances):
    suffix_length = len(benchmark.ext_additional) + 1 # +1 for dot
    return  [ instance[:-suffix_length] for instance in instances]


def strip_extension(benchmark: Benchmark,instances):
    extensions_stripped = list()
    for instance in instances:
        instance_file_name, instance_file_extension = os.path.splitext(instance)
        extensions_stripped.append(instance_file_name)
    return sorted(extensions_stripped)
def is_complete(benchmark: Benchmark):
    num_formats = len(benchmark.format)
    if num_formats > 1:
        # apx_instances = get_instances(benchmark.path,'apx')
        # tgf_instances = get_instances(benchmark.path,'tgf')
        # arg_instances = get_instances(benchmark.path,benchmark.ext_additional)
        # apx_instances_names = np.array(strip_extension(benchmark, apx_instances))
        # tgf_instances_names = np.array(strip_extension(benchmark, tgf_instances))
        # arg_instances_names = np.array(_strip_extension_arg_files(benchmark,arg_instances))
        # if apx_instances_names.size == tgf_instances_names.size == arg_instances_names.size:
        #     return np.logical_and( (apx_instances_names==tgf_instances_names).all(), (tgf_instances_names==arg_instances_names).all() )
        # else:
        #     return False

         # Get a list of all files in the specified path
        files_in_path = [f for f in os.listdir(benchmark.path) if os.path.isfile(os.path.join(benchmark.path, f))]

        # Create a dictionary to store filenames for each extension
        filenames_by_extension = {}

        # Iterate over each file in the path
        for file_name in files_in_path:
            # Split the file name and extension
            base_name, file_extension = os.path.splitext(file_name)

            # If the extension is in the provided list
            if file_extension[1:] in benchmark.format:
                # Add the base name to the dictionary for the corresponding extension
                if file_extension[1:] not in filenames_by_extension:
                    filenames_by_extension[file_extension[1:]] = set()
                filenames_by_extension[file_extension[1:]].add(base_name)

        # Check if the same filenames are present for all extensions
        return all(len(names) == 1 for names in filenames_by_extension.values())
    else:
        preset_format = benchmark.format[0]
        present_instances = get_instances(benchmark.path,preset_format)
        arg_instances = get_instances(benchmark.path,benchmark.ext_additional)
        present_instances_names =  np.array(strip_extension(benchmark, present_instances))
        arg_instances_names = np.array(_strip_extension_arg_files(benchmark, arg_instances))
        if present_instances_names.size == arg_instances_names.size:
            return (present_instances_names==arg_instances_names).all()
        else:
            return False
def get_missing_files_per_format(benchmark: Benchmark):
    instance_formats = benchmark.format
    missing_instances = dict()
    if len(instance_formats) > 1:
        apx_instances_path = set(strip_extension(benchmark, get_instances(benchmark.path,'apx')))
        tgf_instances_path = set(strip_extension(benchmark, get_instances(benchmark.path,'tgf')))
        apx_missing_path = tgf_instances_path.difference(apx_instances_path)
        tgf_missing_path = apx_instances_path.difference(tgf_instances_path)
        apx_missing_names = [(os.path.basename(x) + '.apx') for x in apx_missing_path]
        tgf_missing_names = [(os.path.basename(x) + '.tgf') for x in tgf_missing_path]
        missing_instances['apx'] = {'paths':apx_missing_path, 'names':apx_missing_names}
        missing_instances['tgf'] = {'paths':tgf_missing_path, 'names': tgf_missing_names}
        arg_instances_names = set(_strip_extension_arg_files(benchmark, get_instances(benchmark.path,benchmark.ext_additional)))
        present_instances = set.union(apx_instances_path,tgf_instances_path)
        arg_missing_path = present_instances.difference(arg_instances_names)
        arg_missing_names = [(os.path.basename(x) + f".{ benchmark.ext_additional }") for x in arg_missing_path]
        missing_instances[benchmark.ext_additional] = {'paths':arg_missing_path, 'names': arg_missing_names}
        return missing_instances
    else:
        arg_instances_names = set(_strip_extension_arg_files(benchmark,get_instances(benchmark.path,benchmark.ext_additional)))
        present_instances =  set(strip_extension(benchmark,get_instances(benchmark.path,benchmark.format[0])))
        arg_missing_path = present_instances.difference(arg_instances_names)
        arg_missing_names = [(os.path.basename(x) + f".{benchmark.ext_additional}") for x in arg_missing_path]
        missing_instances[benchmark.ext_additional] = {'paths':arg_missing_path, 'names': arg_missing_names}
        return missing_instances
def generate_missing_files_with_format(missing_format,missing_paths,present_format):
    if missing_paths:
        num_generated = 0
        for instance in missing_paths:
            gen_single_instance(f'{instance}.{present_format}',f'{instance}.{missing_format}',missing_format)
            num_generated += 1
        print(f'{num_generated} .{missing_format} instances generated.')
def generate_missing_files(benchmark: Benchmark,missing_files: dict):
    """Generate all missing files for each format.
    Returns:
    """
    missing_formats = missing_files.keys()
    if 'apx' in missing_formats:
        apx_missing = missing_files['apx']['paths']
        generate_missing_files_with_format('apx',apx_missing,'tgf')
    if 'tgf' in missing_formats:
        tgf_missing = missing_files['tgf']['paths']
        generate_missing_files_with_format('tgf',tgf_missing,'apx')
    if benchmark.ext_additional in missing_formats:
        arg_missing = missing_files[benchmark.ext_additional]['paths']
        if arg_missing:
            generate_argument_files(benchmark, extension=benchmark.ext_additional,to_generate=arg_missing)
def _print_missing(missing_files):
    print("The following files are missing:")
    for formats,missing_instances in missing_files.items():
        if missing_instances:
            num_missing = len(missing_instances['names'])
            print(f"Format: {formats}\n#Missing: {num_missing}\nInstances: {missing_instances['names']}\n\n")
        else:
            continue
def check(benchmark: Benchmark):

    if not is_complete(benchmark):
        missing_files = get_missing_files_per_format(benchmark)
        _print_missing(missing_files)
        if click.confirm("Do you want to create the missing files?", default=True):
            generate_missing_files(benchmark, missing_files)
            if not is_complete(benchmark):
                exit("Something went wrong when generating the missing instances.")
        else:
            print("Missing files not created.")

def load_benchmark_df(columns=None):
    if columns is None:
        return pd.read_json(definitions.BENCHMARK_FILE_PATH)
    else:
        return pd.read_json(definitions.BENCHMARK_FILE_PATH)[columns]

def load_benchmark_json():
    with open(definitions.BENCHMARK_FILE_PATH,'r+') as benchmark_file:
            benchmark_data = json.load(benchmark_file)
    return benchmark_data

def load_benchmark(identifiers):

    if identifiers == 'all':
        return load_benchmark_json()
    else:
        return load_benchmark_by_identifier(identifiers)


def load_benchmark_by_identifier(identifier: list) -> list:
    """Load solvers by name or id

    Args:
        identifier (list): _description_

    Returns:
        list: _description_
    """
    if not isinstance(identifier,list):
        identifier = [identifier]
    benchmark_json = load_benchmark_json()
    benchmark_list = []
    for benchmark in benchmark_json:
        if benchmark['name'] in identifier:
            benchmark_list.append(benchmark)
        elif benchmark['id'] in identifier or str(benchmark['id']) in identifier:
            benchmark_list.append(benchmark)
    return benchmark_list


def _update_benchmark_json(benchmarks: list):
    json_str = json.dumps(benchmarks, indent=2)

    with open(definitions.BENCHMARK_FILE_PATH, "w") as f:
        f.write(json_str)

def delete_benchmark(id):
    benchmarks = load_benchmark('all')
    deleted = False
    if id.isdigit():
        id = int(id)
    for b in benchmarks:
        if b['id'] == id or b['name'] == id:
            deleted = True
            benchmarks.remove(b)
    if deleted:
        _update_benchmark_json(benchmarks)
        print(f"Benchmark {id} deleted")
    else:
        print("Benchmark not found.")




def delete_all_benchmarks():
    with open(definitions.BENCHMARK_FILE_PATH,"w") as f:
        f.write("")


def convert_benchmark(options: ConvertBenchmarkOptions):
    """
    Converts benchmark instances to specified formats and optionally converts argument files.

    Parameters:
    options: An instance of ConvertBenchmarkOptions containing user-specified options including
    benchmark identifier, target formats, and save locations.
    """

    present_benchmark = Benchmark(**load_benchmark_by_identifier(options.id)[0])

    if not present_benchmark:
        raise ValueError(f'No benchmark with id {options.id} found. Please run benchmarks command to get a list of benchmarks in the database.')
    # Check if present benchmark is not empty
    
    for format in options.formats:
    # Check if format is in the default formats
        if format not in definitions.DefaultInstanceFormats.as_list():
            print(f"Format {format} not supported!")
            continue
        
    # Check if a name in the options is set for the new benchmark, if not set the name to the old name + _{format}
        options.benchmark_name = options.benchmark_name or f"{present_benchmark.name}_{format}"
    # Generate a directory at options.save_to for the new benchmark to save the new instances
        converted_benchmark_path = os.path.join(options.save_to,options.benchmark_name)
        os.makedirs(converted_benchmark_path,exist_ok=True)
    
    # Generated new instances with specified format and save them to the new directory
        generate_instances(present_benchmark, format,save_to=converted_benchmark_path)

    # Convert arg file
    new_extension = None
    if not options.skip_args:
        if format == 'i23':
            new_extension='i23.arg'
            print(f'Generating query argument files...',end='')
            list_of_mapping_files = generate_mappings_argument_to_integer_mapping(present_benchmark, save_to=converted_benchmark_path)
            convert_arg_files(list_of_mapping_files,present_benchmark.path,present_benchmark.ext_additional,save_to=converted_benchmark_path,new_extension=new_extension)
            print('Done!')
    
    converted_benchmark = Benchmark(name=options.benchmark_name,
                                    format=['i23'],
                                    ext_additional=new_extension,
                                    path=converted_benchmark_path,
                                    dynamic_files=present_benchmark.dynamic_files,
                                    id=None,
                                    has_references=False,
                                    references_path=None,
                                    extension_references=None,
                                    meta_data={'info': f'This benchmark was converted from {present_benchmark.name} with id {present_benchmark.id}'})
    return converted_benchmark
            

def convert_arg_files(arg_id_map_paths, benchmark_path, argument_file_extension, save_to, new_extension):
    # Ensure the save_to directory exists
    os.makedirs(save_to, exist_ok=True)

    generated_files = []

    benchmark_files = get_instances(benchmark_path=benchmark_path,extension=argument_file_extension)
    
    # List all files in the benchmark_path with the specified extension
    for filename in benchmark_files:
        
        print(f'{filename=}')
        print(f'{argument_file_extension=}')
        if filename.endswith(f".{argument_file_extension}"):

            
            raw_file_name = extract_filename(filename,argument_file_extension)
            print(f'{raw_file_name=}')
            if raw_file_name in arg_id_map_paths.keys():
             # Load the mapping from the specified mapping file
                mapping_file_path = arg_id_map_paths[raw_file_name]
                with open(mapping_file_path, 'r') as map_file:
                    # Assuming the mapping file is in JSON format
                    mapping = json.load(map_file)
            # Construct full file path
                file_path = os.path.join(benchmark_path, filename)
            
            # Read the file contents
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                if argument_file_extension == 'i23' and (new_extension =='tgf' or new_extension =='apx'):
                    mapping = {value: key for key, value in mapping.items()}
                
            # Process each line to replace argument names with IDs
                processed_lines = [str(mapping[line.strip()]) + '\n' for line in lines if line.strip() in mapping]
                
                # Construct the new filename with extension
                new_filename = f'{raw_file_name}.{new_extension}'
                new_file_path = os.path.join(save_to, new_filename)
                print(f'{new_file_path=}')
                # Write the processed lines to the new file
                with open(new_file_path, 'w') as new_file:
                    new_file.writelines(processed_lines)
                
                generated_files.append(new_file_path)
    return generated_files


       
    
        
            
    
    
    #     if options.to_format not in benchmark.format:
    #         if options.to_format.lower() in definitions.DefaultInstanceFormats.as_list():
    #             generate_instances(present_benchmark, options.to_format)
    #         else:
    #             print(f"Format {options.to_format} not supported!")
    #     else:
    #         print(f"Benchmark already in {options.to_format} format.")
    # pass

parse_functions_dict =  { "APX":{"TGF": __parse_apx_from_tgf, "I23": __parse_apx_from_i23},"I23": {"TGF": __parse_i23_from_tgf},"TGF": {'APX': __parse_tgf_from_apx ,'I23': __parse_tgf_from_i23} }

def extract_filename(path, extension):
    import os
    base_name = os.path.basename(path)
    extension_with_dot = '.' + extension if not extension.startswith('.') else extension
    if base_name.endswith(extension_with_dot):
        extension_start_index = base_name.rfind(extension_with_dot)
        raw_filename = base_name[:extension_start_index]
    else:
        raw_filename = base_name
    return raw_filename
