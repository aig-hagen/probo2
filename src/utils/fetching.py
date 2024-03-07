"""Module to fetch benchmarks from web"""

from importlib.resources import path
import shutil
from urllib import request
from src.utils.utils import dispatch_on_value

import os
from click import echo
from pathlib import Path
import lzma

@dispatch_on_value
def fetch_benchmark(benchmark_name, save_to, options):
    print(f"Benchmark {benchmark_name} not found.")


def _fetch_from_url(url,save_to):
    try:
        request.urlretrieve(url,save_to)
    except Exception as e:
        print(f"Something went wrong downloading the benchmark:\n{e}")
    return save_to

@fetch_benchmark.register('ICCMA15')
def _fetch_iccma15_benchmark(benchmark_name, save_to, options):
    echo(f"Fetching benchmark {benchmark_name}...",nl=False)
    url = options['url']
    archive_type = options['archive_type']
    save_directory = _fetch_from_url(url,os.path.join(save_to,f'{benchmark_name}.{archive_type}'))
    echo("finished.")

    save_unzipped = os.path.join(save_to,benchmark_name)
    echo('Unzipping files...',nl=False)
    shutil.unpack_archive(save_directory,save_unzipped)
    echo('finished.')

    return save_unzipped

def _fetch_benchmark(benchmark_name, save_to, options):
    """
    Fetches a benchmark from a given URL and saves it to the specified directory.

    Args:
        benchmark_name (str): The name of the benchmark.
        save_to (str): The directory path where the benchmark will be saved.
        options (dict): A dictionary containing the URL and archive type of the benchmark.

    Returns:
        str: The path to the unzipped benchmark.

    Raises:
        OSError: If there is an error while removing directories.

    """

    echo(f"Fetching benchmark {benchmark_name}...",nl=False)
    url = options['url']
    archive_type = options['archive_type']
    save_directory = _fetch_from_url(url,os.path.join(save_to,f'{benchmark_name}.{archive_type}'))
    echo("finished.")

    save_unzipped = os.path.join(save_to,benchmark_name)
    echo('Unzipping files...',nl=False)
    shutil.unpack_archive(save_directory,save_unzipped)

    if benchmark_name == 'ICCMA21':
        instances_19 = os.path.join(save_unzipped,"instances","2019")
        instances_21 = os.path.join(save_unzipped,"instances","2021")
        unpack_lzma(save_unzipped, instances_19)
        unpack_lzma(save_unzipped,instances_21)
        try:
            shutil.rmtree(instances_19)
            shutil.rmtree(instances_21)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    
    if benchmark_name == 'ICCMA23':
        save_unzipped = os.path.join(save_unzipped,'benchmarks','main')
        for filename in os.listdir(save_unzipped):
            if filename.endswith('af'):
            # Construct the old and new file paths
                old_file_path = os.path.join(save_unzipped, filename)
                new_file_path = os.path.join(save_unzipped, filename.rsplit('.', 1)[0] + '.' + 'i23')

                # Rename the file
                os.rename(old_file_path, new_file_path)


    echo('finished.')

    return save_unzipped

import os

def unpack_lzma(save_unzipped, instances_19):
    """
    Unpacks LZMA-compressed files in the specified directory and saves the uncompressed files in another directory.

    Args:
        save_unzipped (str): The directory path where the uncompressed files will be saved.
        instances_19 (str): The directory path where the LZMA-compressed files are located.

    Returns:
        None
    """
    for f in os.listdir(instances_19):
        full_path = os.path.join(instances_19,f)
        with lzma.open(full_path) as f:
            file_content = f.read()

        decoded_file_content = file_content.decode("UTF-8")
        instance_file_name_unpacked = os.path.join(save_unzipped,os.path.basename(full_path.rstrip(".lzma")))
        print(instance_file_name_unpacked)
        with open(instance_file_name_unpacked,'w') as i:
            i.write(decoded_file_content)



