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
    # try:
    #     request.urlretrieve(url,save_to)
    # except Exception as e:
    #     print(f"Something went wrong downloading the benchmark:\n{e}")
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
        print(f'{save_unzipped=}\n{instances_19=}\n{instances_21}')
        unpack_lzma(save_unzipped, instances_19)
        unpack_lzma(save_unzipped,instances_21)
        try:
            shutil.rmtree(instances_19)
            shutil.rmtree(instances_21)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    echo('finished.')

    return save_unzipped

def unpack_lzma(save_unzipped, instances_19):
    for f in os.listdir(instances_19):
        full_path = os.path.join(instances_19,f)
        with lzma.open(full_path) as f:
            file_content = f.read()


        decoded_file_content = file_content.decode("UTF-8")
        instance_file_name_unpacked = os.path.join(save_unzipped,os.path.basename(full_path.rstrip(".lzma")))
        print(instance_file_name_unpacked)
        with open(instance_file_name_unpacked,'w') as i:
            i.write(decoded_file_content)



