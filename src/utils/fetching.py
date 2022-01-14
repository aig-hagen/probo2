"""Module to fetch benchmarks from web"""

from urllib import request

from markupsafe import re
from src.utils.utils import dispatch_on_value

import os
import zipfile
from click import echo


urls = {
    'ICCMA15': 'http://argumentationcompetition.org/2015/iccma2015_benchmarks.zip'
}

@dispatch_on_value
def fetch_benchmark(benchmark_name, save_to):
    print(f"Benchmark {benchmark_name} not found.")


def _fetch_from_url(url,save_to):
    try:
        request.urlretrieve(url,save_to)
    except Exception as e:
        print(f"Something went wrong downloading the benchmark:\n{e}")
    return save_to

@fetch_benchmark.register('ICCMA15')
def _fetch_iccma15_benchmark(benchmark_name, save_to):
    echo(f"Fetching benchmark {benchmark_name}...",nl=False)
    save_directory = _fetch_from_url(urls[benchmark_name],os.path.join(save_to,f'{benchmark_name}.zip'))
    echo("finished.")

    save_unzipped = os.path.join(save_to,benchmark_name)
    echo('Unzipping files...',nl=False)
    with zipfile.ZipFile(save_directory, 'r') as zip_ref:
        zip_ref.extractall(save_unzipped)
    echo('finished.')

    return save_unzipped


