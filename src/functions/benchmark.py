"""Costum functions for benchmarks.
 Within this file you can implement functions to add additional attributes to the benchmarks.

"""

import src.functions.register as register
from src.utils import benchmark_handler

def get_instance_counts(benchmark: benchmark_handler.Benchmark) -> dict:
    """Get instance count per format.

    Args:
        benchmark_path (str): Absolute path to instances

    Returns:
        dict: Instance count per format with format as key and count as value.
    """
    instance_counts = {}
    for fo in benchmark.format:
        instances = benchmark_handler.get_instances(benchmark.path,fo)
        instance_counts[f'{fo}_count'] = len(instances)

    instance_counts[f"{benchmark.ext_additional}_count"] = len(benchmark_handler.get_instances(benchmark.path, benchmark.ext_additional))

    return instance_counts

def get_benchmark_size(benchmark: benchmark_handler.Benchmark) -> dict:
    size_counts = {}
    total_size = 0
    for fo in benchmark.format:
        instances = benchmark_handler.get_instances(benchmark.path,fo)

        size_counts[f'{fo}_size'] = benchmark_handler.get_file_size(instances)

    size_counts[f"{benchmark.ext_additional}_size"] = benchmark_handler.get_file_size((benchmark_handler.get_instances(benchmark.path, benchmark.ext_additional)))

    return size_counts

# Register functions
register.benchmark_functions_register('instance_count', get_instance_counts)
register.benchmark_functions_register('file_sizes', get_benchmark_size)

