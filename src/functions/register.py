def register(key, function, function_dict):
    if key in function_dict:
        raise KeyError('Key {} is already pre-defined.'.format(key))
    else:
        function_dict[key] = function

stat_dict = {}
def register_stat(key, function):
    register(key, function, stat_dict)

benchmark_functions_dict = {}
def benchmark_functions_register(key, function):
    register(key, function, benchmark_functions_dict)