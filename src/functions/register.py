def register(key, function, function_dict):
    if key in function_dict:
        raise KeyError('Key {} is already pre-defined.'.format(key))
    else:
        function_dict[key] = function

stat_dict = {}
def register_stat(key, function):
    register(key, function, stat_dict)

score_functions_dict = {}
def register_score_function(key, function):
    register(key, function, score_functions_dict)

benchmark_functions_dict = {}
def benchmark_functions_register(key, function):
    register(key, function, benchmark_functions_dict)

plot_dict = {}
def plot_functions_register(key, function):
     register(key, function, plot_dict)

print_functions_dict = {}
def print_functions_register(key,function):
    register(key, function, print_functions_dict)

table_export_functions_dict = {}

def table_export_functions_register(key,function):
    register(key, function, table_export_functions_dict)

archive_functions_dict = {}
def archive_functions_register(key, function):
    register(key, function, archive_functions_dict)


run_functions_dict = {}
def run_functions_register(key, function):
    register(key, function, run_functions_dict)

validation_functions_dict = {}
def validation_functions_register(key, function):
    register(key, function, validation_functions_dict)

plot_validation_functions_dict = {}
def plot_validation_functions_register(key, function):
    register(key, function, plot_validation_functions_dict)

print_validation_functions_dict = {}
def print_validation_functions_register(key, function):
    register(key, function, print_validation_functions_dict)


significance_functions_dict = {}
def significance_functions_register(key, function):
    register(key, function, significance_functions_dict)
