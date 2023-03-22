
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

post_hoc_table_export_functions_dict = {}
def post_hoc_table_export_functions_register(key,function):
    register(key, function, post_hoc_table_export_functions_dict)

validation_table_export_functions_dict = {}
def validation_table_export_functions_register(key,function):
    register(key, function, validation_table_export_functions_dict)

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


parametric_significance_functions_dict = {}
def parametric_significance_functions_register(key, function):
    register(key, function, parametric_significance_functions_dict)

non_parametric_significance_functions_dict = {}
def non_parametric_significance_functions_register(key, function):
    register(key, function, non_parametric_significance_functions_dict)

parametric_post_hoc_functions_dict = {}
def parametric_post_hoc_functions_register(key, function):
    register(key, function, parametric_post_hoc_functions_dict)

non_parametric_post_hoc_functions_dict = {}
def non_parametric_post_hoc_functions_register(key, function):
    register(key, function, non_parametric_post_hoc_functions_dict)

plot_post_hoc_functions_dict = {}
def plot_post_hoc_functions_register(key, function):
    register(key, function, plot_post_hoc_functions_dict)


print_significance_functions_dict = {}
def print_significance_functions_register(key, function):
    register(key, function, print_significance_functions_dict)

feature_calculation_functions_dict = {}
def feature_calculation_register(key,function):
    register(key, function, feature_calculation_functions_dict)


homophilic_feature_calculation_functions_dict = {}
def homophilic_feature_calculation_register(key,function):
    register(key, function, homophilic_feature_calculation_functions_dict)

embeddings_calculation_functions_dict = {}
def embeddings_calculation_register(key,function):
    register(key, function, embeddings_calculation_functions_dict)

parse_user_input_dict = {}
def parse_user_input_register(key, function):
    register(key, function, parse_user_input_dict)
    
