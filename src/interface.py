



import csv
import logging
import os
import subprocess
import sys
from functools import reduce

import click
import colorama
import pandas as pd
import yaml
from tabulate import tabulate

import src.functions.register as register
import src.utils.CustomClickOptions as CustomClickOptions
import src.utils.definitions as definitions
import src.utils.Status as Status
from src.functions import (archive, benchmark, non_parametric_post_hoc,
                           non_parametric_significance, parametric_post_hoc,
                           parametric_significance)
from src.functions import plot as plotter
from src.functions import (plot_post_hoc, plot_validation,
                           post_hoc_table_export, print_significance,
                           print_validation, printing, score, statistics,
                           table_export, validation, validation_table_export)
from src.functions.ml import features
from src.generators import generator
from src.generators import generator_utils as gen_utils
from src.utils import definitions, utils
from src.handler import config_handler,benchmark_handler
csv.field_size_limit(sys.maxsize)




logging.basicConfig(filename=str(definitions.LOG_FILE_PATH),format='[%(asctime)s] - [%(levelname)s] : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

def init():
    print('init')
    if not os.path.exists(definitions.SOLVER_FILE_PATH):
        file = open(definitions.SOLVER_FILE_PATH,'w+')



_version = "1.1"

@click.group()
def cli():
    pass

@click.command()
def version():
    """
    Prints the version of probo2.
    """
    print(f"probo2 {_version}")


#@click.command(cls=CustomClickOptions.command_required_option_from_option('fetch'))
@click.command()
@click.option("--name", "-n", required=False, help="Name of the solver")
@click.option("--path","-p",
              required=True,
              type=click.Path(exists=True,resolve_path=True),
              help="Path to solver executable")
@click.option("--format",
              "-f",
              multiple=True,
              required=False,
              help="Supported format of solver.")
@click.option('--tasks',
              "-t",
              required=False,
              default=[],
              callback=CustomClickOptions.check_problems,
              help="Supported computational problems")
@click.option("--version",
              "-v",
              type=click.types.STRING,
              required=False,
              help="Version of solver.")
@click.option(
    "--fetch","-ft",
    is_flag=True,
    help="Pull supported file format and computational problems from solver.")
@click.option("--yes",is_flag=True,help="Skip prompt.")
@click.option("--no_check", is_flag=True)
@click.pass_context
def add_solver(ctx, name, path, format, tasks, version, fetch,yes, no_check):
    """ Adds a solver to the database.
    Adding has to be confirmed by user.
    \f
      Args:
          tasks: Supported tasks
          guess: Pull supported file format and computational problems from solver.
          name: Name of the Solver as string.
          path: Full path to the executable of the solver as string.
          format: Supported file format of the solver as string.
          version: Version of solver.
     Returns:
         None
     Raises:
          None
      """
    from src.utils.options.CommandOptions import AddSolverOptions
    from src.handler import solver_handler
    options = AddSolverOptions(**ctx.params)
    options.check()
    solver_handler.add_solver(options)



@click.command()
@click.option("--name",
              "-n",
              type=click.types.STRING,
              help="Name of benchmark/fileset")
@click.option("--path",
              "-p",
              type=click.Path(exists=True,resolve_path=True),
              required=True,
              help="Path to instances")
@click.option("--format",
              "-f",
              multiple=True,
              type=click.Choice(definitions.DefaultInstanceFormats.as_list()),
              help="Supported formats of benchmark/fileset")
@click.option(
    "--query_extension",
    "-ext",
    multiple=True,
    default=None,
    help="Extension of additional query argument parameter for DC/DS problems.")
@click.option("--no_check",
              is_flag=True,
              help="Checks if the benchmark is complete.")
@click.option("--generate",
              "-g",
              type=click.types.Choice(['apx', 'tgf','i23']),
              help="Generate instances in specified format")
@click.option("--random_arguments",
              "-rnd",
              is_flag=True,
              help="Generate additional argument files with a random argument."
              )
@click.option("--dynamic_files",
              "-d",
              is_flag=True,
              help="Generate additional argument files with a random argument."
              )
@click.option("--function",'-fun', type=click.Choice(register.benchmark_functions_dict.keys()),multiple=True,help=' Custom functions to add additional attributes to benchmark.' )
@click.option("--yes",is_flag=True,help="Skip prompt.")
@click.option("--references_path",'-ref',type=click.Path(exists=True,resolve_path=True),help='Path to reference results for validation.')
@click.option("--extension_references",'-refext',help='Extensions of reference files.')
@click.pass_context
def add_benchmark(ctx,
                  name,
                  path,
                  format,
                  additional_extension,
                  dynamic_files,
                  no_check,
                  generate,
                  random_arguments,
                  function,
                  yes,
                  references_path,
                  extension_references
                  ):
    """ Adds a benchmark to the database.
     Before a benchmark is added to the database, it is checked if each instance is present in all specified file formats.
     Missing instances can be generated after the completion test (user has to confirm generation) or beforehand via the --generate/-g option.
     It is also possilbe to generate random argument files for the DC/DS problems with the --random_arguments/-rnd option.
     \f
       Args:
           name: Name of the Solver as string.
           path: Full path to instances.
           format: Supported formats of benchmark/fileset.
           competition: Competition benchmark was used in.
           graph_type: Version of solver.
           hardness: Hardness of benchmark/fileset.
           extension_arg_files: Extension of additional argument files
           no_check: Disable fileset check
           no_args: No addition argument files in benchmark
           random_arguments: Generate additional argument files with random arguments
           generate: Generate instances with specified format
      Returns:
          None
      Raises:
           None
       """
    from src.handler.benchmark_handler import add_benchmark
    from src.utils.options.CommandOptions import AddBenchmarkOptions

    
    options = AddBenchmarkOptions(name=name,
                                  path=path,
                                  format=format,
                                  additional_extension=query_extension,
                                  dynamic_files=dynamic_files,
                                  no_check=no_check,
                                  generate=generate,
                                  random_arguments=random_arguments,
                                  function=function,
                                  references_path=references_path,
                                  extension_references=extension_references,
                                  yes=yes)
    options.check()
    add_benchmark(options)

@click.command()
@click.option(
    "--all",
    "-a",
    required=False,
    is_flag=True,
    help="Execute all solvers supporting the specified tasks on specified instances.")
@click.option("--solver",
              "-solv",
              required=False,
              default=None,
              cls=CustomClickOptions.StringAsOption,
              help=" Comma-seperated list of ids or names of solvers (in database) to run.")
@click.option(
    "--benchmark",
    "-b",
    required=False,
    cls=CustomClickOptions.StringAsOption,
    default=None,
    help="Comma-seperated list of ids or names of benchmarks (in database) to run solvers on.")
@click.option("--task",
              cls=CustomClickOptions.StringAsOption,
              required=False,
              #callback=CustomClickOptions.check_problems,
              help="Comma-seperated list of tasks to solve.")
@click.option("--timeout",
              "-t",
              required=False,
              default=None, type=click.types.INT,
              help=" Instance cut-off value in seconds. If cut-off is exceeded instance is marked as timed out.")
@click.option("--dry",
              is_flag=True,
              help=" Print results to command-line without saving to the database.")
@click.option(
    "--track",
    cls=CustomClickOptions.TrackToProblemClass,
    default="",
    type=click.types.STRING,
    is_eager=True,
    help="Comma-seperated list of tracks to solve."
)
@click.option(
    "--name",
    required=False,
    help=
    "Tag for individual experiments.This tag is used to identify the experiment."
)
@click.option(
    "--notify",
    help=
    "Send a notification to the email address provided as soon as the experiments are finished."
)
@click.option("--repetitions","-reps",required=False,type=click.types.INT, help="Number of repetitions per instance. Run time is the avg of the n runs.")
@click.option("--rerun",'-rn',is_flag=True, help='Rerun last experiment')
@click.option("--subset","-sub",type=click.types.INT, help="Run only the first n instances of a benchmark.")
@click.option("--multi", is_flag=True,help="Run experiment on mutiple cores.")
@click.option("--config",'-cfg',type=click.Path(exists=True,resolve_path=True))
@click.option("--plot",'-plt',default=None, type=click.Choice(list(register.plot_dict.keys()) + ['all']),multiple=True)
@click.option("--statistics",'-stats',default=None,type=click.Choice(list(register.stat_dict.keys()) + ['all']),multiple=True)
@click.option("--printing",'-p',default=None,type=click.Choice(register.print_functions_dict.keys() ),multiple=True)
@click.option("--table_export",'-t',default=None,type=click.Choice(register.table_export_functions_dict.keys()),multiple=True)
@click.option("--archive",'-a',default=None,type=click.Choice(register.archive_functions_dict.keys()),multiple=True)
@click.option("--save_to", "-st",type=click.Path(resolve_path=True,exists=True), help="Directory to store tables")
@click.option("--name_map","-n",cls=CustomClickOptions.StringAsOption,
              required=False,default=None,help='Comma seperated list of solver names mapping. Format: <old>:<new>')
@click.option("--score",'-s',default=None,type=click.Choice(list(register.score_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--solver_arguments",type=click.Path(exists=True,resolve_path=True),help='Path to solver arguments file.')
@click.pass_context
def run(ctx, all,benchmark, task, solver, timeout, dry, name,
        notify, track, repetitions, rerun, subset, save_to, multi,statistics, config,plot,name_map,printing, table_export,archive,score,solver_arguments):
    """Run solver.
    \f
    Args:
        all (Boolean): Execute all solvers supporting the specified tasks.
        select (Boolean): Execute (via solver option) selected solver supporting the specified tasks.
        benchmark (str): Comma-seperated list of ids or names of benchmark to run solvers on.
        task (str): Comma-seperated list of tasks to solve.
        save_to ([type]): [description]
        solver (str):  Comma-seperated list of ids or names of solvers to run.
        timeout (int): Instance cut-off value. If cut-off is exceeded instance is marked as timed out.
        dry (Boolean): Print results to command-line without saving to the database.
        tag (str): Unique tag for experiment.
        notify (Boolean): Get e-mail notification when experiments are finished.
        report (Boolean): Create summary report of experiment.
        track (str): Comma-seperated list of tracks to solve.

    """
    from src.handler import experiment_handler

    cfg = config_handler.load_default_config()

    if config:
        user_cfg_yaml = config_handler.load_config_yaml(config)

        cfg.merge_user_input(user_cfg_yaml)
    if rerun:
        experiment_handler.set_config_of_last_experiment(cfg)
    
    cfg.merge_user_input(ctx.params)

    is_valid = cfg.check()
    
    if not is_valid:
        exit()
    
    try:
        experiment_handler.run_experiment(cfg)
    except Exception:
        experiment_id = experiment_handler.get_last_experiment()['id']
        experiment_handler.set_experiment_status(definitions.EXPERIMENT_INDEX,
                                                 experiment_id,
                                                 experiment_handler.ExperimentStatus.ABORTED)

    result_df = experiment_handler.load_results_via_name(cfg.name)

    saved_file_paths = []
    from src.handler import statistics_handler,scores_handler
    
    print(colorama.Fore.GREEN + "========== EVALUATION ==========")
    if cfg.statistics is not None:
        save_stats_to = os.path.join(cfg.save_to,'statistics')
        try:
            statistics_handler.calculate(cfg.statistics,cfg.raw_results_path,save_to=save_stats_to)
        except Exception as e:
            print(colorama.Fore.RED + f"An unexpected error occurred while calculating the stats: {e}")
            logging.error(f"An unexpected error occurred while calculating the stats: {e}")

    if cfg.score is not None:
        save_scores_to = os.path.join(cfg.save_to,'scores')
        try:
            scores_handler.calculate(cfg.score,cfg.raw_results_path,save_to=save_scores_to)
        except Exception as e:
            print(colorama.Fore.RED + f"An unexpected error occurred while calculating the scores: {e}")
            logging.error(f"An unexpected error occurred while calculating the scores: {e}")

    if cfg.plot is not None:
        try:
            saved_plots = plotter.create_plots(result_df,cfg)
        except Exception as e:
            print(colorama.Fore.RED + f"An unexpected error occurred while creating the plots: {e}")
            logging.error(f"An unexpected error occurred while creating the plots: {e}")

    to_merge = []
    others =  []
    if cfg.statistics is not None:

        if cfg.statistics =='all':
            cfg.statistics = register.stat_dict.keys()
        stats_results = []
        print("========== STATISTICS ==========")
        for stat in cfg.statistics:
            _res = register.stat_dict[stat](result_df)

            stats_results.append(_res)
        for res in stats_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if cfg.score is not None:
        score_results = []
        if cfg.score =='all' or 'all' in cfg.score:
                    cfg.score = register.score_functions_dict.keys()
        for s in cfg.score:
            _res = register.score_functions_dict[s](result_df)
            score_results.append(_res)
        for res in score_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if len(to_merge) >0:
        df_merged = reduce(lambda  left,right: pd.merge(left,right,how='inner'), to_merge)
        register.print_functions_dict[cfg.printing](df_merged,['tag','task','benchmark_name'])
        if cfg.table_export is not None:
            if cfg.table_export == 'all' or 'all' in cfg.table_export:
                cfg.table_export = register.table_export_functions_dict.keys()
            for format in cfg.table_export:
                register.table_export_functions_dict[format](df_merged,cfg,['tag','task','benchmark_name'])

    if cfg.validation['mode']:
        validation_results = validation.validate(result_df, cfg)
        print_validation.print_results(validation_results)
        if cfg.validation['plot']:
            plot_validation.create_plots(validation_results['pairwise'],cfg)
        if 'pairwise' in cfg.validation['mode']:
            if cfg.validation['table_export']:
                if cfg.validation['table_export'] == 'all' or 'all' in cfg.validation['table_export']:
                    cfg.validation['table_export'] = register.validation_table_export_functions_dict.keys()
                for f in cfg.validation['table_export']:
                    register.validation_table_export_functions_dict[f](validation_results['pairwise'],cfg)

    test_results = {}
    post_hoc_results = {}

    if cfg.significance['parametric_test']:
        test_results.update(parametric_significance.test(result_df,cfg))
    if cfg.significance['non_parametric_test']:
        test_results.update(non_parametric_significance.test(result_df,cfg))
    if cfg.significance['parametric_post_hoc']:
        post_hoc_results.update(parametric_post_hoc.test(result_df,cfg))
    if cfg.significance['non_parametric_post_hoc']:
        post_hoc_results.update(non_parametric_post_hoc.test(result_df,cfg))

    if test_results:
        print("========== Significance Analysis Summary ==========")
        for test in test_results.keys():
            print_significance.print_results(test_results[test],test)

    if post_hoc_results:
        print("========== Post-hoc Analysis Summary ==========")
        for test in post_hoc_results.keys():
            print_significance.print_results_post_hoc(post_hoc_results[test],test)

        if cfg.significance['plot']:
            for post_hoc_test in post_hoc_results.keys():
                plot_post_hoc.create_plots(post_hoc_results[post_hoc_test],cfg,post_hoc_test)
        if cfg.significance['table_export']:
            if cfg.significance['table_export'] == 'all' or 'all' in cfg.significance['table_export']:
                cfg.significance['table_export'] = register.post_hoc_table_export_functions_dict.keys()
            for post_hoc_test in post_hoc_results.keys():
                for f in cfg.significance['table_export']:
                    register.post_hoc_table_export_functions_dict[f](post_hoc_results[post_hoc_test],cfg,post_hoc_test)


    if cfg.copy_raws:
        click.echo('Copying raw files...',nl=False)
        experiment_handler.copy_raws(cfg)
        click.echo('done!')

    if cfg.archive is not None:
        click.echo('Creating archives...',nl=False)
        for _format in cfg.archive:
            register.archive_functions_dict[_format](cfg.save_to)
        click.echo('done!')


@click.command()
# @click.option("--par",
#               "-p",
#               type=click.types.INT,
#               help="Penalty multiplier for PAR score")
# @click.option("--solver",
#               "-s",

#               cls=CustomClickOptions.StringAsOption,
#               default=[],
#               help="Comma-separated list of solver ids")
# @click.option("--filter",
#               "-f",
#               cls=CustomClickOptions.FilterAsDictionary,
#               multiple=True)
# @click.option("--task",
#               "-t",
#               required=False,
#               callback=CustomClickOptions.check_problems,
#               help="Computational problems")
#@click.option("--benchmark",
            #   "-b",
            #   cls=CustomClickOptions.StringAsOption,
            #   default=[])
#@click.option("--print_format",
            #   "-pfmt",default='fancy_grid',
            #   type=click.Choice([
            #       "plain", "simple", "github", "grid", "fancy_grid", "pipe",
            #       "orgtbl", "jira", "presto", "pretty", "psql", "rst",
            #       "mediawiki", "moinmoin", "youtrack", "html", "unsafehtml"
            #       "latex", "latex_raw", "latex_booktabs", "textile"
            #   ]))
@click.option("--name", "-n",default=None,help='Experiment name')
#@click.option("--combine",
            #  "-c",
            #   cls=CustomClickOptions.StringAsOption,
            #   default=[])
#@click.option("--vbs", is_flag=True, help="Create virtual best solver")
#@click.option("--save_to", "-st",type=click.Path(resolve_path=True,exists=True), help="Directory to store tables")
#@click.option("--export","-e",type=click.Choice(['latex','csv','json']),default=None,multiple=True)
#@click.option("--css",default="styled-table.css",help="CSS file for table style.")
#@click.option("--statistics",'-s',type=click.Choice(['mean','sum','min','max','median','var','std','coverage','timeouts','solved','errors','all','best']),multiple=True)
#@click.option("--verbose",'-v',is_flag=True,help='Show additional information for some statistics.')
@click.option("--last", "-l",is_flag=True,help="Calculate stats for the last finished experiment.")
#@click.option("--send", required=False, help="Send plots via E-Mail.")
@click.option("--config",'-cfg', default=None,type=click.Path(exists=True,resolve_path=True),help='Path to experiment config.')
@click.option("--raw",'-r',default=None, type=click.Path(exists=True,resolve_path=True), help='Path to raw result file.')
@click.option("--statistics",'-stat',default=None,type=click.Choice(list(register.stat_dict.keys()) + ['all']),multiple=True)
@click.option("--score",'-s',default=None,type=click.Choice(list(register.score_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--printing",default='default')
#@click.option("--archive",'-a',default=None,type=click.Choice(register.archive_functions_dict.keys()),multiple=True)
@click.option("--table_export",'-t',default=None,type=click.Choice(list(register.table_export_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--save_to", "-st",type=click.Path(resolve_path=True,exists=True), help="Directory to store tables")
def calculate(
              name, statistics,score,raw,config,printing,table_export, save_to,last):
    """Calculate statistics and scores for experiment results."""
    from src.handler import experiment_handler
    if last:
        last_experiment = experiment_handler.get_last_experiment()
        if last_experiment is not None:
            cfg = config_handler.load_config_via_name(last_experiment['name'])
            result_df= experiment_handler.load_results_via_name(cfg.name)
        else:
            print("No experiment found.")

    elif name is not None:
        result_df= experiment_handler.load_results_via_name(name)
        cfg = config_handler.load_config_via_name(name)
    elif raw is not None:
        result_df = pd.read_csv(raw)
    elif config is not None:
        cfg = config_handler.load_config_yaml(config,as_obj=True)
        result_df = experiment_handler.load_experiments_results(cfg)
    else:
        print('Unable to load results')
        exit()


    to_merge = []
    others =  []
    print("========== RESULTS ==========")
    if statistics is not None:
        stats_results = []
        if statistics =='all' or 'all' in statistics:
                statistics = register.stat_dict.keys()
        for stat in statistics:
            _res = register.stat_dict[stat](result_df)
            stats_results.append(_res)
        for res in stats_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if score is not None:
        score_results = []
        if score =='all' or 'all' in score:
                score = register.score_functions_dict.keys()
        for s in score:
            _res = register.score_functions_dict[s](result_df)
            score_results.append(_res)
        for res in score_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])

    if len(to_merge) > 0:
        df_merged = reduce(lambda  left,right: pd.merge(left,right,how='inner'), to_merge)
        register.print_functions_dict[printing](df_merged,['tag','task','benchmark_name'])

    for other in others:
        register.print_functions_dict[printing](other,['tag','task','benchmark_name'])


    if not save_to:
        cfg.save_to = os.getcwd()
    else:
        cfg.save_to  = save_to
    if table_export is not None:
        if 'all' in table_export or table_export == 'all':
            table_export = register.table_export_functions_dict.keys()
        for format in table_export:
            register.table_export_functions_dict[format](df_merged,cfg,['tag','task','benchmark_name'])
@click.command()
@click.pass_context
@click.option("--name", "-n",cls=CustomClickOptions.StringAsOption, default=[])
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store plots in. Filenames will be generated automatically.")
#@click.option("--vbs", is_flag=True, help="Create virtual best solver.")
#@click.option("--x_max", "-xm", type=click.types.INT)
#@click.option("--y_max", "-ym", type=click.types.INT)
#@click.option("--alpha",
            #   "-a",
            #   type=click.FloatRange(0, 1),
            #   help="Alpha value (only for scatter plots)")
# @click.option("--backend",
#               "-b",
#               type=click.Choice(['pdf', 'pgf', 'png', 'ps', 'svg']),
#               default='png',
#               help="Backend to use")
# @click.option("--no_grid", "-ng", is_flag=True, help="Do not show a grid.")
# @click.option("--grid_plot",is_flag=True)
# @click.option("--combine",
#               "-c",
#               type=click.Choice(['tag','task_id','benchmark_id']),help='Combine results on specified key.')
@click.option("--kind",'-k',default=None, type=click.Choice(list(register.plot_dict.keys()) + ['all']),multiple=True)
# @click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
# @click.option("--send", required=False, help="Send plots via E-Mail.")
# @click.option("--last", "-l",is_flag=True,help="Plot results for the last finished experiment.")
#@click.option("--axis_scale",'-as',type=click.Choice(['linear','log']),default='log',help="Scale of x and y axis." )
#@click.option("--set_default", is_flag=True)
@click.option("--config",'-cfg', default=None,type=click.Path(exists=True,resolve_path=True))
@click.option("--raw",'-r',default=None, type=click.Path(exists=True,resolve_path=True))
@click.option("--last", "-l",is_flag=True,help="Calculate stats for the last finished experiment.")

def plot(ctx, name,  save_to,kind,raw,config,last):
    """Create plots of experiment results.

    The --tag option is used to specify which experiment the plots should be created for.
    With the options --solver, --task and --benchmark you can further restrict this selection.
    If only a tag is given, a plot is automatically created for each task and benchmark of this experiment.
    With the option --kind you determine what kind of plot should be created.
    It is also possible to combine the results of different experiments, benchmarks and tasks with the --combine option.

    \f


    Args:
        ctx ([type]): [description]
        tag ([type]): [description]
        task ([type]): [description]
        benchmark ([type]): [description]
        solver ([type]): [description]
        save_to ([type]): [description]
        filter ([type]): [description]
        vbs ([type]): [description]
        backend ([type]): [description]
        combine ([type]): [description]
        kind ([type]): [description]
        compress ([type]): [description]
        send ([type]): [description]
        last ([type]): [description]
    """
    from src.handler import experiment_handler
    cfg = config_handler.load_default_config()
    if last:
        last_experiment = experiment_handler.get_last_experiment()
        if last_experiment is not None:
            cfg = config_handler.load_config_via_name(last_experiment['name'])
            result_df= experiment_handler.load_results_via_name(cfg.name)
        else:
            print("No experiment found.")
    elif name is not None:
        result_df = experiment_handler.load_results_via_name(name)
    elif raw is not None:
        result_df = pd.read_csv(raw)
    elif config is not None:
        user_cfg_yaml = config_handler.load_config_yaml(config)
        cfg.merge_user_input(user_cfg_yaml)
        result_df = experiment_handler.load_experiments_results(cfg)
    else:
        print('Unable to load results')
        exit()

    cfg.merge_user_input(ctx.params)

    if not save_to:
        cfg.save_to = os.getcwd()


    cfg.check()

    if kind is not None:

        if kind =='all' or 'all' in kind:
            cfg.plot = register.plot_dict.keys()
        else:
            cfg.plot = list(kind)
        saved_plots = plotter.create_plots(result_df,cfg)
    else:
        print("Please specify a plot type with the --kind option.")


@click.command()
@click.option("--verbose","-v",is_flag=True,help="Prints additional information on benchmark")
@click.option("--id",type=click.INT,help="Print information of benchmark")
def benchmarks(verbose,id):
    """ Prints benchmarks in database to console.

    Args:
        None
    Return:
        None
    Raises:
        None
   """
    from src.handler import benchmark_handler
    if verbose:
        benchmark_handler.print_benchmarks(extra_columns=['ext_additional','dynamic_files'])
    else:
        if id:
            benchmark = benchmark_handler.load_benchmark_by_identifier([id])
            print(f"========== Summary {benchmark[0]['name']} ==========")
            benchmark[0]['num_instances'] = benchmark_handler.get_instances_count(benchmark[0]['path'],benchmark[0]['format'][0])
            print(yaml.dump(benchmark, allow_unicode=True, default_flow_style=False))

        else:
            benchmark_handler.print_benchmarks()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
@click.option("--id",type=click.INT,help="Print information of solver")
def solvers(verbose,id):
    """Prints solvers in database to console.

    Args:
        verbose ([type]): [description]
    """
    from src.handler import solver_handler
    if verbose:
        solver_handler.print_solvers(extra_columns=['tasks'])
    else:
        if id:
            solver = solver_handler.load_solver_by_identifier([id])
            print(f"========== Summary {solver[0]['name']} ==========")
            print(yaml.dump(solver, allow_unicode=True, default_flow_style=False))

        else:

            solver_handler.print_solvers()

@click.command()
@click.option('--tag','-t',help='Tag of experiments to show status.')
def status(tag):
    """Provides an overview of the progress of the currently running experiment.

    """
    from src.handler import experiment_handler
    from src.handler import config_handler
    experiment_df = pd.read_csv(definitions.EXPERIMENT_INDEX)
    selected_experiment = experiment_df[experiment_df.name == tag]
    if tag:
        experiment_df = pd.read_csv(definitions.EXPERIMENT_INDEX)
        selected_experiment = experiment_df[experiment_df.name == tag]
        cfg_selected_experiment = config_handler.load_config_yaml(selected_experiment['config_path'].iloc[0],as_obj=True)

        Status.print_status_summary(cfg_selected_experiment.status_file_path)

    elif os.path.exists(str(definitions.STATUS_FILE_DIR)):
        Status.print_status_summary()
    else:
        print("No status query is possible.")


@click.command()
@click.option("--name","-n",help="Experiment tag to be validated")
@click.option("--reference",
              "-r",
              type=click.Path(resolve_path=True,exists=True),
              help="Path to reference files.")
@click.option("--raw", is_flag=True, help='Export raw validation results in csv format.')
@click.option(
    "--save_to",
    "-st",
    type=click.Path(resolve_path=True,exists=True),
    required=False,
    help=
    "Directory to store plots and data in. Filenames will be generated automatically.")

@click.option('--extension','-ext',multiple=True, help="Reference file extension")
@click.option("--config",'-cfg', default=None,type=click.Path(exists=True,resolve_path=True),help='Path to experiment config.')
@click.option("--mode",'-m',default=None,type=click.Choice(list(register.validation_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--plot",'-m',default=None,type=click.Choice(list(register.plot_validation_functions_dict.keys())),multiple=True)
@click.option("--last", "-l",is_flag=True,help="Validate last finished experiment.")
@click.option('--table_export',
              '-e',
              type=click.Choice(list(register.validation_table_export_functions_dict.keys()) + ['all']),multiple=True)
def validate(name,config,mode,reference,extension,
             save_to, table_export,plot,raw,last):
    """Validate experiments results.

    With the validation, we have the choice between a pairwise comparison of the results or a validation based on reference results.
    Pairwise validation is useful when no reference results are available. For each solver pair, the instances that were solved by both solvers are first identified.
    The results are then compared and the percentage of accordance is calculated and output in the form of a table.
    It is also possible to show the accordance of the different solvers as a heatmap with the "--plot heatmap" option.
    Note: The SE task is not supported here.

    For the validation with references, we need to specify the path to our references results with the option "--ref". It's important to note that each reference instance has to follow a naming pattern to get matched with the correct result instance.
    The naming of the reference instance has to include (1) the full name of the instance to validate, (2) the task, and (3) the specified extension. For example for the instance "my_instance.apx",  the corresponding reference instance for the EE-PR task would be named as follows: "my_instance_EE-PR.apx"
    The order of name and task does not matter.
    The extension is provided via the "--extension" option.

    \f



    Args:
        tag ([type]): [description]
        task ([type]): [description]
        benchmark ([type]): [description]
        solver ([type]): [description]
        filter ([type]): [description]
        reference ([type]): [description]
        pairwise ([type]): [description]
        save_to ([type]): [description]
        export ([type]): [description]
        update_db ([type]): [description]
        extension ([type]): [description]
    """
    from src.handler import experiment_handler
    if last:
        last_experiment = experiment_handler.get_last_experiment()
        if last_experiment is not None:
            cfg = config_handler.load_config_via_name(last_experiment['name'])
            result_df= experiment_handler.load_results_via_name(cfg.name)
        else:
            print("No experiment found.")
    elif name is not None:
        result_df= experiment_handler.load_results_via_name(name)
        cfg = config_handler.load_config_via_name(name)
    elif config is not None:
        cfg = config_handler.load_config_yaml(config,as_obj=True)
        result_df = experiment_handler.load_experiments_results(cfg)
    else:
        print('Unable to load results.')
        exit()

    if not save_to:
        cfg.save_to = os.getcwd()
    else:
        cfg.save_to = save_to


    validation_cfg = {"mode": mode,
  "plot": list(plot),
  "table_export": list(table_export)}
    cfg.validation = validation_cfg
    if mode:
        validation_results = validation.validate(result_df, cfg)
        print_validation.print_results(validation_results)

        if cfg.validation['plot'] and "pairwise" in validation_results.keys():
            plot_validation.create_plots(validation_results['pairwise'],cfg)

        if 'pairwise' in mode:

            if cfg.validation['table_export']:
                if cfg.validation['table_export'] == 'all' or 'all' in cfg.validation['table_export']:
                    cfg.validation['table_export'] = register.validation_table_export_functions_dict.keys()
                for f in cfg.validation['table_export']:
                    register.validation_table_export_functions_dict[f](validation_results['pairwise'],cfg)


    else:
        print("Please specify a validation mode via the --mode/-m Option.")
        exit()

@click.command()
@click.option("--name", cls=CustomClickOptions.StringAsOption, default=[],help="Experiment tag to be tested.")
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Comma-separated list of task IDs or symbols to be tested.")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[],help="Benchmark name or id to be tested.")
@click.option("--solver",
              "-s",
              required=False,
              cls=CustomClickOptions.StringAsOption,
              default=[],
              help="Comma-separated list of solver ids")
@click.option("--filter",
              "-f",
              cls=CustomClickOptions.FilterAsDictionary,
              multiple=True,
              help="Filter results in database. Format: [column:value]")
@click.option("--combine",
              "-c",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--parametric","-p", type=click.Choice(list(register.parametric_significance_functions_dict.keys()) + ['all']),multiple=True,help="Parametric significance test. ANOVA for mutiple solvers and t-test for two solvers.")
@click.option("--non_parametric",
              "-np",
              type=click.Choice(list(register.non_parametric_significance_functions_dict.keys()) + ['all']),multiple=True,help="Non-parametric significance test. kruksal for mutiple solvers and mann-whitney-u for two solvers.")
@click.option("--post_hoc_parametric",
              "-php",
              type=click.Choice(list(register.parametric_post_hoc_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--post_hoc_non_parametric",
              "-phn",
               type=click.Choice(list(register.non_parametric_post_hoc_functions_dict.keys()) + ['all']),multiple=True)
@click.option("--p_adjust", type=click.Choice(
                  ['holm']),default='holm')
@click.option("--alpha", "-a", default=0.05, help="Significance level.")
@click.option('--table_export',
              '-e',
              type=click.Choice(list(register.post_hoc_table_export_functions_dict.keys()) + ['all']),multiple=True)
@click.option('--plot',

              type=click.Choice(list(register.plot_post_hoc_functions_dict.keys()) + ['all']),multiple=True,help="Create a heatmap for pairwise comparision results and a count plot for validation with references.")
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")
@click.option("--last", "-l",is_flag=True,help="Test the last finished experiment.")
def significance(name, task, benchmark, solver, filter, combine, parametric,
                 non_parametric, post_hoc_parametric, post_hoc_non_parametric,
                 alpha, table_export, save_to, p_adjust,last,plot):
    """Parmatric and non-parametric significance and post-hoc tests.

    Args:
        tag ([type]): [description]
        task ([type]): [description]
        benchmark ([type]): [description]
        solver ([type]): [description]
        filter ([type]): [description]
        combine ([type]): [description]
        parametric ([type]): [description]
        non_parametric ([type]): [description]
        post_hoc_parametric ([type]): [description]
        post_hoc_non_parametric ([type]): [description]
        alpha ([type]): [description]
        export ([type]): [description]
        save_to ([type]): [description]
        p_adjust ([type]): [description]
        equal_sample_size ([type]): [description]
        last ([type]): [description]
    """
    from src.handler import experiment_handler

    if last:
        last_experiment = experiment_handler.get_last_experiment()
        if last_experiment is not None:
            cfg = config_handler.load_config_via_name(last_experiment['name'])
            result_df= experiment_handler.load_results_via_name(cfg.name)
        else:
            print("No experiment found.")
    elif name is not None:
        result_df= experiment_handler.load_results_via_name(name)
        cfg = config_handler.load_config_via_name(name)
    else:
        print('Unable to load results.')
        exit()

    if not save_to:
        cfg.save_to = os.getcwd()
    else:
        cfg.save_to = save_to

    if table_export == 'all' or 'all' in table_export:
        table_export = register.post_hoc_table_export_functions_dict.keys()


    sig_cfg =   {   "parametric_test": list(parametric),
                    "non_parametric_test": list(non_parametric),
                    "parametric_post_hoc": list(post_hoc_parametric),
                    "non_parametric_post_hoc": list(post_hoc_non_parametric),
                    "p_adjust": "holm",
                    "plot": list(plot),
                    "table_export": list(table_export)
                }
    cfg.significance = sig_cfg
    test_results = {}
    post_hoc_results = {}
    if cfg.significance['parametric_test']:
        test_results.update(parametric_significance.test(result_df,cfg))
    if cfg.significance['non_parametric_test']:
        test_results.update(non_parametric_significance.test(result_df,cfg))
    if cfg.significance['parametric_post_hoc']:
        post_hoc_results.update(parametric_post_hoc.test(result_df,cfg))
    if cfg.significance['non_parametric_post_hoc']:
        post_hoc_results.update(non_parametric_post_hoc.test(result_df,cfg))

    if test_results:
        print("========== Significance Analysis Summary ==========")
        for test in test_results.keys():
            print_significance.print_results(test_results[test],test)

    if post_hoc_results:
        print("========== Post-hoc Analysis Summary ==========")
        for test in post_hoc_results.keys():
            print_significance.print_results_post_hoc(post_hoc_results[test],test)
        if cfg.significance['plot']:
            for post_hoc_test in post_hoc_results.keys():
                plot_post_hoc.create_plots(post_hoc_results[post_hoc_test],cfg,post_hoc_test)
        if cfg.significance['table_export']:
            for post_hoc_test in post_hoc_results.keys():
                for f in cfg.significance['table_export']:
                    register.post_hoc_table_export_functions_dict[f](post_hoc_results[post_hoc_test],cfg,post_hoc_test)


@click.command()
@click.option("--id", required=False,help='ID or name of solver to delete.')
@click.option("--all", is_flag=True,help='Delete all solvers in database.')
def delete_solver(id, all):
    """ Deletes a solver from the database.
    Deleting has to be confirmed by user.

    Args:
        id: Solver id
    Return:
        None
    Raises:
        None
   """
    import src.handler.solver_handler as solver_handler
    if id is not None:
        click.confirm(
                 "Are you sure you want to delete this solver in the database?",
                 abort=True,default=True)
        solver_handler.delete_solver(id)
    if all:
        click.confirm(
                 "Are you sure you want to delete all solvers in the database?",
                 abort=True,default=True)
        solver_handler.delete_all_solvers()

@click.command()
@click.option("--id", required=False,help='ID of benchmark to delete.')
@click.option("--all", is_flag=True, help='Delete all benchmarks in database')
def delete_benchmark(id, all):
    """ Deletes a benchmark from the database.
    Deleting has to be confirmed by user.

    Args:
        id: benchmark id
    Return:
        None
    Raises:
        None
   """
    from src.handler import benchmark_handler
    if id is not None:
        click.confirm(
                 "Are you sure you want to delete this benchmark in the database?",
                 abort=True,default=True)
        benchmark_handler.delete_benchmark(id)
    if all:
        click.confirm(
                 "Are you sure you want to delete all benchmarks in the database?",
                 abort=True,default=True)
        benchmark_handler.delete_all_benchmarks()

@click.command()
@click.option('--verbose','-v',is_flag=True, help='Show summary of whole experiment.')
def last(verbose):
    """Shows basic information about the last finished experiment.

        Infos include experiment tag, benchmark names, solved tasks, executed solvers and and the time when the experiment was finished

    """
    from src.handler import experiment_handler


    last_experiment = experiment_handler.get_last_experiment()
    if last_experiment is not None:
        cfg = config_handler.load_config_via_name(last_experiment['name'])
        cfg.print()
    else:
        print("No experiment found.")

@click.command()
@click.option("--id", required=False,help='ID of benchmark to calculate features.')
@click.option( "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark features in. Default is the current working directory.")
@click.option('--feature','-f',multiple=True,type=click.Choice(list(register.feature_calculation_functions_dict.keys())),help='Node features to calculate.')
@click.option('--embedding','-e',multiple=True, type=click.Choice(list(register.embeddings_calculation_functions_dict.keys())),help='Node embeddings to generate.')
@click.option('--concat','-c',is_flag=True)
@click.option('--benchmark_feature','-bf', multiple=True, type=click.Choice(['num_nodes','num_edges','avg_out_deg','avg_in_deg']))
def features(id, save_to, feature, embedding, concat, benchmark_feature):
    """Calculates node features and embeddings
    """
    from src.functions.ml import features
    from src.handler import benchmark_handler
    import glob
    import pandas as pd
    from os import chdir
    if save_to is None:
        from os import getcwd
        save_to = getcwd()



    benchmark_info = benchmark_handler.load_benchmark(id)[0]
    if benchmark_feature:
        features.calculate_benchmark_level_features(benchmark_info,benchmark_feature,save_to)

    if feature:
        features_files_path, embeddings_file_path = features.calculate_graph_level_features(benchmark_info, feature, embedding, save_to)
        if concat:
            chdir(features_files_path)
            extension = 'csv'
            all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
            combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
            combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
cli.add_command(features)
@click.command()
@click.option("--benchmark","-b", required=False,multiple=True,help='ID of benchmark to calculate features.')
@click.option("--solver","-s", required=False,multiple=True,help='ID of ground-truth solver to calculate features.')
@click.option( "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark features in. Default is the current working directory.")
@click.option("--timeout",type=click.INT,help='Timeout for accaptance tasks')
@click.option("--task","-t",multiple=True, help='Accaptance task to create labels')
def labels(benchmark,solver,save_to,task,timeout):
    """Calculates accaptance labels for solvers on benchmarks and tasks."""
    from src.handler import benchmark_handler
    from src.handler import solver_handler
    from src.functions.ml import labels
    if save_to is None:
        from os import getcwd
        save_to = getcwd()
    solver = solver_handler.load_solver(list(solver))[0]
    benchmarks = benchmark_handler.load_benchmark(list(benchmark))
    tasks = list(task)

    if tasks:
        options = labels.AccaptanceLabelOptions(solver=solver,tasks=tasks,benchmarks=benchmarks,timeout=timeout,save_to=save_to)
        labels.get_label_accaptance(options)

cli.add_command(labels)

@click.command()
@click.option('--name','-n',help='Experiment name')
@click.option('--raw','-r', type=click.Path(exists=True, resolve_path=True),help='Path of raw.csv')
def board(name,raw):
    """Launch a dashboard to visualize results."""
    from src.handler import experiment_handler
    from src.probo2board import board
    if name:
        experiment_df = experiment_handler.load_results_via_name(name)
    elif raw:
        experiment_df = pd.read_csv(raw)
    else:
        print('Experiment not found!')

    if experiment_df is None:
        exit()
    board.launch(experiment_df)


cli.add_command(board)






@click.command()
@click.option("--list","-l", is_flag=True)
@click.option("--name","-n",help="Experiment tag")
def experiments(list,name):
    """Prints some basic information about the experiment speficied with "--tag" option.

    Args:
        tag ([type]): [description]
    """
    from src.handler import experiment_handler

    if list:
        print("========== Experiments Overview ==========")
        experiment_handler.print_experiment_index()

    if name is not None:
        print('========== Experiment Summary ==========')
        #result_df= experiment_handler.load_results_via_name(name)
        cfg = config_handler.load_config_via_name(name)
        cfg.print()


def recursive_help(cmd,text_dict,parent=None):
    #print(f'cmd dict: {cmd.__dict__}\n\n')

    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    text_dict[cmd.__dict__['name']] =get_help(cmd,ctx)
    #print(get_help(cmd,ctx))

    click.core.Command.get_help
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub,text_dict,ctx)



@click.command()
def dumphelp_markdown():
    text_dict = {}
    with open("dump_help.txt",'a') as fp:
        recursive_help(cli,text_dict)
    #1.[Setup](#setup)
    table_of_contents_commands = ''
    for i,command in enumerate(text_dict):
        table_of_contents_commands += f'+ [{command}](#{command.lower()})\n'

    combined_help_text = ""
    for command, help_text in text_dict.items():
        combined_help_text += (f'### {command}\n{help_text}\n')

    final = (table_of_contents_commands + combined_help_text).replace('cli','probo2').replace('INTEGER','').replace('TEXT','')
    print(final)


def get_help(cmd,ctx):

    formatter =  CustomClickOptions.MyFormatter(
            width=ctx.terminal_width, max_width=ctx.max_content_width)
    cmd.format_help(ctx, formatter)
    return formatter.getvalue().rstrip("\n")

@click.command()
def logs():
    """
    Prints the contents of the log file.
    """
    with open(str(definitions.LOG_FILE_PATH),"r") as log_file:
        print(log_file.read())

@click.command()
@click.option('--benchmark','-b', type=click.Choice(['ICCMA15','ICCMA19','ICCMA21','ICCMA23']),multiple=True, help='Name of benchmark to fetch')
@click.option('--solver','-s', type=click.Choice(['ICCMA19','ICCMA21','ICCMA23']),multiple=True, help='Name of solver to fetch' )
@click.option('--install',is_flag=True,help='Install fetched solvers and add them to probo2.')

@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark in. Default is the current working directory.")
@click.pass_context
def fetch(ctx,benchmark, save_to,solver,install):
    """Download ICCMA competition benchmakrs and add them to the database.

    Args:
        ctx ([type]): [description]
        name ([type]): [description]
        save_to ([type]): [description]
    """
    import json

    from src.utils import fetching

    with open(definitions.FETCH_BENCHMARK_DEFAULTS_JSON,'r') as file:
        json_string = file.read()
    fetch_options = json.loads(json_string)

    if not save_to:
        save_to = os.getcwd()
    for benchmark_name in benchmark:
        current_benchmark_options = fetch_options[benchmark_name]
        path_benchmark = fetching._fetch_benchmark(benchmark_name,save_to,current_benchmark_options)
        if path_benchmark:
            ctx.invoke(add_benchmark,
                   name=benchmark_name,
                   path=path_benchmark,
                   format=tuple(current_benchmark_options['format']),
                   random_arguments=current_benchmark_options['random_arguments'],
                   query_extension=current_benchmark_options['extension_arg_files'],
                   generate=current_benchmark_options['generate'],
                    )
                   
    if solver:
        if not os.path.exists(os.path.join(save_to,"ICCMA-Solvers")):
            subprocess.call('git clone https://github.com/jklein94/ICCMA-Solvers.git', shell = True, cwd=save_to)
        else:
            subprocess.call('git pull', shell = True, cwd=os.path.join(save_to,"ICCMA-Solvers"))

        if install:
            for s in solver:
                solver_dir = os.path.join(save_to,'ICCMA-Solvers',f'{s}_solvers')
                if os.path.exists(solver_dir):
                    subprocess.call(f'bash install.sh',shell=True,cwd=solver_dir)



# @click.command()
# @click.pass_context
# @click.option("--num","-n",required=True,type=click.INT, help='Number of instances to generate')
# @click.option("--name",help='Benchmark name')
# @click.option(
#     "--save_to",
#     "-st",
#     type=click.Path(exists=True, resolve_path=True),
#     help="Directory to store benchmark in. Default is the current working directory.")
# @click.option("--num_args","-a", help='Number of arguments.')
# @click.option("--generate_solutions", "-gs", is_flag=True,help="Generate solutions for instance.")
# @click.option("--generate_arg_files", "-ga", help='Generate additional argument files for DS/DC problems.')
# @click.option("--solver", "-s", help='Solver (id/name) for solution generation')
# @click.option("--timeout", "-t",type=click.types.FLOAT, help='Timeout for instances in seconds. Instances that exceed this value are discarded')
# @click.option("--timemin","-tm",type=click.types.FLOAT, help='Minimum solving time of instance.')
# @click.option("--format","-f", type=click.Choice(['apx','tgf']),help='Format of instances to generate')
# @click.option("--task","-t",help='Tasks to generate solutions for.')
# @click.option("--random_size","-rs",nargs=2,type=click.INT, help='Random number of arguments per instances. Usage: -rs [min] [max]')
# @click.option("--add",is_flag=True,help="Add benchmark to database.")
# @click.option("--solver")
# @click.option('--task')
# def stable_generator(ctx,num, name, save_to, num_args, generate_solutions, generate_arg_files,solver, timeout, timemin, format, task, random_size, add):
#     """ Python port of the original StableGenerator of probo by Matthias Thimm.
#         This is the generator for abstract argumentation graphs used
#         in ICCMA'15 for the second group of problems (testcases 4-6). It generates
#         graphs that (likely) possess many stable, preferred, and complete extensions.
#         It works roughly as follows:
#         1.) A set of arguments is identified to form an acyclic subgraph, containing
#          the likely grounded extension
#         2.) a subset of arguments is randomly selected and attacks are randomly added
#           from some arguments within this set to all arguments outside the set (except
#           to the arguments identified in 1.)
#         3.) Step 2 is repeated until a number of desired stable extensions is reached *
#         For more details see the code.

#         This generator can be used by adapting the variables in the configuration block in the
#         beginning of the class. Once started, this generator generates graphs continuously
#         (it will not terminate on its own, the application has to be terminated forcefully).

#          Matthias Thimm
#     """
#     if not save_to:
#         save_to = os.getcwd()
#         ctx.params['save_to'] = save_to
#     if len(random_size) < 1:
#         ctx.params['random_size'] = None
#     default_config =  gen_utils.read_default_config(str(definitions.GENERATOR_DEFAULTS_JSON))
#     gen_utils.update_configs(ctx, default_config,generator_type='stable')
#     if default_config['general']['timemin'] and default_config['general']['timeout']:
#         if (solver and task):

#             engine = DatabaseHandler.get_engine()
#             session = DatabaseHandler.create_session(engine)
#             ref_solver =  DatabaseHandler.get_solver(session, solver)
#             ref_task = DatabaseHandler.get_task(session, task)
#             if ref_task not in ref_solver.supported_tasks:
#                 print("Solver does not support the specified task")
#             default_config['general']['save'] = False
#             accepted_graphs = []
#             step_size = 100
#             i = 0
#             while len(accepted_graphs) < default_config["general"]["num"]:
#                 i += 1
#                 num = default_config["general"]["num"] - len(accepted_graphs)
#                 graphs = generator.generate('stable', default_config)
#                 accepted_graphs.extend(gen_utils._ensure_runtime(graphs, default_config, ref_solver, ref_task))
#                 print(f'{accepted_graphs=}')
#                 num_extension = default_config['stable']['num_extension']
#                 size_extension = default_config['stable']['size_extension']
#                 size_grounded = default_config['stable']['size_grounded']
#                 default_config['stable']['num_extension'] = [num_extension[0] + step_size, num_extension[1] + (2*step_size)]
#                 default_config['stable']['size_extension'] = [size_extension[0] + step_size, size_extension[1] + (2*step_size)]
#                 default_config['stable']['size_grounded'] = [size_grounded[0] + step_size, size_grounded[1] + (2*step_size)]
#                 print(default_config['stable']['num_extension'])
#                 print(default_config['stable']['size_extension'])
#                 print(default_config['stable']['size_grounded'])
#                 print(f'{step_size=}')
#                 if i % 5 == 0:
#                     step_size += int(round(step_size * 0.5))
#             gen_utils.save_instances(accepted_graphs, default_config['general'],'stable')



#         else:
#             print("To ensure instance runtime you have to specify a solver and a task")
#             exit()

#     else:
#         save_directory = generator.generate('stable',default_config)
#     if add:
#         general_configs = default_config['general']
#         ctx.invoke(add_benchmark,
#                    name=general_configs['name'],
#                    path=save_directory,
#                    format=tuple(general_configs['format']),
#                    random_arguments=False,
#                    extension_arg_files="arg",
#                    generate=None
#                    )

# @click.command()
# @click.pass_context
# @click.option("--num","-n",required=True,type=click.INT, help='Number of instances to generate')
# @click.option("--name",help='Benchmark name')
# @click.option(
#     "--save_to",
#     "-st",
#     type=click.Path(exists=True, resolve_path=True),
#     help="Directory to store benchmark in. Default is the current working directory.")
# @click.option("--num_args","-a", help='Number of arguments.')
# @click.option("--generate_solutions", "-gs", is_flag=True,help="Generate solutions for instance.")
# @click.option("--generate_arg_files", "-ga", help='Generate additional argument files for DS/DC problems.')
# @click.option("--solver", "-s", help='Solver (id/name) for solution generation')
# @click.option("--timeout", "-t",help='Timeout for instances in seconds. Instances that exceed this value are discarded')
# @click.option("--timemin","-tm",help='Minimum solving time of instance.')
# @click.option("--format","-f", type=click.Choice(['apx','tgf']),help='Format of instances to generate')
# @click.option("--task","-t",help='Tasks to generate solutions for.')
# @click.option("--random_size","-rs",nargs=2,type=click.INT,default=None, help='Random number of arguments per instances. Usage: -rs [min] [max]')
# @click.option("--add",is_flag=True,help="Add benchmark to database.")
# def scc_generator(ctx,num, name, save_to, num_args, generate_solutions, generate_arg_files,solver, timeout, timemin, format, task, random_size, add):

#     if not save_to:
#         save_to = os.getcwd()
#         ctx.params['save_to'] = save_to
#     if len(random_size) < 1:
#         ctx.params['random_size'] = None
#     default_config =  gen_utils.read_default_config(str(definitions.GENERATOR_DEFAULTS_JSON))
#     gen_utils.update_configs(ctx, default_config,generator_type='scc')
#     if default_config['general']['timemin'] and default_config['general']['timeout']:
#         if (solver and task):

#             engine = DatabaseHandler.get_engine()
#             session = DatabaseHandler.create_session(engine)
#             ref_solver =  DatabaseHandler.get_solver(session, solver)
#             ref_task = DatabaseHandler.get_task(session, task)
#             if ref_task not in ref_solver.supported_tasks:
#                 symbols = [ t.symbol for t in ref_solver.supported_tasks]
#                 print(f"Solver {ref_solver.solver_full_name} does not support the specified task {ref_task.symbol}.")
#                 print(f'Please choose one of the following tasks:\n{symbols}')
#                 exit()

#             default_config['general']['save'] = False
#             num_to_generate = default_config["general"]["num"]
#             generated = 0
#             save_directory = ''
#             threshold = default_config['general']['threshold']
#             while generated < num_to_generate:
#                 num = num_to_generate - generated

#                 sizes_batchs = gen_utils.get_batch_size(num,default_config['general']['batch_size'])
#                 print(sizes_batchs)
#                 sizes_batchs = list(filter(lambda num: num != 0, sizes_batchs))
#                 for size_batch in sizes_batchs:
#                     default_config["general"]["num"] = size_batch
#                     graphs = generator.generate('scc', default_config)
#                     curr_accepted,too_easy,too_hard = gen_utils._ensure_runtime(graphs, default_config, ref_solver, ref_task)
#                     print(f'{too_easy=} {too_hard=}')
#                     if len(curr_accepted) > 0:

#                         save_directory = gen_utils.save_instances(curr_accepted, default_config['general'],'scc',generated)
#                         generated += len(curr_accepted)
#                     if len(curr_accepted) < size_batch * threshold: # less than 20 % of graphs in batch are accepted
#                         gen_utils.tweak_configs('scc',default_config,too_easy, too_hard)
#                         break

#         else:
#             print("To ensure instance runtime you have to specify a solver and a task")
#             exit()
#     else:

#         save_directory = generator.generate('scc',default_config)
#     if add:
#         general_configs = default_config['general']
#         ctx.invoke(add_benchmark,
#                 name=general_configs['name'],
#                 path=save_directory,
#                 format=tuple(general_configs['format']),
#                 random_arguments=False,
#                 extension_arg_files="arg",
#                 generate=None
#                 )
@click.command()
@click.pass_context
@click.option("--num","-n",required=True,type=click.INT, help='Number of instances to generate')
@click.option("--name",help='Benchmark name')
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark in. Default is the current working directory.")
@click.option("--num_args","-a", help='Number of arguments.')
@click.option("--generate_solutions", "-gs", is_flag=True,help="Generate solutions for instance.")
@click.option("--generate_arg_files", "-ga", help='Generate additional argument files for DS/DC problems.')
@click.option("--solver", "-s", help='Solver (id/name) for solution generation')
@click.option("--timeout", "-t",help='Timeout for instances in seconds. Instances that exceed this value are discarded')
@click.option("--timemin","-tm",help='Minimum solving time of instance.')
@click.option("--format","-f", type=click.Choice(['apx','tgf','i23']),help='Format of instances to generate')
@click.option("--task","-t",help='Tasks to generate solutions for.')
@click.option("--random_size","-rs",nargs=2,type=click.INT,default=None, help='Random number of arguments per instances. Usage: -rs [min] [max]')
@click.option("--add",is_flag=True,help="Add benchmark to database.")
def grounded_generator(ctx,num, name, save_to, num_args, generate_solutions, generate_arg_files,solver, timeout, timemin, format, task, random_size, add):

    if not save_to:
        save_to = os.getcwd()
        ctx.params['save_to'] = save_to
    if len(random_size) < 1:
        ctx.params['random_size'] = None
    default_config =  gen_utils.read_default_config(str(definitions.GENERATOR_DEFAULTS_JSON))
    gen_utils.update_configs(ctx, default_config,generator_type='grounded')
    save_directory = generator.generate('grounded',default_config)
    if add:
        general_configs = default_config['general']
        ctx.invoke(add_benchmark,
                   name=general_configs['name'],
                   path=save_directory,
                   format=tuple(general_configs['format']),
                   random_arguments=False,
                   extension_arg_files="arg",
                   generate=None
                   )
@click.command()
@click.option("--start",'-s',is_flag=True)
@click.option("--close",'-c',is_flag=True)
def web(start, close):
    """
    Starts or closes the web interface.

    Args:
        start (bool): If True, starts the web interface.
        close (bool): If True, closes the web interface.
    """
    
    # from daemonize import Daemonize
    # pid = "/tmp/probo2_web.pid"
    # from src.webinterface import deamon_test
    # deamon_test.start()
    if start:
        from src.webinterface import web_interface
        web_interface.start()
        # subprocess.run(['python',definitions.WEB_INTERFACE_FILE])
    if close:
        subprocess.run(['python','/home/jklein/dev/probo2/src/webinterface/web_interface_shutdown.py'])


# @click.command()
# @click.option(
#     "--save_to",
#     "-st",
#     type=click.Path(exists=True, resolve_path=True),
#     help="Directory to generate files in. Default is the current working directory.")
# def tikz(save_to):
#     from os import getcwd
#     from networkx import DiGraph
#     from networkx.drawing import nx_latex

#     print("Enter a graph in format:")
#     args = []
#     attacks = []
#     attacks_section = False
#     while True:
#         try:
#             line = input()
#         except EOFError:
#             break
#         if "#" in line:
#             attacks_section = True
#             continue
#         if not attacks_section:
#             args.append(line)
#         else:
#             attacks.append(tuple(line.split(" ")))

#     af = DiGraph()
#     af.add_nodes_from(args)
#     af.add_edges_from(attacks)
#     tikz_str = nx_latex.to_latex(af,tikz_options=" > = stealth,shorten > = 1pt,auto, node distance = 3cm,semithick")
#     with open(os.path.join(save_to,'graph.tikz'),'w') as f:
#         f.write(tikz_str)

# cli.add_command(tikz)

@click.command()
@click.option('--task','-t',help='Task to solve')
@click.option('--solver','-s',help='Solver to use.')
@click.option('--arg','-a',help='Query argument for DS and DC problems.')
@click.option('--timeout',default=600)
@click.option('--file','-f',help='Input file. If not provided you can input a instances via the terminal.')
@click.option('--format','-fo',help='Instances Format')
def quick(task,solver,arg,timeout,file,format):
    """Quick run for single instances or user input instances.
    """
    from src.functions import cli_input
    from src.handler import solver_handler
    from os import getcwd,path,remove

    file_from_cli = False
    if file is None:
        user_input = cli_input.get_user_input()
        input_graph_format = cli_input.get_input_format(user_input)
        instance_path = path.join(getcwd(),f'temp_instance_file.{input_graph_format}')
        with open(instance_path,'w') as f:
            f.write("\n".join(user_input))
        file = instance_path
        format = input_graph_format
        file_from_cli = True

    if ('DS' in task or 'DC' in task) and arg is None:
        arg = input(f'Please specify a query argument for the {task} task:\n')


    if format is None:
        format = file[-3:]

    if solver is None:
    # Select any solver which supports the specified task
        all_solvers = solver_handler.load_solver('all')
        for s in all_solvers:
            if task in s['tasks']:
                solver_to_run = s
                break
    else:
        solver_to_run = solver_handler.load_solver([solver])[0]

    solver_handler.dry_run(solver_to_run,task,file,arg,format,timeout)

    # Clean up
    if file_from_cli:
        # Delete tmp file
        try:
            remove(instance_path)
        except Exception as e:
            print(f'Something went wrong when deleting the tmp file: {e}')




@click.command()
@click.option('--num_arguments',default=250,help='Number of arguments per instance')
@click.option('--num_skept_arguments', default=75, help='Number of skeptical accepted arguments')
@click.option('--size_ideal_extension',default=50)
@click.option('--num_cred_arguments',default=30)
@click.option('--num_pref_exts',default=10)
@click.option('--p_ideal_attacked',default=0.5 )
@click.option('--p_ideal_attack_back',default=0.5 )
@click.option('--p_other_skept_args_attacked',default=0.5)
@click.option('--p_other_skept_args_attack_back',default=0.5 )
@click.option('--p_cred_args_attacked',default=0.5)
@click.option('--p_cred_args_attack_back',default=0.5)
@click.option('--p_other_attacks',default=0.5)
@click.option('--num_instances',default=100)
@click.option('--name','-n',default='KWT_instances',help='Name of created benchmark.')
@click.option('--add','-a',help='Add benchmark to database.')
@click.option('--query_args',is_flag=True,help='Generate query arguments for DS,DC problems')
@click.option('--save_to','-st',type=click.Path(exists=True, resolve_path=True),help="Directory to store benchmark in. Default is the current working directory.")
@click.option('--format','-fo',multiple=True,help='Output format of graphs')
@click.option('--config','-cfg',type=click.Path(exists=True, resolve_path=True),help="Full path to config file")
@click.option('--extension_query','-ext',default='arg',help='Extension of query arguments')
def kwt_gen(num_arguments,
            num_skept_arguments,
            size_ideal_extension,
            num_cred_arguments,
            num_pref_exts,
            p_ideal_attacked,
            p_ideal_attack_back,
            p_other_skept_args_attacked,
            p_other_skept_args_attack_back,
            p_cred_args_attacked,
            p_cred_args_attack_back,
            p_other_attacks,
            num_instances,
            save_to,
            format,
            config,
            add,
            name,
            query_args,extension_query):
    """Generate instances using the kwt generator"""
    from src.generators import kwt_generator
    from src.generators import generator_utils
    from tqdm import tqdm
    from random import choice
    import yaml
    if config is not None:
        with open(config,'r') as config_file:
            if config.endswith('yaml'):
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        kwt_generator = kwt_generator.KwtDungTheoryGenerator(**config['generator'])
        name = config['general']['name']
        format = config['general']['format']
        add = config['general']['add']
        query_args = config['general']['query_args']
        extension_query = config['general']['extension_query']
        save_to = config['general']['save_to']
        num_instances = config['general']['num_instances']
    else:

        kwt_generator = kwt_generator.KwtDungTheoryGenerator(num_arguments=num_arguments,
                                                         num_skept_arguments=num_skept_arguments,
                                                         size_ideal_extension=size_ideal_extension,
                                                         num_cred_arguments=num_cred_arguments,
                                                         num_pref_exts=num_pref_exts,
                                                         p_ideal_attacked=p_ideal_attacked,
                                                         p_ideal_attack_back=p_ideal_attack_back,
                                                         p_other_skept_args_attacked=p_other_skept_args_attacked,
                                                         p_other_skept_args_attack_back=p_other_skept_args_attack_back,
                                                         p_cred_args_attacked=p_cred_args_attacked,
                                                         p_cred_args_attack_back=p_cred_args_attack_back,
                                                         p_other_attacks=p_other_attacks)
    if save_to is None:
        save_to = os.getcwd()
    kwt_instances_path = os.path.join(save_to,name)
    os.makedirs(kwt_instances_path,exist_ok=True)
    for i in tqdm(range(0,num_instances),desc='Generating kwt instances'):
        af = kwt_generator.generate_instance()
        if query_args:
            query_arg = choice(list(af.nodes()))
            af_query_path = os.path.join(kwt_instances_path,f'kwt_{i}.{extension_query}')
            with open(af_query_path,'w') as af_query_file:
                af_query_file.write(query_arg)

        for f in format:
            af_parsed = generator_utils.parse_graph(f,af)
            af_path = os.path.join(kwt_instances_path,f'kwt_{i}.{f}')
            with open(af_path,'w') as af_file:
                af_file.write(af_parsed)

    if add:
        from src.handler.benchmark_handler import add_benchmark
        from src.utils.options.CommandOptions import AddBenchmarkOptions


        options = AddBenchmarkOptions(name=name,path=kwt_instances_path,format=format,additional_extension=extension_query,no_check=False,dynamic_files=None,random_arguments=False,generate=None,function=None,yes=True)
        options.check()
        add_benchmark(options)


@click.command()
@click.pass_context
@click.option("--id",type=click.INT, help='ID of solver to edit')
@click.option('--name','-n',type=click.STRING,help='Edit name of solver')
@click.option('--version','-v',type=click.STRING,help='Edit version of solver')
@click.option('--path','-p',type=click.Path(exists=True, resolve_path=True),help='Edit path of solver')
@click.option('--tasks','-t',help='Edit tasks of solver')
def edit_solver(ctx,id, name, version, path, tasks):
    """Edit solver in database.
    """
    from src.utils.options.CommandOptions import EditSolverOptions
    from src.handler import solver_handler
    options = EditSolverOptions(**ctx.params)
    solver_handler.edit_solver(options)


cli.add_command(edit_solver)

@click.command()
@click.pass_context
@click.option("--id",type=click.INT,required=True, help='ID of benchmark to edit')
@click.option('--name','-n',type=click.STRING,help='Edit name attribute of benchmark')
@click.option('--format','-v',multiple=True,default=None,help='Edit formats of benchmark')
@click.option('--path','-p',type=click.Path(exists=True, resolve_path=True),help='Edit path attribute of benchmark')
@click.option('--ext_additional','-ext',help='Extension of query arguments files.')
@click.option('--dynamic_files','-df', help='Edit flag for dynamic files')
def edit_benchmark(ctx,id, name,path,format,ext_additional, dynamic_files):
    """Edit a benchmark in the database.
    """
    from src.utils.options.CommandOptions import EditBenchmarkOptions
    from src.handler import benchmark_handler
    if not format:
        ctx.params['format'] = None
    options = EditBenchmarkOptions(**ctx.params)
    benchmark_handler.edit_benchmark(options)

cli.add_command(edit_benchmark)




@click.command()
@click.pass_context
@click.option("--id",type=click.INT,required=True, help='ID of benchmark to convert')
@click.option('--name','-n',type=click.STRING,help='Name of new generated benchmark. If not specified format suffix is added to old name.')
@click.option('--formats','-f',type=click.STRING,multiple=True,required=True,default=None,help='Formats to convert selected benchmark to. For each format a sperate benchmark is created.')
@click.option('--save_to','-st',type=click.Path(exists=True, resolve_path=True),help='Directory to store converted benchmark in. Default is the current working directory.')
@click.option('--add','-a',type=click.BOOL,is_flag=True,help='Add generated benchmark to database.')
@click.option('--skip_args','-s',type=click.BOOL, is_flag=True,help='Skip the creation of argument files.')
def convert_benchmark(ctx,id, name,formats,save_to, add, skip_args):
    """Convert benchmarks to different formats including the query argument files.
    """
    from src.utils.options.CommandOptions import ConvertBenchmarkOptions,AddBenchmarkOptions
    from src.handler import benchmark_handler
    from os import getcwd
    if not save_to:
        save_to = getcwd()
    
    options = ConvertBenchmarkOptions(id=id,benchmark_name=name,formats=formats,save_to=save_to,add=add,skip_args=skip_args)
    converted_benchmark = benchmark_handler.convert_benchmark(options)

    if add:
        add_options = AddBenchmarkOptions(name=converted_benchmark.name,
                                      path=converted_benchmark.path,
                                      format=converted_benchmark.format,
                                      additional_extension=converted_benchmark.ext_additional,
                                      no_check=False,
                                      generate=None,
                                      random_arguments=False,
                                      dynamic_files=False,
                                      function=None,
                                      yes=False,
                                      references_path=None,
                                      extension_references=None,
                                      has_references=False)
        add_options.check()
        benchmark_handler.add_benchmark(options=add_options)

cli.add_command(convert_benchmark)




cli.add_command(kwt_gen)
cli.add_command(quick)
cli.add_command(web)
cli.add_command(grounded_generator)
# cli.add_command(scc_generator)
# cli.add_command(stable_generator)
cli.add_command(fetch)
cli.add_command(logs)
#cli.add_command(dumphelp_markdown)
cli.add_command(experiments)
cli.add_command(last)
cli.add_command(delete_benchmark)
cli.add_command(add_solver)
cli.add_command(add_benchmark)
cli.add_command(benchmarks)
cli.add_command(solvers)
cli.add_command(run)
cli.add_command(calculate)
cli.add_command(plot)
cli.add_command(status)
cli.add_command(validate)
cli.add_command(significance)
cli.add_command(delete_solver)
cli.add_command(version)
