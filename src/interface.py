import datetime
import hashlib
import itertools
import json
import logging
import os
import pathlib
from re import S
from socket import getnameinfo
import sys
import timeit
from email.policy import default
from glob import glob
from importlib.resources import path
from itertools import chain
from random import choice
import time
from urllib import request
from zipfile import Path

import click
from matplotlib.pyplot import step
from numpy import append, nanargmax, number
import pandas as pd
import numpy as np
import psutil
import tabulate
from click.core import iter_params_for_processing
from click.decorators import command
from sqlalchemy import and_, engine, or_
from sqlalchemy.sql.expression import false
from tabulate import tabulate
from src import functions

import src.analysis.statistics as stats
import src.analysis.validation as validation
import src.database_models.DatabaseHandler as DatabaseHandler
import src.plotting.plotting_utils as pl_util
import src.utils.CustomClickOptions as CustomClickOptions
import src.utils.definitions as definitions
import src.utils.Status as Status
from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Benchmark import Benchmark
from src.database_models.Result import Result
from src.database_models.Solver import Solver
from src.utils import benchmark_handler, definitions, experiment_config, fetching, utils
from src.utils.Notification import Notification
from src.generators import generator_utils as gen_utils
from src.generators import generator
import src.functions.register as register
from src.functions import benchmark
from src.utils import solver_handler, experiment_handler
from src.utils import experiment_config


#TODO: Dont save files when save_to not speficied and send is specified, Ausgabe für command benchmarks und solvers überarbeiten, Logging system,
logging.basicConfig(filename=str(definitions.LOG_FILE_PATH),format='[%(asctime)s] - [%(levelname)s] : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

def init():
    print('init')
    if not os.path.exists(definitions.SOLVER_FILE_PATH):
        file = open(definitions.SOLVER_FILE_PATH,'w+')





@click.group()
def cli():
    pass
    #init()
    # if not os.path.exists(definitions.DATABASE_DIR):
    #     os.makedirs(definitions.DATABASE_DIR)
    #     logging.info("Database directory created.")

    # if not os.path.exists(definitions.TEST_DATABASE_PATH):
    #     engine = DatabaseHandler.get_engine()
    #     DatabaseHandler.init_database(engine)
    # DatabaseHandler.check_database_version()



@click.command(cls=CustomClickOptions.command_required_option_from_option('guess'))
@click.option("--name", "-n", required=True, help="Name of the solver")
@click.option("--path","-p",
              required=True,
              type=click.Path(exists=True,resolve_path=True),
              help="Path to solver executable")
@click.option("--format",
              "-f",
              type=click.Choice(['apx', 'tgf'], case_sensitive=False),
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
              required=True,
              help="Version of solver.")
@click.option(
    "--guess","-g",
    is_flag=True,
    help="Pull supported file format and computational problems from solver.")
def add_solver(name, path, format, tasks, version, guess):
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
    if guess:
        try:
            if not tasks:
                tasks = solver_handler.fetch_tasks(path)
            if not format:
                format = solver_handler.fetch_format(path)
        except ValueError as e:
            print(e)
            exit()
    solver_info = {'name': name,'version': version,
                   'path': path,'tasks': tasks,'format': format}
    print(solver_info)
    is_working = solver_handler.check_interface(solver_info)

    if is_working:
        solver_handler.print_summary(solver_info)
        click.confirm(
            "Are you sure you want to add this solver to the database?"
            ,abort=True,default=True)
        id = solver_handler.add_solver(solver_info)
        print(f"Solver {solver_info['name']} added with ID: {id}")
        logging.info(f"Solver {solver_info['name']} added with ID: {id}")

@click.command()
@click.option("--name",
              "-n",
              type=click.types.STRING,
              required=True,
              help="Name of benchmark/fileset")
@click.option("--path",
              "-p",
              type=click.Path(exists=True,resolve_path=True),
              required=True,
              help="Path to instances")
@click.option("--format",
              "-f",
              required=True,
              multiple=True,
              type=click.Choice(["apx","tgf"]),
              help="Supported formats of benchmark/fileset")
@click.option(
    "--extension_arg_files",
    "-ext",
    type=click.types.STRING,
    default='arg',
    help="Extension of additional argument parameter for DC/DS problems.")
@click.option("--no_check",
              is_flag=True,
              help="Checks if the benchmark is complete.")
@click.option("--generate",
              "-g",
              type=click.types.Choice(['apx', 'tgf']),
              help="Generate instances in specified format")
@click.option("--random_arguments",
              "-rnd",
              is_flag=True,
              help="Generate additional argument files with a random argument."
              )
@click.option("--function",'-fun', type=click.Choice(register.benchmark_functions_dict.keys()),multiple=True )
def add_benchmark(name, path, format, extension_arg_files, no_check, generate, random_arguments, function):
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
    benchmark_info = {'name': name, 'path': path,'format': list(format),'ext_additional': extension_arg_files}

    if generate:
        benchmark_handler.generate_instances(benchmark_info, generate)
        #new_benchmark.generate_instances(generate)
    if random_arguments:
        benchmark_handler.generate_argument_files(benchmark_info,extension=extension_arg_files)
    if not no_check:
        benchmark_handler.check(benchmark_info)

    for fun in function:
        benchmark_info.update(register.benchmark_functions_dict[fun](benchmark_info))

    benchmark_handler.print_summary(benchmark_info)
    click.confirm(
        "Are you sure you want to add this benchmark to the database?",
        abort=True,default=True)
    benchmark_handler.add_benchmark(benchmark_info)
    print(f"Benchmark {benchmark_info['name']} added to database with ID: {benchmark_info['id']}.")
    logging.info(f"Benchmark {benchmark_info['name']} added to database with ID: {benchmark_info['id']}.")
    print(benchmark_handler.load_benchmark('all'))


@click.command()
@click.option(
    "--all",
    "-a",
    required=False,
    is_flag=True,
    help="Execute all solvers supporting the specified tasks on specified instances.")
@click.option("--solver",
              "-s",
              required=False,
              default=[],
              cls=CustomClickOptions.StringAsOption,
              help=" Comma-seperated list of ids or names of solvers (in database) to run.")
@click.option(
    "--benchmark",
    "-b",
    required=True,
    cls=CustomClickOptions.StringAsOption,
    default=[],
    help="Comma-seperated list of ids or names of benchmarks (in database) to run solvers on.")
@click.option("--task",
              cls=CustomClickOptions.StringAsOption,
              required=False,
              #callback=CustomClickOptions.check_problems,
              help="Comma-seperated list of tasks to solve.")
@click.option("--timeout",
              "-t",
              required=False,
              default=600,
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
    "--tag",
    required=False,
    help=
    "Tag for individual experiments.This tag is used to identify the experiment."
)
@click.option(
    "--notify",
    help=
    "Send a notification to the email address provided as soon as the experiments are finished."
)
@click.option("--n_times","-n",required=False,type=click.types.INT,default=1, help="Number of repetitions per instance. Run time is the avg of the n runs.")
@click.option("--rerun",'-rn',is_flag=True, help='Rerun last experiment')
@click.option("--subset","-sub",type=click.types.INT, help="Run only the first n instances of a benchmark.")
@click.option("--multi", is_flag=True,help="Run experiment on mutiple cores.")
@click.pass_context
def run(ctx, all,benchmark, task, solver, timeout, dry, tag,
        notify, track, n_times, rerun, subset, multi):
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
    config = experiment_config.load_config(os.path.join(definitions.CONFIGS_DIRECTORY,'WS_test.yaml'))
    print(config.__dict__)
    experiment_handler.run_experiment(config)
    exit()


    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    run_parameter = ctx.params.copy()
    if rerun:
        last_json = utils.get_last_experiment_json()
        utils.set_run_parameters(run_parameter, last_json)



    if run_parameter['tag'] is None and not(dry):
        print("Please specify a experiment tag via the --tag option.")
        sys.exit()
    if DatabaseHandler.tag_in_database(session, run_parameter['tag']) and not(dry):
        print("Tag is already used. Please use another tag.")
        sys.exit()



    benchmarks = DatabaseHandler.get_benchmarks(session, run_parameter['benchmark'])
    if run_parameter['track']:
        tasks = DatabaseHandler.get_tasks(session, run_parameter['track'])
    else:

        tasks = DatabaseHandler.get_tasks(session, run_parameter['task'])



    run_parameter['benchmark'] = benchmarks
    run_parameter['task'] = tasks
    run_parameter['session'] = session

    if run_parameter['solver']:
        run_parameter['select'] = True
        solvers =  DatabaseHandler.get_solvers(session, run_parameter['solver'])
        #Status.init_status_file(tasks, benchmarks, tag, solvers)
        run_parameter['solver'] = solvers

    else:
        run_parameter['select'] = False
        solvers_all_tasks = [ (t.solvers) for t in tasks]
        solver_to_run = [ solver for sub_list_solver in solvers_all_tasks for solver in sub_list_solver]
        run_parameter['solver'] = solver_to_run

    run_parameter['solver'] = utils.check_solver_paths(run_parameter['solver'])

    Status.init_status_file(tasks, benchmarks, run_parameter['tag'], run_parameter['solver'])

    tag = run_parameter['tag']

    logging.info(f"Started to run Experiment {tag}")
    if multi:
        start = time.time()
        utils._multiprocess_run_experiment(run_parameter)
        end = time.time()
        print(f'\nTotal runtime:{end - start}')
    else:
        start = time.time()
        utils.run_experiment(run_parameter)
        end = time.time()
        print(f'\nTotal runtime:{end - start}')
    logging.info(f"Finished to run Experiment {tag}")
    df = stats.prepare_data(
        DatabaseHandler.get_results(session, [], [], [], [tag],
                                    None))

    if not dry:
        if not df.empty:
            summary = stats.get_experiment_summary_as_string(df)
            print("")
            print(summary)
            with open(definitions.LAST_EXPERIMENT_SUMMARY_JSON_PATH,'w') as f:
                f.write(summary)
            now = datetime.datetime.now()
            mode = 'select' if run_parameter['select'] else 'all'



            last_experiment_dict = {'Tag':  run_parameter['tag'],
                                    'Benchmarks' : list(df.benchmark_name.unique()),
                                    'Benchmark IDs': [ int(b_id) for b_id in list(df.benchmark_id.unique())],
                                    'Tasks' : list(df.task.unique()),
                                    'Track' :  run_parameter['track'],
                                    'Solvers': list(df.solver_full_name.unique()),
                                    'Solver IDs': [ int(s_id) for s_id in list(df.solver_id.unique())],
                                    'Finished': now.strftime("%m/%d/%Y, %H:%M:%S"),
                                    'Timeout':  run_parameter['timeout'],
                                    'n_times': run_parameter['n_times'],
                                    'Mode': mode,
                                    'Notify': run_parameter['notify'],
                                    'Dry': run_parameter['dry']

                                   }



            with open(definitions.LAST_EXPERIMENT_JSON_PATH,'w') as f:
                json.dump(last_experiment_dict,f)
        else:
            summary = 'Something went wrong. Please check the logs for more information.'

    if run_parameter['notify']:
        id_code = int(hashlib.sha256(tag.encode('utf-8')).hexdigest(), 16) % 10**8
        notification = Notification(run_parameter['notify'],message=f"Here a little summary of your experiment:\n{summary}",id=id_code)

        notification.send()
        send_to = run_parameter['notify']
        logging.info(f'Sended Notfication e-mail to {send_to}\nID: {id_code}')
        print(f"\n{notification.foot}\nYour e-mail identification code: {id_code}")

@click.command()
@click.option("--par",
              "-p",
              type=click.types.INT,
              help="Penalty multiplier for PAR score")
@click.option("--solver",
              "-s",
              required=True,
              cls=CustomClickOptions.StringAsOption,
              default=[],
              help="Comma-separated list of solver ids")
@click.option("--filter",
              "-f",
              cls=CustomClickOptions.FilterAsDictionary,
              multiple=True)
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--benchmark",
              "-b",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--print_format",
              "-pfmt",default='fancy_grid',
              type=click.Choice([
                  "plain", "simple", "github", "grid", "fancy_grid", "pipe",
                  "orgtbl", "jira", "presto", "pretty", "psql", "rst",
                  "mediawiki", "moinmoin", "youtrack", "html", "unsafehtml"
                  "latex", "latex_raw", "latex_booktabs", "textile"
              ]))
@click.option("--tag", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--combine",
              "-c",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--vbs", is_flag=True, help="Create virtual best solver")
@click.option("--save_to", "-st",type=click.Path(resolve_path=True,exists=True), help="Directory to store tables")
@click.option("--export","-e",type=click.Choice(['latex','csv','json']),default=None,multiple=True)
#@click.option("--css",default="styled-table.css",help="CSS file for table style.")
@click.option("--statistics",'-s',type=click.Choice(['mean','sum','min','max','median','var','std','coverage','timeouts','solved','errors','all','best']),multiple=True)
@click.option("--verbose",'-v',is_flag=True,help='Show additional information for some statistics.')
@click.option("--last", "-l",is_flag=True,help="Calculate stats for the last finished experiment.")
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
@click.option("--send", required=False, help="Send plots via E-Mail.")
@click.option("--custom","-cs",type=click.Choice(register.stat_dict.keys()),multiple=True)
def calculate(par, solver, task, benchmark,
              tag,combine, vbs, export, save_to, statistics,print_format,filter,last, send, compress, verbose, custom):
    if last:
        tag.append(utils.get_from_last_experiment("Tag"))


    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_id']
    export_columns = ['tag','solver','task','benchmark']

    if combine:
        grouping = [x for x in grouping if x not in combine]

    if 'all' in statistics:
        functions_to_call = ['mean','sum','min','max','median','var','std','coverage','timeouts','solved','errors','best']
    else:
        functions_to_call = list(statistics)

    df = stats.prepare_data(
        DatabaseHandler.get_results(session, solver, task, benchmark, tag,
                                    filter))




    if vbs:
        grouping_vbs = ['tag', 'task_id', 'benchmark_id', 'instance']
        vbs_id = -1
        vbs_df = df.groupby(grouping_vbs,as_index=False).apply(lambda df: stats.create_vbs(df,vbs_id))
        df = df.append(vbs_df)

    if par:
        functions_to_call.append(f'PAR{par}')
    else:
        par=0
    custom_stats_df = []
    for custom_stat_func in custom:
        custom_stats_df.append(register.stat_dict[custom_stat_func](df))
    for _df in custom_stats_df:
        print(tabulate(_df[_df.columns.values],headers='keys',tablefmt=print_format,showindex=False))

    if functions_to_call:
        stats_df = stats.calculate_stats(df,grouping,functions_to_call,par_penalty=par)
    # stats_df = (df
    #             .groupby(grouping,as_index=False)
    #             .apply(lambda df: stats.dispatch_function(df,functions_to_call,par_penalty=par))
    #             )


    if functions_to_call:
        print_headers = ['solver','task','benchmark']
        print_headers.extend(functions_to_call)
        utils.print_df(stats_df,['tag','benchmark','task'],headers=print_headers,format=print_format)

    if verbose:
        verbose_grouping = grouping.copy()
        if 'solver_id' in verbose_grouping:
            verbose_grouping.remove('solver_id')
        verbose_stats_to_call = set.intersection(set(functions_to_call),{'timeouts','errors','solved'})
        for to_call in verbose_stats_to_call:
            df.groupby(verbose_grouping).apply(lambda _df: stats._verbose_output(to_call,_df))

    saved_files = []
    if export:
        for export_format in list(export):
            saved_files.append(stats_df.groupby(['tag','benchmark','task']).apply(lambda df_: stats.export(export_format,df_,save_to, functions_to_call,par)))

    saved_files = list(itertools.chain(*[x.values for x in saved_files]))
    if compress:
        archive_name = "_".join(tag)

        archive_save_path = utils.create_archive_from_files(save_to,archive_name,saved_files,compress,'calculations')

    if send:
        id_code = int(hashlib.sha256(save_to.encode('utf-8')).hexdigest(), 16) % 10**8
        email_notification = Notification(send,subject="Hi, there. I have your files for you.",message=f"Enclosed you will find your files.",id=id_code)
        if compress:
            email_notification.attach_file(f'{archive_save_path}.{compress}')
        else:
            email_notification.attach_mutiple_files(saved_files)
        email_notification.send()
        print(f"\n{email_notification.foot}\nYour e-mail identification code: {id_code}")






@click.command(cls=CustomClickOptions.command_required_tag_if_not('last'))
@click.pass_context
@click.option("--tag", "-t",cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--task",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Comma-separated list of task IDs or symbols to be selected.")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[],help="Comma-separated list of task IDs or symbols to be selected.")
@click.option("--solver",
              "-s",
              cls=CustomClickOptions.StringAsOption,
              default=[],
              help="Comma-separated list of solver IDs or names to be selected.")
@click.option("--filter",
              "-f",
              cls=CustomClickOptions.FilterAsDictionary,
              multiple=True,
              help="Filter results in database. Format: [column:value]")
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store plots in. Filenames will be generated automatically.")
@click.option("--vbs", is_flag=True, help="Create virtual best solver.")
#@click.option("--x_max", "-xm", type=click.types.INT)
#@click.option("--y_max", "-ym", type=click.types.INT)
#@click.option("--alpha",
            #   "-a",
            #   type=click.FloatRange(0, 1),
            #   help="Alpha value (only for scatter plots)")
@click.option("--backend",
              "-b",
              type=click.Choice(['pdf', 'pgf', 'png', 'ps', 'svg']),
              default='png',
              help="Backend to use")
# @click.option("--no_grid", "-ng", is_flag=True, help="Do not show a grid.")
# @click.option("--grid_plot",is_flag=True)
@click.option("--combine",
              "-c",
              type=click.Choice(['tag','task_id','benchmark_id']),help='Combine results on specified key.')
@click.option("--kind",'-k',type=click.Choice(['cactus','count','dist','pie','box','scatter','all']),multiple=True)
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
@click.option("--send", required=False, help="Send plots via E-Mail.")
@click.option("--last", "-l",is_flag=True,help="Plot results for the last finished experiment.")
@click.option("--axis_scale",'-as',type=click.Choice(['linear','log']),default='log',help="Scale of x and y axis." )
@click.option("--set_default", is_flag=True)
def plot(ctx, tag, task, benchmark, solver, save_to, filter, vbs, backend,combine, kind, compress, send, last, axis_scale, set_default):
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
    if set_default and (save_to is not None):
        def_options =  pl_util.read_default_options(str(definitions.PLOT_JSON_DEFAULTS))
        def_options['settings']['default_dir'] = save_to
        with open(def_options['def_path'],'w') as option_file:
            json.dump(def_options,option_file,indent=4)
        print(f'Default path for plots set to: {save_to}')
        exit()

    default_options = pl_util.read_default_options(str(definitions.PLOT_JSON_DEFAULTS))
    pl_util.set_user_options(ctx, default_options)
    if not save_to:
        if default_options['settings']['default_dir'] is not None:
            save_to = default_options['settings']['default_dir']
        else:
            save_to = os.getcwd()
    if last:
        tag.append(utils.get_from_last_experiment("Tag"))



    if 'all' in kind:
        kind = ['cactus','count','dist','scatter','pie','box']

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_id']
    if combine:
        grouping = [x for x in grouping if x not in combine]

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    og_df =  DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        benchmark,
                                        tag,
                                        filter,
                                        only_solved=False)

    df = stats.prepare_data(og_df)
    if vbs:
        grouping_vbs = ['tag', 'task_id', 'benchmark_id', 'instance']
        stats.create_and_append_vbs(df, grouping_vbs)
    saved_files = []


    for plot_kind in list(kind):
        saved_files.append(pl_util.create_plots(plot_kind, df, save_to, default_options, grouping))
    saved_files_df = pd.concat(saved_files).to_frame().rename(columns={0:'saved_files'})

    saved_files_paths = []
    for file in saved_files_df.saved_files.to_list():
        if isinstance(file,list):
            saved_files_paths.extend([  f'{x}.{backend}' for x in file ])
        else:
            saved_files_paths.append( f'{file}.{backend}')


    if compress:
        archive_name = "_".join(tag)

        archive_save_path = pl_util.create_archive_from_plots(save_to,archive_name,saved_files_paths,compress)

    if send:
        id_code = int(hashlib.sha256(save_to.encode('utf-8')).hexdigest(), 16) % 10**8
        email_notification = Notification(send,subject="Hi, there. I have your files for you.",message=f"Enclosed you will find your files.",id=id_code)
        if compress:
            email_notification.attach_file(f'{archive_save_path}.{compress}')
        else:
            email_notification.attach_mutiple_files(saved_files_paths)
        email_notification.send()
        print(f"\n{email_notification.foot}\nYour e-mail identification code: {id_code}")




@click.command()
@click.option("--verbose","-v",is_flag=True,help="Prints additional information on benchmark")
def benchmarks(verbose):
    """ Prints benchmarks in database to console.

    Args:
        None
    Return:
        None
    Raises:
        None
   """
    benchmark_handler.print_benchmarks()
    # engine = DatabaseHandler.get_engine()
    # session = DatabaseHandler.create_session(engine)
    # benchmarks = session.query(Benchmark).all()
    # tabulate_data = []
    # if not verbose:
    #     for benchmark in benchmarks:
    #         tabulate_data.append([
    #             benchmark.id, benchmark.benchmark_name, benchmark.format_instances
    #         ])
    #     print(tabulate(tabulate_data, headers=["ID", "Name", "Format"]))

    # else:
    #     for benchmark in benchmarks:
    #         b_formats = benchmark.get_formats()
    #         b_formats.append(benchmark.extension_arg_files)
    #         num_instances = []
    #         for f in b_formats:
    #             num_instances.append(len(benchmark.get_instances(f)))
    #         tabulate_data.append([
    #             benchmark.id, benchmark.benchmark_name, benchmark.format_instances, *num_instances, benchmark.benchmark_path
    #         ])

    #     str_b_formats = [f'#{x}' for x in b_formats]
    #     print(tabulate(tabulate_data,headers=["ID", "Name", "Format",*str_b_formats,'Path']))
    # session.close()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
@click.option("--id",help="Print summary of solver with specified id")
@click.option("--path",is_flag=True)
def solvers(verbose,id,path):
    """Prints solvers in database to console.

    Args:
        verbose ([type]): [description]
    """
    solver_handler.print_solvers()



@click.command()
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
@click.option("--task",
              "-t",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--benchmark",
              "-b",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--tag",
              "-t",
              required=False,
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
@click.option("--only_tags",is_flag=True)
def results(verbose, solver, task, benchmark, tag, filter,only_tags):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    result_df = DatabaseHandler.get_results(session,
                                            solver,
                                            task,
                                            benchmark,
                                            tag,
                                            filter,
                                            only_solved=False)
    if only_tags:
        print(",".join(list(result_df.tag.unique())))

    print(result_df[['correct_solved','incorrect_solved','no_reference']])

    # results = session.query(Result).all()
    # tabulate_data = []
    # if verbose:
    #     for result in results:
    #         tabulate_data.append(
    #             [result.instance, result.task.symbol, result.runtime, result.solver.solver_full_name, result.tag,
    #              result.exit_with_error, result.error_code, result.benchmark.benchmark_name])
    #     print(tabulate.tabulate(tabulate_data,
    #                             headers=["INSTANCE", "TASK", "RUNTIME", "SOLVER", "TAG", "EXIT_WITH_ERROR",
    #                                      "ERROR_CODE", "BENCHMARK"], tablefmt=format))
    # else:
    #     for result in results:
    #         tabulate_data.append(
    #             [result.instance, result.task.symbol, result.runtime, result.solver.solver_full_name, result.tag])
    #     print(
    #         tabulate.tabulate(tabulate_data, headers=["INSTANCE", "TASK", "RUNTIME", "SOLVER", "TAG"], tablefmt=format))
    session.close()


@click.command()
@click.option("--save_to",
    "-st",
    type=click.Path(resolve_path=True,exists=True),
    required=False,
    help=
    "Directory to store plots and data in. Filenames will be generated automatically.")
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
@click.option("--problem",
              "-p",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--task",
              "-t",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--benchmark",
              "-b",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--tag",
              "-t",
              required=False,
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--format",
              type=click.Choice(['csv', 'json', 'xlsx']),
              default='csv')
@click.option("--group_by",
              "-g",
              type=click.Choice(
                  ['tag', 'solver_name', 'benchmark_name', 'symbol']))
@click.option("--file_name", required=False, default='data')
@click.option("--include_column",
              required=False,
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--exclude_column",
              required=False,
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--only_solved", is_flag=True, default=False)
@click.option("--all_columns",is_flag=True)
@click.option("--last",'-l',is_flag=True,help='Export results of last experiment')
def export(save_to, solver, filter, problem, benchmark, tag, task, format,
           group_by, file_name, include_column, exclude_column, only_solved,all_columns,last):

    if last:
        tag.append(utils.get_from_last_experiment("Tag"))
    if not save_to:
        save_to = os.getcwd()
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    colums_to_export = [
        'id', 'tag', 'solver_id', 'benchmark_id', 'task_id', 'instance',
        'cut_off', 'timed_out', 'exit_with_error', 'runtime',
        'additional_argument', 'anon_1', 'benchmark_name', 'symbol'
    ]
    # ['solver_name','solver_version','benchmark_name','instance','timed_out','runtime','symbol','additional_argument','tag','exit_with_error']

    result_df = DatabaseHandler.get_results(session,
                                            solver,
                                            task,
                                            benchmark,
                                            tag,
                                            filter,
                                            only_solved=only_solved)


    if include_column:
        colums_to_export.extend(include_column)
    if exclude_column:
        colums_to_export = [
            ele for ele in colums_to_export if ele not in exclude_column
        ]
    if all_columns:
         export_df = result_df
    else:

        export_df = result_df[colums_to_export]
    # export_df = result_df

    if group_by:
        grouped = export_df.groupby(group_by)
        for name, group in grouped:
            group.to_csv(os.path.join(save_to, '{}.zip'.format(name)),
                         index=False)

    file = os.path.join(save_to, file_name)
    if format == 'xlsx':
        export_df.to_excel("{}.{}".format(file, 'xlsx'), index=False)
    if format == 'csv':
        export_df.to_csv("{}.{}".format(file, 'csv'), index=False)
    if format == 'json':
        export_df.to_json("{}.{}".format(file, 'json'))


@click.command()
def status():
    """Provides an overview of the progress of the currently running experiment.

    """
    if os.path.exists(str(definitions.STATUS_FILE_DIR)):
        Status.print_status_summary()
    else:
        print("No status query is possible.")


@click.command()
@click.option("--tag",help="Experiment tag to be validated")
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Comma-separated list of task IDs or symbols to be validated.")
@click.option("--benchmark","-b", required=True,help="Benchmark name or id to be validated.")
@click.option("--solver",
              "-s",
              required=True,
              cls=CustomClickOptions.StringAsOption,
              default=[],
              help="Comma-separated list of solver IDs or names to be validated.")
@click.option("--filter",
              "-f",
              cls=CustomClickOptions.FilterAsDictionary,
              multiple=True,
              help="Filter results in database. Format: [column:value]")
@click.option("--reference",
              "-r",
              type=click.Path(resolve_path=True,exists=True),
              help="Path to reference files.")
@click.option("--update_db", is_flag=True,help="Update instances status (correct, incorrect, no reference) in database.")
@click.option("--pairwise", "-pw", is_flag=True,help="Pairwise comparision of results. Not supported for the SE task.")
@click.option('--export',
              '-e',
              type=click.Choice(
                  ['json', 'latex','csv']),
              multiple=True,help="Export results in specified format." )
@click.option("--raw", is_flag=True, help='Export raw validation results in csv format.')
@click.option('--plot',
              '-p',
              type=click.Choice(
                  ['heatmap','count']),
              multiple=True,help="Create a heatmap for pairwise comparision results and a count plot for validation with references.")
@click.option(
    "--save_to",
    "-st",
    type=click.Path(resolve_path=True,exists=True),
    required=False,
    help=
    "Directory to store plots and data in. Filenames will be generated automatically.")

@click.option('--extension','-ext',multiple=True, help="Reference file extension")
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
@click.option("--send", required=False, help="Send plots and data via E-Mail.")
@click.option("--verbose","-v", is_flag=True,help="Verbose output for validation with reference. For each solver the instance names of not validated and incorrect instances is printed to the console.")
def validate(tag, task, benchmark, solver, filter, reference, pairwise,
             save_to, export,plot, update_db,extension,compress,send, verbose,raw):
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
    if not save_to:
        save_to = os.getcwd()
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    og_df = DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        [benchmark],
                                        [tag],
                                        filter,
                                        only_solved=True)
    result_df = validation.prepare_data(og_df)

    saved_files = []



    if pairwise:
        pairwise_result = result_df.groupby(['tag','benchmark_id','task']).apply(lambda df_: validation._validate_pairwise(df_)).reset_index()
        validation.print_validate_pairwise(pairwise_result)

        if export:
            for export_format in list(export):
                saved_files.append(pairwise_result.groupby(['tag','benchmark_name','task']).apply(lambda df_: validation.export_pairwise(export_format,df_,save_to,file_name_suffix='pairwise')))

        if plot:
            for kind in list(plot):
                saved_files.append(pairwise_result.groupby(['tag','benchmark_name','task']).apply(lambda df_: validation.plot_pairwise(kind,df_,save_to)))

    else:
        unique_tasks = list(result_df.task.unique())

        # Replace SE with EE for reference as SE problems are not unique
        for i,t in enumerate(unique_tasks):
            if 'SE' in t:
                unique_tasks[i] = unique_tasks[i].replace('SE','EE')

        ref_dict = validation.get_reference(extension, unique_tasks, reference)
        validation_results = validation.validate(result_df, ref_dict, extension)
        analyse = validation_results.groupby(['tag','benchmark_id','task_id','solver_id'],as_index=False).apply(lambda df: validation.analyse(df) )
        validation.print_validate_with_reference_results(analyse)
        if verbose:
            validation.print_not_validated(validation_results)




        if export:
            if raw:
                saved_files.append(validation_results.groupby(['tag','benchmark_name','task']).apply(lambda df_: validation.export('csv',df_,save_to,file_name_suffix='raw')))


            for export_format in list(export):
                saved_files.append(analyse.groupby(['tag','benchmark_name','task']).apply(lambda df_: validation.export(export_format,df_,save_to)))
        if plot:
            for kind in list(plot):
                saved_files.append(validation_results.groupby(['tag','benchmark_name','task']).apply(lambda df_: validation.plot(kind,df_,save_to)))




        if update_db and not pairwise:
            og_df.set_index("id", inplace=True)
            og_df.no_reference.update(validation_results.set_index("id").no_reference)
            og_df.correct_solved.update(validation_results.set_index("id").correct)
            og_df.incorrect_solved.update(validation_results.set_index("id").incorrect)

            result_objs = session.query(Result).filter(
                Result.tag.in_(tag)).all()
            id_result_mapping = dict()
            for res in result_objs:
                id_result_mapping[res.id] = res
            og_df.apply(lambda row: validation.update_result_object(id_result_mapping[row.name], row.correct_solved,row.incorrect_solved,row.no_reference),
                        axis=1)
            session.commit()

    saved_files = list(itertools.chain(*[x.values for x in saved_files]))

    if compress:
        if pairwise:
            archive_name = f'{tag}_pairwise'
        else:
            archive_name = f'{tag}_reference'

        archive_save_path = utils.create_archive_from_files(save_to,archive_name,saved_files,compress,'validations')

    if send:
        id_code = int(hashlib.sha256(save_to.encode('utf-8')).hexdigest(), 16) % 10**8
        email_notification = Notification(send,subject="Hi, there. I have your files for you.",message=f"Enclosed you will find your files.",id=id_code)
        if compress:
            email_notification.attach_file(f'{archive_save_path}.{compress}')
        else:
            email_notification.attach_mutiple_files(saved_files)
        email_notification.send()
        print(f"\n{email_notification.foot}\nYour e-mail identification code: {id_code}")

import src.analysis.significance_testing as sig


@click.command()
@click.option("--tag", cls=CustomClickOptions.StringAsOption, default=[],help="Experiment tag to be tested.")
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Comma-separated list of task IDs or symbols to be tested.")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[],help="Benchmark name or id to be tested.")
@click.option("--solver",
              "-s",
              required=True,
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
@click.option("--parametric","-p", type=click.Choice(['ANOVA', 't-test']),help="Parametric significance test. ANOVA for mutiple solvers and t-test for two solvers.")
@click.option("--non_parametric",
              "-np",
              type=click.Choice(['kruskal', 'mann-whitney-u']),help="Non-parametric significance test. kruksal for mutiple solvers and mann-whitney-u for two solvers.")
@click.option("--post_hoc_parametric",
              "-php",
              type=click.Choice(
                  ['scheffe', 'tamhane', 'ttest', 'tukey', 'tukey_hsd']),multiple=True,default=())
@click.option("--post_hoc_non_parametric",
              "-phn",
              type=click.Choice([
                  'conover', 'dscf', 'mannwhitney', 'nemenyi', 'dunn',
                  'npm_test', 'vanwaerden', 'wilcoxon'
              ]),multiple=True,default=())
@click.option("--p_adjust", type=click.Choice(
                  ['holm']),default='holm')
@click.option("--alpha", "-a", default=0.05, help="Significance level.")
@click.option('--export',
              '-e',
              type=click.Choice(
                  ['json', 'latex','csv']),multiple=True)
@click.option('--plot',
              '-p',
              type=click.Choice(
                  ['heatmap']),
              multiple=True,help="Create a heatmap for pairwise comparision results and a count plot for validation with references.")
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")
@click.option("--last", "-l",is_flag=True,help="Test the last finished experiment.")
def significance(tag, task, benchmark, solver, filter, combine, parametric,
                 non_parametric, post_hoc_parametric, post_hoc_non_parametric,
                 alpha, export, save_to, p_adjust,last):
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

    if not save_to:
        save_to = os.getcwd()
    if last:
        tag.append(utils.get_from_last_experiment("Tag"))

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    og_df = DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        benchmark,
                                        tag,
                                        filter,
                                        only_solved=True)
    clean = sig.prepare_data(og_df)

    grouping = ['tag', 'task_id', 'benchmark_id']

    #result_dict = {'parametric': None, 'non_parametric': None}

    if combine:
        grouping = [x for x in grouping if x not in combine]

    significance_tests = [non_parametric,parametric]
    significance_tests = [ x for x in significance_tests if x is not None]

    if significance_tests:
        sig_results_list = [ clean.groupby(grouping,as_index=False).apply(lambda df_: sig.test(kind_test,df_,equal_sample_size=True)) for kind_test in significance_tests]
        sig_results_df = pd.concat(sig_results_list)
        sig.print_result('significance',sig_results_df)


    post_hocs = list(post_hoc_parametric)
    post_hocs.extend(list(post_hoc_non_parametric))
    if post_hocs:
        ph_results_list = [ (clean.groupby(grouping,as_index=False).apply(lambda df_: sig.test_post_hoc(df_,ph,True,p_adjust))) for ph in post_hocs]
        ph_res_df = pd.concat(ph_results_list)
        ph_res_df.apply(lambda row: sig.print_post_hoc_results(row),axis=1)

    if export:
        pass




@click.command()
@click.option("--id", type=click.types.INT, required=False,help='ID of solver to delete.')
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
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    try:
        if all:
            click.confirm(
                "Are you sure you want to delete all solvers in the database?",
                abort=True,default=True)
            session.query(Solver).delete()
            session.commit()
            print("All solvers deleted.")
        else:
            click.confirm(
                "Are you sure you want to delete this solvers in the database?",
                abort=True)
            DatabaseHandler.delete_solver(session, id)
            session.commit()
            print("Solver deleted.")

    except ValueError as value_error:
        session.rollback()
        print(value_error)
    finally:
        session.close()

@click.command()
@click.option("--id", type=click.types.INT, required=False,help='ID of benchmark to delete.')
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
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    try:
        if all:
            click.confirm(
                "Are you sure you want to delete all benchmarks in the database?",
                abort=True,default=True)
            session.query(Benchmark).delete()
            session.commit()
            print("All benchmarks deleted.")
        else:
            click.confirm(
                "Are you sure you want to delete this benchmark in the database?",
                abort=True,default=True)
            DatabaseHandler.delete_benchmark(session, id)
            session.commit()
            print("Benchmark deleted.")

    except ValueError as value_error:
        session.rollback()
        print(value_error)
    finally:
        session.close()

@click.command()
@click.option("--symbol","-s",multiple=True,required=True)
def add_task(symbol):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    click.confirm(
        "Are you sure you want to add this task to the database?",
        abort=True, default=True)

    try:
        for new_symbol in list(symbol):
            new_task_id = DatabaseHandler.add_task(
            session, new_symbol)
            session.commit()
            print(f"Task {new_symbol} added to database with ID: {new_task_id}.")
    except ValueError as e:
        session.rollback()
        print(e)
    finally:
        session.close()

@click.command()
@click.option("--table","-t",multiple=True, type=click.Choice(['Task']),required=True)
def update_db(table):

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    tasks_in_db = set(DatabaseHandler.get_supported_tasks(session))
    tasks_in_definition = set(definitions.SUPPORTED_TASKS)
    diff = set.difference(tasks_in_definition,tasks_in_db)

    try:
        for new_symbol in list(diff):
            new_task_id = DatabaseHandler.add_task(
            session, new_symbol)
            session.commit()
            print(f"Task {new_symbol} added to database with ID: {new_task_id}.")
    except ValueError as e:
        session.rollback()
        print(e)
    finally:
        session.close()


@click.command()
def tasks():
   engine = DatabaseHandler.get_engine()
   session = DatabaseHandler.create_session(engine)
   tasks_in_db = sorted((DatabaseHandler.get_supported_tasks(session)))
   print(tasks_in_db)

@click.command()
@click.option('--verbose','-v',is_flag=True, help='Show summary of whole experiment.')
def last(verbose):
    """Shows basic information about the last finished experiment.

        Infos include experiment tag, benchmark names, solved tasks, executed solvers and and the time when the experiment was finished

    """
    if os.path.isfile(definitions.LAST_EXPERIMENT_JSON_PATH):
        with open(definitions.LAST_EXPERIMENT_JSON_PATH,'r') as file:
            json_string = file.read()
        json_obj = json.loads(json_string)
        print("Last experiment configuration:\n")
        for key,value in json_obj.items():
            print(f'{key}: {str(value)}')
        if verbose:
            if os.path.isfile(definitions.LAST_EXPERIMENT_SUMMARY_JSON_PATH):
                with open(definitions.LAST_EXPERIMENT_SUMMARY_JSON_PATH,'r') as file:
                    summary = file.read()

                print(f'\n{summary}')
    else:
        print("No experiments finished yet.")
@click.command()
@click.option("--tag","-t",required=True, help="Experiment tag.")
def experiment_info(tag):
    """Prints some basic information about the experiment speficied with "--tag" option.

    Args:
        tag ([type]): [description]
    """
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    try:
        df = stats.prepare_data(DatabaseHandler.get_results(session,[],[],[],[tag],[]))
    except ValueError as e:
        print(e)
    finally:
        session.close()
    if not df.empty:
        print(stats.get_experiment_info(df))
    else:
        print(f"No experiment with tag {tag} found.")

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
    with open(str(definitions.LOG_FILE_PATH),"r") as log_file:
        print(log_file.read())

@click.command()
@click.option('--name','-n', type=click.Choice(['ICCMA15','ICCMA19','ICCMA21']),multiple=True,required=True, help='Name of benchmark to fetch')
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark in. Default is the current working directory.")
@click.pass_context
def fetch(ctx,name, save_to):
    """Download ICCMA competition benchmakrs and add them to the database.

    Args:
        ctx ([type]): [description]
        name ([type]): [description]
        save_to ([type]): [description]
    """

    with open(definitions.FETCH_BENCHMARK_DEFAULTS_JSON,'r') as file:
        json_string = file.read()
    fetch_options = json.loads(json_string)

    if not save_to:
        save_to = os.getcwd()
    for benchmark_name in name:
        current_benchmark_options = fetch_options[benchmark_name]
        path_benchmark = fetching._fetch_benchmark(benchmark_name,save_to,current_benchmark_options)
        if path_benchmark:
            ctx.invoke(add_benchmark,
                   name=benchmark_name,
                   path=path_benchmark,
                   format=tuple(current_benchmark_options['format']),
                   random_arguments=current_benchmark_options['random_arguments'],
                   extension_arg_files=current_benchmark_options['extension_arg_files'],
                   generate=current_benchmark_options['generate']
                   )
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
@click.option("--timeout", "-t",type=click.types.FLOAT, help='Timeout for instances in seconds. Instances that exceed this value are discarded')
@click.option("--timemin","-tm",type=click.types.FLOAT, help='Minimum solving time of instance.')
@click.option("--format","-f", type=click.Choice(['apx','tgf']),help='Format of instances to generate')
@click.option("--task","-t",help='Tasks to generate solutions for.')
@click.option("--random_size","-rs",nargs=2,type=click.INT, help='Random number of arguments per instances. Usage: -rs [min] [max]')
@click.option("--add",is_flag=True,help="Add benchmark to database.")
@click.option("--solver")
@click.option('--task')
def stable_generator(ctx,num, name, save_to, num_args, generate_solutions, generate_arg_files,solver, timeout, timemin, format, task, random_size, add):
    """ Python port of the original StableGenerator of probo by Matthias Thimm.
        This is the generator for abstract argumentation graphs used
        in ICCMA'15 for the second group of problems (testcases 4-6). It generates
        graphs that (likely) possess many stable, preferred, and complete extensions.
        It works roughly as follows:
        1.) A set of arguments is identified to form an acyclic subgraph, containing
         the likely grounded extension
        2.) a subset of arguments is randomly selected and attacks are randomly added
          from some arguments within this set to all arguments outside the set (except
          to the arguments identified in 1.)
        3.) Step 2 is repeated until a number of desired stable extensions is reached *
        For more details see the code.

        This generator can be used by adapting the variables in the configuration block in the
        beginning of the class. Once started, this generator generates graphs continuously
        (it will not terminate on its own, the application has to be terminated forcefully).

         Matthias Thimm
    """
    if not save_to:
        save_to = os.getcwd()
        ctx.params['save_to'] = save_to
    if len(random_size) < 1:
        ctx.params['random_size'] = None
    default_config =  gen_utils.read_default_config(str(definitions.GENERATOR_DEFAULTS_JSON))
    gen_utils.update_configs(ctx, default_config,generator_type='stable')
    if default_config['general']['timemin'] and default_config['general']['timeout']:
        if (solver and task):

            engine = DatabaseHandler.get_engine()
            session = DatabaseHandler.create_session(engine)
            ref_solver =  DatabaseHandler.get_solver(session, solver)
            ref_task = DatabaseHandler.get_task(session, task)
            if ref_task not in ref_solver.supported_tasks:
                print("Solver does not support the specified task")
            default_config['general']['save'] = False
            accepted_graphs = []
            step_size = 100
            i = 0
            while len(accepted_graphs) < default_config["general"]["num"]:
                i += 1
                num = default_config["general"]["num"] - len(accepted_graphs)
                graphs = generator.generate('stable', default_config)
                accepted_graphs.extend(gen_utils._ensure_runtime(graphs, default_config, ref_solver, ref_task))
                print(f'{accepted_graphs=}')
                num_extension = default_config['stable']['num_extension']
                size_extension = default_config['stable']['size_extension']
                size_grounded = default_config['stable']['size_grounded']
                default_config['stable']['num_extension'] = [num_extension[0] + step_size, num_extension[1] + (2*step_size)]
                default_config['stable']['size_extension'] = [size_extension[0] + step_size, size_extension[1] + (2*step_size)]
                default_config['stable']['size_grounded'] = [size_grounded[0] + step_size, size_grounded[1] + (2*step_size)]
                print(default_config['stable']['num_extension'])
                print(default_config['stable']['size_extension'])
                print(default_config['stable']['size_grounded'])
                print(f'{step_size=}')
                if i % 5 == 0:
                    step_size += int(round(step_size * 0.5))
            gen_utils.save_instances(accepted_graphs, default_config['general'],'stable')



        else:
            print("To ensure instance runtime you have to specify a solver and a task")
            exit()

    else:
        save_directory = generator.generate('stable',default_config)
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
@click.option("--format","-f", type=click.Choice(['apx','tgf']),help='Format of instances to generate')
@click.option("--task","-t",help='Tasks to generate solutions for.')
@click.option("--random_size","-rs",nargs=2,type=click.INT,default=None, help='Random number of arguments per instances. Usage: -rs [min] [max]')
@click.option("--add",is_flag=True,help="Add benchmark to database.")
def scc_generator(ctx,num, name, save_to, num_args, generate_solutions, generate_arg_files,solver, timeout, timemin, format, task, random_size, add):

    if not save_to:
        save_to = os.getcwd()
        ctx.params['save_to'] = save_to
    if len(random_size) < 1:
        ctx.params['random_size'] = None
    default_config =  gen_utils.read_default_config(str(definitions.GENERATOR_DEFAULTS_JSON))
    gen_utils.update_configs(ctx, default_config,generator_type='scc')
    if default_config['general']['timemin'] and default_config['general']['timeout']:
        if (solver and task):

            engine = DatabaseHandler.get_engine()
            session = DatabaseHandler.create_session(engine)
            ref_solver =  DatabaseHandler.get_solver(session, solver)
            ref_task = DatabaseHandler.get_task(session, task)
            if ref_task not in ref_solver.supported_tasks:
                symbols = [ t.symbol for t in ref_solver.supported_tasks]
                print(f"Solver {ref_solver.solver_full_name} does not support the specified task {ref_task.symbol}.")
                print(f'Please choose one of the following tasks:\n{symbols}')
                exit()

            default_config['general']['save'] = False
            num_to_generate = default_config["general"]["num"]
            generated = 0
            save_directory = ''
            threshold = default_config['general']['threshold']
            while generated < num_to_generate:
                num = num_to_generate - generated

                sizes_batchs = gen_utils.get_batch_size(num,default_config['general']['batch_size'])
                print(sizes_batchs)
                sizes_batchs = list(filter(lambda num: num != 0, sizes_batchs))
                for size_batch in sizes_batchs:
                    default_config["general"]["num"] = size_batch
                    graphs = generator.generate('scc', default_config)
                    curr_accepted,too_easy,too_hard = gen_utils._ensure_runtime(graphs, default_config, ref_solver, ref_task)
                    print(f'{too_easy=} {too_hard=}')
                    if len(curr_accepted) > 0:

                        save_directory = gen_utils.save_instances(curr_accepted, default_config['general'],'scc',generated)
                        generated += len(curr_accepted)
                    if len(curr_accepted) < size_batch * threshold: # less than 20 % of graphs in batch are accepted
                        gen_utils.tweak_configs('scc',default_config,too_easy, too_hard)
                        break

        else:
            print("To ensure instance runtime you have to specify a solver and a task")
            exit()
    else:

        save_directory = generator.generate('scc',default_config)
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
@click.option("--format","-f", type=click.Choice(['apx','tgf']),help='Format of instances to generate')
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
@click.option("--tag", cls=CustomClickOptions.StringAsOption, default=[],help="Experiment tag to be tested.")
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Comma-separated list of task IDs or symbols to be tested.")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[],help="Benchmark name or id to be tested.")
@click.option("--solver",
              "-s",
              required=True,
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
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    help="Directory to store benchmark in. Default is the current working directory.")
@click.option("--last", "-l",is_flag=True,help="Calculate stats for the last finished experiment.")
@click.option("--file_name",'-fn')
def label(tag, task, benchmark, solver, filter, combine, last, save_to, file_name):
    if last:
        tag.append(utils.get_from_last_experiment("Tag"))


    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_full_name','instance']
    if combine:
        grouping = [x for x in grouping if x not in combine]

    df = stats.prepare_data(DatabaseHandler.get_results(session, solver, task, benchmark, tag,
                                    filter))


    mean_df = df.groupby(grouping,as_index=False).apply(lambda _df: _df.runtime.mean())
    mean_df.columns = mean_df.columns.fillna('runtime')
    best_df = mean_df.groupby('instance',as_index=False).apply(lambda _df: _df.loc[_df.runtime.idxmin()][['solver_full_name','instance','runtime']])
    best_df['label'] = best_df.solver_full_name.astype('category').cat.codes

    solvers = session.query(Solver).all()

    mapping_path = {}
    mapping_format = {}
    for s in solvers:
        mapping_path[s.solver_full_name] = s.solver_path
        mapping_format[s.solver_full_name] = s.solver_format


    best_df['solver_path'] = best_df.solver_full_name.map(mapping_path)
    best_df['solver_format'] = best_df.solver_full_name.map(mapping_format)
    if not file_name:
        file_name = 'labels.csv'
    else:
        file_name = f'{file_name}.csv'

    best_df.to_csv(os.path.join(save_to,file_name))




@click.command()
def test():
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    ref_solver =  DatabaseHandler.get_solver(session, 1)
    ref_task = DatabaseHandler.get_task(session, 'CE-PR')
    start = time.perf_counter()
    with open('/home/jklein/new_dev/benchmarks/scc_gen_test/scc_1.tgf','r') as f:
        instance = f.read()
        print(instance)
    end = time.perf_counter()
    print(end-start)
    gen_utils.solve(ref_solver.solver_path, ref_task, instance, 600)

cli.add_command(test)
cli.add_command(label)
cli.add_command(grounded_generator)
cli.add_command(scc_generator)
cli.add_command(stable_generator)
cli.add_command(fetch)
cli.add_command(logs)
cli.add_command(dumphelp_markdown)
cli.add_command(experiment_info)
cli.add_command(last)
cli.add_command(delete_benchmark)
cli.add_command(update_db)
cli.add_command(add_solver)
cli.add_command(add_benchmark)
cli.add_command(benchmarks)
cli.add_command(solvers)
cli.add_command(results)
cli.add_command(tasks)
cli.add_command(run)
cli.add_command(calculate)
cli.add_command(plot)
cli.add_command(export)
cli.add_command(status)
cli.add_command(validate)
cli.add_command(significance)
cli.add_command(delete_solver)
cli.add_command(add_task)
