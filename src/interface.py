import itertools
import click
import json
import os
import sys
from click.core import iter_params_for_processing
from click.decorators import command
import pandas as pd
import shutil
import tabulate
import hashlib
import pathlib
import datetime
from sqlalchemy import and_, or_
from sqlalchemy import engine
from sqlalchemy.sql.expression import false
from jinja2 import Environment, FileSystemLoader
from src.utils import utils
from tabulate import tabulate
from src.utils import definitions
import timeit
import logging
from itertools import chain
from glob import glob

from random import choice


from src.reporting.validation_report import Validation_Report

import src.analysis.statistics as stats
import src.analysis.validation as validation
import src.database_models.DatabaseHandler as DatabaseHandler
import src.plotting.plotting_utils as pl_util
import src.utils.CustomClickOptions as CustomClickOptions
import src.utils.Status as Status
import src.utils.definitions as definitions
from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Benchmark import Benchmark
from src.database_models.Result import Result
from src.database_models.Solver import Solver
from src.utils.Notification import Notification


logging.basicConfig(filename=definitions.LOG_FILE_PATH,format='[%(asctime)s] - [%(levelname)s] : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)
#TODO: Dont save files when save_to not speficied and send is specified, Ausgabe für command benchmarks und solvers überarbeiten, Logging system,


@click.group()
def cli():

    if not os.path.exists(definitions.DATABASE_DIR):
        os.makedirs(definitions.DATABASE_DIR)
        logging.info("Database directory created.")

    if not os.path.exists(definitions.TEST_DATABASE_PATH):
        engine = DatabaseHandler.get_engine()
        DatabaseHandler.init_database(engine)



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
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    new_solver = Solver(solver_name=name,
                        solver_path=path,
                        solver_version=version,
                        solver_format=format)
    if guess:
        try:
            tasks = new_solver.fetch_tasks(tasks)
            format = new_solver.fetch_format(format)
        except ValueError as e:
            print(e)
            exit()
    new_solver.solver_format = format
    is_working = new_solver.check_interface(choice(tasks))
    if not is_working:
        exit()

    try:
        new_solver_id = DatabaseHandler.add_solver(session, new_solver, tasks)
        new_solver.print_summary()
        click.confirm(
            "Are you sure you want to add this solver to the database?"
            ,abort=True)
        session.commit()

        print("Solver {0} added to database with ID: {1}".format(
            name, new_solver_id))
        logging.info(f"Solver {name} added to database with ID: {new_solver_id}")
    except ValueError as e:
        session.rollback()
        print(e)
        logging.exception(f'Unable to add solver {name} to database')
    finally:
        session.close()


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
@click.option("--graph_type",
              "-gt",
              type=click.types.STRING,
              help="Graph type of instances")
@click.option("--format",
              "-f",
              required=True,
              multiple=True,
              type=click.Choice(["apx","tgf"]),
              help="Supported formats of benchmark/fileset")
@click.option("--hardness",
              "-h",
              type=click.types.STRING,
              help="Hardness of benchmark/fileset")
@click.option("--competition",
              "-c",
              type=click.types.STRING,
              help="Competition benchmark was used in")
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
def add_benchmark(name, path, graph_type, format, hardness, competition,
                  extension_arg_files, no_check, generate,
                  random_arguments):
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
    meta_data = {
        'graph_type': graph_type,
        'hardness': hardness,
        'benchmark_competition': competition
    }

    if len(format) > 1:
        format = ",".join(list(format))
    else:
        format = format[0]



    path_resolved = os.fspath(pathlib.Path(path).resolve())


    new_benchmark = Benchmark(benchmark_name=name,
                              benchmark_path=path_resolved,
                              format_instances=format,
                              extension_arg_files=extension_arg_files,
                              **meta_data)

    if generate:
        new_benchmark.generate_instances(generate)
    if random_arguments:
        new_benchmark.generate_argument_files(extension=extension_arg_files)
    if not no_check:
        new_benchmark.check()

    click.confirm(
        "Are you sure you want to add this benchmark to the database?",
        abort=True)

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    try:

        new_benchmark_id = DatabaseHandler.add_benchmark(
            session, new_benchmark)
        session.commit()
        print(f"Benchmark {new_benchmark.benchmark_name} added to database with ID: {new_benchmark_id}.")
        logging.info(f"Benchmark {new_benchmark.benchmark_name} added to database with ID: {new_benchmark_id}.")
    except ValueError as e:
        session.rollback()
        print(e)
        logging.exception(f'Unable to add benchmark {new_benchmark.benchmark_name} to database')
    finally:
        session.close()


@click.command()
@click.option(
    "--all",
    "-a",
    required=False,
    is_flag=True,
    help="Execute all solvers supporting the specified tasks on specified instances.")
@click.option("--select",
              "-slct",
              is_flag=True,
              help="Execute (via solver option) selected solver supporting the specified tasks.")
@click.option("--solver",
              "-s",
              required=False,
              default=[],
              cls=CustomClickOptions.StringAsOption,
              help=" Comma-seperated list of ids or names of solvers (in database) to run.")
@click.option(
    "--benchmark",
    "-b",
    required=False,
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
    required=True,
    help=
    "Tag for individual experiments.This tag is used to identify the experiment."
)
@click.option(
    "--notify",
    help=
    "Send a notification to the email address provided as soon as the experiments are finished."
)
#@click.option("--report", is_flag=True,help="Create summary report of experiment.")
@click.option("--n_times","-n",required=False,type=click.types.INT,default=1, help="Number of repetitions per instance. Run time is the avg of the n runs.")
@click.pass_context
def run(ctx, all, select, benchmark, task, solver, timeout, dry, tag,
        notify, track, n_times):
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
    run_parameter = ctx.params.copy()
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = DatabaseHandler.get_benchmarks(session, benchmark)
    if track:
        tasks = DatabaseHandler.get_tasks(session, track)
    else:

        tasks = DatabaseHandler.get_tasks(session, task)

    if DatabaseHandler.tag_in_database(session, tag) and not(dry):
        print("Tag is already used. Please use another tag.")
        sys.exit()

    Status.init_status_file(tasks, benchmarks, tag)
    run_parameter['benchmark'] = benchmarks
    run_parameter['task'] = tasks
    run_parameter['session'] = session

    if select:
        run_parameter['solver'] = DatabaseHandler.get_solvers(session, solver)


    logging.info(f"Stared to run Experiment {tag}")
    utils.run_experiment(run_parameter)
    logging.info(f"Finished to run Experiment {tag}")
    df = stats.prepare_data(
        DatabaseHandler.get_results(session, [], [], [], [tag],
                                    None))

    if not dry:
        if not df.empty:
            summary = stats.get_experiment_summary_as_string(df)
            print("")
            print(summary)
            now = datetime.datetime.now()
            last_experiment_dict = {'Tag': tag,
                                    'Benchmarks' : list(df.benchmark_name.unique()),
                                    'Tasks' : list(df.task.unique()),
                                    'Solvers': list(df.solver_full_name.unique()),
                                    'Finished': now.strftime("%m/%d/%Y, %H:%M:%S")
                                   }
            with open(definitions.LAST_EXPERIMENT_JSON_PATH,'w') as f:
                json.dump(last_experiment_dict,f)
        else:
            summary = 'Something went wrong. Please check the logs for more information.'

    if notify:
        id_code = int(hashlib.sha256(tag.encode('utf-8')).hexdigest(), 16) % 10**8
        notification = Notification(notify,message=f"Here a little summary of your experiment:\n{summary}",id=id_code)

        notification.send()
        logging.info(f'Sended Notfication e-mail to {notify}\nID: {id_code}')
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
@click.option("--statistics",'-s',type=click.Choice(['mean','sum','min','max','median','var','std','coverage','timeouts','solved','errors','all']),multiple=True)
@click.option("--last", "-l",is_flag=True,help="Calculate stats for the last finished experiment.")
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
@click.option("--send", required=False, help="Send plots via E-Mail.")
def calculate(par, solver, task, benchmark,
              tag,combine, vbs, export, save_to, statistics,print_format,filter,last, send, compress):

    if last:
        tag.append(utils.get_from_last_experiment("Tag"))


    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_id']
    export_columns = ['tag','solver','task','benchmark']

    if combine:
        grouping = [x for x in grouping if x not in combine]

    if 'all' in statistics:
        functions_to_call = ['mean','sum','min','max','median','var','std','coverage','timeouts','solved','errors']
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

    stats_df = (df
                .groupby(grouping,as_index=False)
                .apply(lambda df: stats.dispatch_function(df,functions_to_call,par_penalty=par))
                )

    print_headers = ['solver','task','benchmark']
    print_headers.extend(functions_to_call)
    utils.print_df(stats_df,['tag','benchmark','task'],headers=print_headers,format=print_format)

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
@click.option("--kind",'-k',type=click.Choice(['cactus','count','dist','pie','box','all']),multiple=True)
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files.")
@click.option("--send", "-s", required=False, help="Send plots via E-Mail.")
@click.option("--last", "-l",is_flag=True,help="Plot results for the last finished experiment.")
def plot(ctx, tag, task, benchmark, solver, save_to, filter, vbs, backend,combine, kind, compress, send, last):
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
    if not save_to:
        save_to = os.getcwd()
    if last:
        tag.append(utils.get_from_last_experiment("Tag"))

    default_options = pl_util.read_default_options(definitions.PLOT_JSON_DEFAULTS)
    pl_util.set_user_options(ctx, default_options)

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

    saved_files_paths = [f'{x}.{backend}' for x in saved_files_df.saved_files.to_list()]



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
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = session.query(Benchmark).all()
    tabulate_data = []
    if not verbose:
        for benchmark in benchmarks:
            tabulate_data.append([
                benchmark.id, benchmark.benchmark_name, benchmark.format_instances
            ])
        print(tabulate(tabulate_data, headers=["ID", "Name", "Format"]))

    else:
        for benchmark in benchmarks:
            b_formats = benchmark.get_formats()
            b_formats.append(benchmark.extension_arg_files)
            num_instances = []
            for f in b_formats:
                num_instances.append(len(benchmark.get_instances(f)))
            tabulate_data.append([
                benchmark.id, benchmark.benchmark_name, benchmark.format_instances, *num_instances, benchmark.benchmark_path
            ])

        str_b_formats = [f'#{x}' for x in b_formats]
        print(tabulate(tabulate_data,headers=["ID", "Name", "Format",*str_b_formats,'Path']))
    session.close()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
@click.option("--id",help="Print summary of solver with specified id")
def solvers(verbose,id):
    """Prints solvers in database to console.

    Args:
        verbose ([type]): [description]
    """
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    if id:
        solver = DatabaseHandler.get_solver(session,id)
        solver.print_summary()
    else:
        solvers = session.query(Solver).all()
        tabulate_data = []
        if verbose:
            for solver in solvers:
                tasks = [t.symbol for t in solver.supported_tasks]
                tabulate_data.append(
                    [solver.solver_id, solver.solver_name,solver.solver_version, solver.solver_format, tasks])

            print(
                tabulate(tabulate_data, headers=["ID", "Name","Version","Format", "Tasks"]))
        else:
            for solver in solvers:
                tabulate_data.append(
                    [solver.solver_id, solver.solver_name,solver.solver_format,solver.solver_path])

            print(
                tabulate(tabulate_data, headers=["ID", "Name", "Format","Path"]))

    session.close()


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
@click.option("--save_to", "-st", required=True)
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
def export(save_to, solver, filter, problem, benchmark, tag, task, format,
           group_by, file_name, include_column, exclude_column, only_solved,all_columns):
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
    if os.path.exists(definitions.STATUS_FILE_DIR):
        Status.print_status_summary()
    else:
        print("No status query is possible.")


@click.command()
@click.option("--tag", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--benchmark","-b", required=True,help="Benchmark name or id to validate.")
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
@click.option("--reference",
              "-r",
              type=click.Path(resolve_path=True,exists=True),
              help="Path to reference files.")
@click.option("--update_db", is_flag=True)
@click.option("--pairwise", "-pw", is_flag=True)
@click.option('--export',
              '-e',
              type=click.Choice(
                  ['json', 'latex', 'excel', 'html', 'heatmap','pie-chart','count-plot','csv']),
              multiple=True)
@click.option(
    "--save_to",
    "-st",
    type=click.Path(resolve_path=True,exists=True),
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")

@click.option('--extension','-ext',multiple=True, help="Reference file extension")
def validate(tag, task, benchmark, solver, filter, reference, pairwise,
             save_to, export, update_db,extension):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    og_df = DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        [benchmark],
                                        tag,
                                        filter,
                                        only_solved=True)
    result_df = validation.prepare_data(og_df)
    if pairwise:
        #validation.validate_pairwise(result_df, save_to=save_to, export=export)
        starttime = timeit.default_timer()
        validation._validate_pairwise(result_df)
        print( timeit.default_timer() - starttime)
    else:
        unique_tasks = list(result_df.task.unique())

        # Replace SE with EE for reference as SE problems are not unique
        for i,t in enumerate(unique_tasks):
            if 'SE' in t:
                unique_tasks[i] = unique_tasks[i].replace('SE','EE')

        ref_dict = validation.get_reference(extension, unique_tasks, reference)
        validation_results = validation.validate(result_df, ref_dict, extension)
        analyse = validation_results.groupby(['tag', 'task_id', 'benchmark_id','solver_id'],as_index=False).apply(lambda df: validation.analyse(df) )
        print(analyse)
        validation.test_table_export(analyse,'/mnt/c/Users/jonas/Entwicklung')
        validation.print_validate_with_reference_results(analyse)

        #print(validation_results.groupby(['tag', 'task_id', 'benchmark_id','solver_id'],as_index=False).apply(lambda df: df[df.no_reference == True][['solver_full_name','task','instance']] ))




        # num_correct = val[val.correct == 'correct']['correct'].count()

        # num_incorrect = val[val.correct == 'incorrect']['correct'].count()
        # num_no_reference = val[val.correct == 'no_reference']['correct'].count()
        # total = val['instance'].count()
        # percentage_validated = (total - num_no_reference) / total * 100

        # print("**********Summary**********")
        # print(f'Total instances: {total}')
        # print(f'Correct instances: {num_correct}')
        # print(f'Incorrect instances: {num_incorrect}')
        # print(f'No reference: {num_no_reference}')
        # print(f'Percentage valdated: {percentage_validated}')
        # print("")


        # print(analyse)

        #val.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.print_summary(df) )
        # pdf = Validation_Report()
        # pdf.print_summary(tag)
        # pdf.output(os.path.join(save_to,'SalesRepot.pdf'), 'F')


        # if 'count-plot' in export:
        #     validation.count_plot(val,save_to,title="Summary",grid=False)
        #     val.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.count_plot(df,save_to))


        #     #validation.count_plot(val,save_to,title="Summary-Grid",grid=True)
        # if "pie-chart" in export:
        #     validation.pie_chart(analyse,save_to,title="Summary")
        #     analyse.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.pie_chart(df,save_to))

        #validation.export_styled_table(analyse[['solver','task','benchmark_name','correct_solved','incorrect_solved','no_reference','total','percentage_validated']],save_to)



        # og_df.set_index("id", inplace=True)

        # og_df.correct.update(val.set_index("id").correct)

        # og_df.validated.update(val.set_index("id").validated)

        # if update_db:
        #     result_objs = session.query(Result).filter(
        #         Result.tag.in_(tag)).all()
        #     id_result_mapping = dict()
        #     for res in result_objs:
        #         id_result_mapping[res.id] = res

        #     og_df.apply(lambda row: validation.update_result_object(
        #         id_result_mapping[row.name], row['correct'], row['validated']),
        #                 axis=1)
        #     session.commit()



import src.analysis.significance_testing as sig


@click.command()
@click.option("--tag", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--task",
              "-t",
              required=False,
              callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[])
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
@click.option("--parametric","-p", type=click.Choice(['ANOVA', 't-test']))
@click.option("--non_parametric",
              "-np",
              type=click.Choice(['kruskal', 'mann-whitney-u']))
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
                  ['json', 'latex', 'excel', 'html', 'heatmap', 'csv']),
              multiple=True)
@click.option(
    "--save_to",
    "-st",
    type=click.Path(exists=True, resolve_path=True),
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")
@click.option("--equal_sample_size",is_flag=True)
@click.option("--last", "-l",is_flag=True,help="Test the last finished experiment.")
def significance(tag, task, benchmark, solver, filter, combine, parametric,
                 non_parametric, post_hoc_parametric, post_hoc_non_parametric,
                 alpha, export, save_to, p_adjust, equal_sample_size, last):

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
        sig_results_list = [ clean.groupby(grouping,as_index=False).apply(lambda df_: sig.test(kind_test,df_,equal_sample_size)) for kind_test in significance_tests]
        sig_results_df = pd.concat(sig_results_list)
        sig_results_df.apply(lambda row: sig.print_significance_results(alpha,row),axis=1)


    post_hocs = list(post_hoc_parametric)
    post_hocs.extend(list(post_hoc_non_parametric))
    if post_hocs:
        ph_results_list = [ (clean.groupby(grouping,as_index=False).apply(lambda df_: sig.test_post_hoc(df_,ph,False,p_adjust))) for ph in post_hocs]
        ph_res_df = pd.concat(ph_results_list)
        ph_res_df.apply(lambda row: sig.print_post_hoc_results(row),axis=1)



    #TODO: Ausgabe für post-hoc und significance test implementieren, Exporting refactoring


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
                abort=True)
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
                abort=True)
            session.query(Benchmark).delete()
            session.commit()
            print("All benchmarks deleted.")
        else:
            click.confirm(
                "Are you sure you want to delete this benchmark in the database?",
                abort=True)
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
        abort=True)

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
def last():
    if os.path.isfile(definitions.LAST_EXPERIMENT_JSON_PATH):
        with open(definitions.LAST_EXPERIMENT_JSON_PATH,'r') as file:
            json_string = file.read()
        json_obj = json.loads(json_string)
        print("Last experiment:")
        for key,value in json_obj.items():
            print(f'{key}: {str(value)}')
    else:
        print("No experiments finished yet.")
@click.command()
@click.option("--tag","-t")
def experiment_info(tag):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    try:
        df = stats.prepare_data(DatabaseHandler.get_results(session,[],[],[],[tag],[]))
    except ValueError as e:
        print(e)
    finally:
        session.close()

    print(stats.get_experiment_info(df))

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