
import click
import json
import os
import sys
import shutil
import tabulate
import hashlib
import pathlib
from sqlalchemy import and_, or_
from sqlalchemy import engine
from sqlalchemy.sql.expression import false
from jinja2 import Environment, FileSystemLoader
from src.utils import utils
from tabulate import tabulate

import logging

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


#TODO: Dont save files when save_to not speficied and send is specified, Ausgabe für command benchmarks und solvers überarbeiten, Logging system,


@click.group()
def cli():

    if not os.path.exists(definitions.DATABASE_DIR):
        os.makedirs(definitions.DATABASE_DIR)

    if not os.path.exists(definitions.TEST_DATABASE_PATH):
        engine = DatabaseHandler.get_engine()
        DatabaseHandler.init_database(engine)


@click.command(cls=CustomClickOptions.command_required_option_from_option('guess'))
@click.option("--name", "-n", required=True, help="Name of the solver")
@click.option("--path",
              required=True,
              callback=CustomClickOptions.check_path,
              type=str,
              help="Full path to solver executable")
@click.option("--format",
              "-f",
              type=click.Choice(['apx', 'tgf'], case_sensitive=False),
              required=False,
              help="Supported format")
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
              help="Version of solver")
@click.option(
    "--guess",
    is_flag=True,
    help="Pull supported file format and computational problems from solver")
def add_solver(name, path, format, tasks, version, guess):
    """ Adds a solver to the database.
    Adding has to be confirmed by user.
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
    path_resolved = os.fspath(pathlib.Path(path).resolve())
    new_solver = Solver(solver_name=name,
                        solver_path=path_resolved,
                        solver_version=version,
                        solver_format=format)
    supported_task_database = DatabaseHandler.get_supported_tasks(session)
    if guess:
        if format:
            new_solver.solver_format = format
        else:
            new_solver.solver_format = new_solver.guess("formats")[
                0]  # select first supported format

        if not tasks:
            problems_output = new_solver.guess("problems")

            for p in problems_output:
                p = p.strip(" ")
                if p in supported_task_database:
                    tasks.append(p)

    # print("Testing solver...", end='')
    # if not new_solver.check_solver(tasks):
    #     print("failed.")
    #     click.confirm("Continue?", abort=True)
    # else:
    #     print("success.")



    try:

        new_solver_id = DatabaseHandler.add_solver(session, new_solver, tasks)
        new_solver.print_summary()
        click.confirm(
            "Are you sure you want to add this solver to the database?"
            ,abort=True)
        session.commit()

        print("Solver {0} added to database with ID: {1}".format(
            name, new_solver_id))
    except ValueError as e:
        session.rollback()
        print(e)
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
              type=click.types.STRING,
              required=True,
              help="Path to instances",
              callback=CustomClickOptions.check_path)
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
    help="Extension of additional argument parameter for DC/DS problems")
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
     Adding has to be confirmed by user.
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
        print("Benchmark {0} added to database with ID: {1}.".format(
            new_benchmark.benchmark_name, new_benchmark_id))
    except ValueError as e:
        session.rollback()
        print(e)
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
#@click.option("--save_to", required=False, help="Path for storing results.")
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
@click.option("-d",is_flag=True)
@click.pass_context
def run(ctx, all, select, benchmark, task, solver, timeout, dry, tag,
        notify, track, n_times,d):
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

    if d:
        logging.basicConfig(level=logging.DEBUG)
        debug_str = ""
        for b in benchmarks:
            debug_str += f"\nBenchmark: {b.benchmark_name}\nPath: {b.benchmark_path}"
        for t in tasks:
            debug_str += (f"\nTask: {t.symbol}\nID: {t.id}\nSupported Solvers:")
            for s in t.solvers:
                debug_str +=(f"\nSolver: {s.solver_name}\nPath: {s.solver_path}")

    logging.debug(debug_str)


    utils.run_experiment(run_parameter)
    df = stats.prepare_data(
        DatabaseHandler.get_results(session, [], [], [], [tag],
                                    None))
    if  df.empty:
         raise click.BadParameter("Something went wrong")


    if not dry:
        if df.empty:
            summary = stats.get_experiment_summary_as_string(df)
            print("")
            print(summary)

    if notify:
        id_code = int(hashlib.sha256(tag.encode('utf-8')).hexdigest(), 16) % 10**8
        note_message = "Note: Since the access data for this e-mail account are public, please do not open any attachments to e-mails in which the identification code does not match the one generated for you."
        notification = Notification(notify,message=f"Here a little summary of your experiment:\n{summary}.\nYour e-mail identification code: {id_code}\n\n{note_message}")

        notification.send()
        print(f"\n{note_message}\nYour e-mail identification code: {id_code}")


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
              "-pf",default='fancy_grid',
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
@click.option("--save_to", "-st", help="Directory to store tables")
@click.option("--export","-e",type=click.Choice(["html","latex","png","jpeg","svg",'csv']),default=None,multiple=True)
@click.option("--css",default="styled-table.css",help="CSS file for table style.")
@click.option("--statistics",'-s',type=click.Choice(['mean','sum','min','max','median','var','std','coverage','num_timed_out','all']),multiple=True)
def calculate(par, solver, task, benchmark,
              tag, filter, combine, vbs, css, export, save_to, statistics,print_format):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_id']
    export_columns = ['tag','solver','task','benchmark']

    if combine:
        grouping = [x for x in grouping if x not in combine]

    if 'all' in statistics:
        functions_to_call = ['mean','sum','min','max','median','var','std','coverage','num_timed_out']
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

    if export:
        export_columns.extend(functions_to_call)
        utils.export(stats_df,export,save_to=save_to,columns=export_columns,css_file=css)


@click.command()
@click.pass_context
@click.option("--tag", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--task",

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
@click.option(
    "--save_to",
    "-st",
    required=True,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")
@click.option("--vbs", is_flag=True, help="Create virtual best solver")
@click.option("--x_max", "-xm", type=click.types.INT)
@click.option("--y_max", "-ym", type=click.types.INT)
@click.option("--alpha",
              "-a",
              type=click.FloatRange(0, 1),
              help="Alpha value (only for scatter plots)")
@click.option("--backend",
              "-b",
              type=click.Choice(['pdf', 'pgf', 'png', 'ps', 'svg']),
              default='png',
              help="Backend to use")
@click.option("--no_grid", "-ng", is_flag=True, help="Do not show a grid.")
@click.option("--grid_plot",is_flag=True)
@click.option("--combine",
              "-c",
              cls=CustomClickOptions.StringAsOption,
              default=[])
@click.option("--kind",'-k',type=click.Choice(['cactus','count','dist','scatter','pie','box','all']),multiple=True)
@click.option("--compress",type=click.Choice(['tar','zip']), required=False,help="Compress saved files")
@click.option("--send", "-s", required=False, help="Send plots via E-Mail")
def plot(ctx, tag, task, benchmark, solver, save_to, filter, vbs,
         x_max, y_max, alpha, backend, no_grid,grid_plot, combine, kind, compress, send):

    ref = definitions.PLOT_JSON_DEFAULTS
    with ref.open('rb') as fp:
        options =json.load(fp)['settings']
        options['def_path'] = ref

    for key, value in ctx.params.items():
        if value is not None:
            options[key] = value

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
        vbs_id = -1
        vbs_df = df.groupby(grouping_vbs,as_index=False).apply(lambda df: stats.create_vbs(df,vbs_id))
        df = df.append(vbs_df)

    for plot_kind in list(kind):
        pl_util.create_plots(plot_kind,df,save_to,options, grouping)
    if compress:
        save_archive_to = save_to.rstrip("/")
        click.echo(f"Creating archive {save_archive_to}.{compress}...",nl=False)
        shutil.make_archive(save_archive_to, compress, save_archive_to)
        click.echo("finished")
    if send:
        id_code = int(hashlib.sha256(save_to.encode('utf-8')).hexdigest(), 16) % 10**8
        note_message = "Note: Since the access data for this e-mail account are public, please do not open any attachments to e-mails in which the identification code does not match the one generated for you."
        email_attachments = Notification(send,subject="Hi, there. I have your files for you.",message=f"Enclosed you will find your files.\nYour e-mail identification code: {id_code}\n\n{note_message}")
        if compress:
            email_attachments.attach_files(f'{save_archive_to}.{compress}')
        else:
            email_attachments.attach_files(save_to)
        email_attachments.send()
        print(f"\n{note_message}\nYour e-mail identification code: {id_code}")




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
    if verbose:
        for benchmark in benchmarks:
            tabulate_data.append([
                benchmark.id, benchmark.benchmark_name, benchmark.format_instances,benchmark.benchmark_path
            ])
        print(tabulate(tabulate_data,
                            headers=["ID", "Name", "Format", "Path"],
                            tablefmt=format))

    else:
        for benchmark in benchmarks:
            tabulate_data.append([
                benchmark.id, benchmark.benchmark_name, benchmark.format_instances
            ])

        print(
            tabulate(tabulate_data,
                            headers=["ID", "Name", "Format"],
                            tablefmt=format))
    session.close()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
def solvers(verbose):
    """Prints solvers in database to console.

    Args:
        verbose ([type]): [description]
    """
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    solvers = session.query(Solver).all()
    tabulate_data = []
    if verbose:
        for solver in solvers:
            tasks = [t.symbol for t in solver.supported_tasks]
            tabulate_data.append(
                [solver.solver_id, solver.solver_name,solver.solver_version, solver.solver_format, tasks])

        print(
            tabulate(tabulate_data,
                              headers=["ID", "Name","Version","Format", "Tasks"],
                              tablefmt=format))
    else:
        for solver in solvers:
            tabulate_data.append(
                [solver.solver_id, solver.solver_name,solver.solver_format,solver.solver_path])

        print(
            tabulate(tabulate_data,
                              headers=["ID", "Name", "Format","Path"],
                              tablefmt=format))

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
def results(verbose, solver, task, benchmark, tag, filter):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    result_df = DatabaseHandler.get_results(session,
                                            solver,
                                            task,
                                            benchmark,
                                            tag,
                                            filter,
                                            only_solved=False)
    print(result_df)


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
@click.option("--reference",
              "-r",
              cls=CustomClickOptions.StringAsOption,
              default=[],
              help="Comma separted list of path to reference files.")
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
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")

def validate(tag, task, benchmark, solver, filter, reference, pairwise,
             save_to, export, update_db):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    og_df = DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        benchmark,
                                        tag,
                                        filter,
                                        only_solved=True)
    result_df = validation.prepare_data(og_df)
    if pairwise:
        validation.validate_pairwise(result_df, save_to=save_to, export=export)
    else:

        unique_benchmark_ids = sorted(result_df['benchmark_id'].unique())

        references = dict(zip(unique_benchmark_ids, reference))

        val = validation.validate(result_df, references)






        num_correct = val[val.correct == 'correct']['correct'].count()

        num_incorrect = val[val.correct == 'incorrect']['correct'].count()
        num_no_reference = val[val.correct == 'no_reference']['correct'].count()
        total = val['instance'].count()
        percentage_validated = (total - num_no_reference) / total * 100

        print("**********Summary**********")
        print(f'Total instances: {total}')
        print(f'Correct instances: {num_correct}')
        print(f'Incorrect instances: {num_incorrect}')
        print(f'No reference: {num_no_reference}')
        print(f'Percentage valdated: {percentage_validated}')
        print("")

        analyse = val.groupby(['tag', 'task_id', 'benchmark_id','solver_id'],as_index=False).apply(lambda df: validation.analyse(df) )
        print(analyse)

        #val.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.print_summary(df) )
        # pdf = Validation_Report()
        # pdf.print_summary(tag)
        # pdf.output(os.path.join(save_to,'SalesRepot.pdf'), 'F')


        if 'count-plot' in export:
            validation.count_plot(val,save_to,title="Summary",grid=False)
            val.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.count_plot(df,save_to))


            #validation.count_plot(val,save_to,title="Summary-Grid",grid=True)
        if "pie-chart" in export:
            validation.pie_chart(analyse,save_to,title="Summary")
            analyse.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.pie_chart(df,save_to))

        #validation.export_styled_table(analyse[['solver','task','benchmark_name','correct_solved','incorrect_solved','no_reference','total','percentage_validated']],save_to)



        og_df.set_index("id", inplace=True)

        og_df.correct.update(val.set_index("id").correct)

        og_df.validated.update(val.set_index("id").validated)

        if update_db:
            result_objs = session.query(Result).filter(
                Result.tag.in_(tag)).all()
            id_result_mapping = dict()
            for res in result_objs:
                id_result_mapping[res.id] = res

            og_df.apply(lambda row: validation.update_result_object(
                id_result_mapping[row.name], row['correct'], row['validated']),
                        axis=1)
            session.commit()



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
@click.option("--parametric", type=click.Choice(['ANOVA', 't-test']))
@click.option("--non_parametric",
              "-np",
              type=click.Choice(['kruskal', 'mann-whitney-u']))
@click.option("--post_hoc_parametric",
              "-php",
              type=click.Choice(
                  ['scheffe', 'tamhane', 'ttest', 'tukey', 'tukey_hsd']))
@click.option("--post_hoc_non_parametric",
              "-phn",
              type=click.Choice([
                  'conover', 'dscf', 'mannwhitney', 'nemenyi', 'dunn',
                  'npm_test', 'vanwaerden', 'wilcoxon'
              ]))
@click.option("--alpha", "-a", default=0.05, help="Significance level.")
@click.option('--export',
              '-e',
              type=click.Choice(
                  ['json', 'latex', 'excel', 'html', 'heatmap', 'csv']),
              multiple=True)
@click.option(
    "--save_to",
    "-st",
    required=False,
    help=
    "Directory to store plots in. Filenames will be generated automatically.")
def significance(tag, task, benchmark, solver, filter, combine, parametric,
                 non_parametric, post_hoc_parametric, post_hoc_non_parametric,
                 alpha, export, save_to):
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

    result_dict = {'parametric': None, 'non_parametric': None}

    if combine:
        grouping = [x for x in grouping if x not in combine]

    if parametric:
        if parametric == 't-test':
            if len(clean['solver_id'].unique()) > 2:
                raise click.ClickException(
                    "Dataset contains more than two samples. Please use option ANOVA for a parametric test of two or more samples."
                )
            else:
                if grouping:
                    clean.groupby(grouping).apply(
                        lambda df: sig.student_t_test(df, alpha))
                else:
                    sig.student_t_test(clean, alpha)

        if parametric == 'ANOVA':
            if grouping:
                result_parametric = clean.groupby(grouping).apply(
                    lambda df: sig.anova_oneway_parametric(df,
                                                           alpha,
                                                           post_hoc=
                                                           post_hoc_parametric,
                                                           p_adjust='holm',
                                                           export=export,
                                                           save_to=save_to))
                result_parametric = result_parametric.reset_index(
                    name='parametric_result')

            else:
                sig.anova_oneway_parametric(clean,
                                            alpha,
                                            post_hoc=post_hoc_parametric,
                                            p_adjust='holm',
                                            export=export,
                                            save_to=save_to)

        result_dict['parametric'] = result_parametric[
            'parametric_result'].tolist()

    if non_parametric:
        if non_parametric == 'mann-whitney-u':
            if len(clean['solver_id'].unique()) > 2:
                raise click.ClickException(
                    "Dataset contains more than two samples. Please use option kruskal for a non-parametric test of two or more samples."
                )
            else:
                if grouping:
                    non_parametric_result = clean.groupby(grouping).apply(
                        lambda df: sig.mann_whitney_u(df, alpha))
                    non_parametric_result = non_parametric.reset_index(
                        name='non_parametric_result')
                else:
                    sig.mann_whitney_u(clean, alpha)
        if non_parametric == 'kruskal':
            if grouping:
                non_parametric_result = clean.groupby(grouping).apply(
                    lambda df: sig.kruskal_non_parametric(
                        df,
                        alpha,
                        post_hoc=post_hoc_non_parametric,
                        p_adjust='holm',
                        export=export,
                        save_to=save_to))

                non_parametric_result = non_parametric_result.reset_index(
                    name='non_parametric_result')

            else:
                sig.kruskal_non_parametric(clean,
                                           alpha,
                                           post_hoc=post_hoc_non_parametric,
                                           p_adjust='holm',
                                           export=export,
                                           save_to=save_to)
        result_dict['non_parametric'] = non_parametric_result[
            'non_parametric_result'].tolist()
    # sig_data = [x['post_hoc_result'] for x in result_dict["non_parametric"]]
    # test_data = []
    # data_list = []
    # for test in sig_data:
    #     for index, row in test.iterrows():
    #         for i, value in row.items():
    #             test_data.append({'x': index, 'y': i, 'value': value})
    #     data_list.append(json.dumps(test_data))
    # env = Environment(loader=FileSystemLoader(
    #     os.path.join(definitions.ROOT_DIR, "src", "html_templates",
    #                  "reports")))
    # template = env.get_template('heatmap_template.html')
    # html = template.render(data_list=data_list)
    # with open('html_report_jinja.html', 'w') as f:
    #     f.write(html)

    #sig.print_result_dict(result_dict)
    # env = Environment(loader=FileSystemLoader(os.path.join(definitions.ROOT_DIR,"src","html_templates","reports")))
    # template = env.get_template('significance_template.html')
    # html = template.render(result_dict=result_dict)
    # with open('html_report_jinja.html', 'w') as f:
    #     f.write(html)


@click.command()
@click.option("--id", type=click.types.INT, required=False)
@click.option("--all", is_flag=True)
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
@click.option("--id", type=click.types.INT, required=False)
@click.option("--all", is_flag=True)
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