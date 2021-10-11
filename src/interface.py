from typing_extensions import ParamSpec
import click
import json
import os
from matplotlib.pyplot import grid
import pandas as pd
import seaborn as sns
import sqlalchemy
import sys
from sqlalchemy.sql.functions import coalesce
import tabulate
import numpy as np
from click.decorators import group
from sqlalchemy import and_, or_
from sqlalchemy import engine
from sqlalchemy.sql.expression import false
from jinja2 import Environment, FileSystemLoader
from src.utils import utils

from src.reporting.validation_report import Validation_Report


import src.analysis.statistics as stats
import src.analysis.validation as validation
import src.database_models.DatabaseHandler as DatabaseHandler
import src.plotting.CactusPlot as CactusPlot
import src.plotting.DistributionPlot as DistributionPlot
import src.plotting.ScatterPlot as ScatterPlot
import src.plotting.plotting_utils as pl_util
import src.utils.CustomClickOptions as CustomClickOptions
import src.utils.Status as Status
import src.utils.definitions as definitions
from src.database_models.Base import Base, Supported_Tasks
from src.database_models.Benchmark import Benchmark
from src.database_models.Result import Result
from src.database_models.Solver import Solver
from src.database_models.Task import Task
from src.utils.Notification import Notification


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
          tasks:
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
    new_solver = Solver(solver_name=name,
                        solver_path=path,
                        solver_version=version,
                        solver_format=format)
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
                if p in definitions.SUPPORTED_TASKS:
                    tasks.append(p)

    # print("Testing solver...", end='')
    # if not new_solver.check_solver(tasks):
    #     print("failed.")
    #     click.confirm("Continue?", abort=True)
    # else:
    #     print("success.")

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    try:

        new_solver_id = DatabaseHandler.add_solver(session, new_solver, tasks)
        new_solver.print_summary()
        click.confirm(
            "Are you sure you want to add this solver to the database?",
            abort=True)
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
@click.option(
    "--no_args",
    is_flag=True,
    help=
    "Flag to indicate that there are no additional argument files for the DC/DS problems."
)
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
                  extension_arg_files, no_check, no_args, generate,
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
    new_benchmark = Benchmark(benchmark_name=name,
                              benchmark_path=path,
                              format_instances=format,
                              extension_arg_files=extension_arg_files,
                              **meta_data)

    if generate:
        new_benchmark.generate_files(generate)
    if random_arguments:
        new_benchmark.generate_argument_files(extension_arg_files)

    if not no_check:
        if not new_benchmark.is_complete():
            if click.confirm(
                    "Some files are missing. Do you want to create the missing files?"
            ):
                new_benchmark.generate_missing_files()
                if not no_args:
                    new_benchmark.generate_argument_files(extension_arg_files)
                if not new_benchmark.is_complete():
                    sys.exit(
                        "Something went wrong when generating the missing instances."
                    )

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
    help="Run all solvers in database on specified problems and instances.")
@click.option("--select",
              "-slct",
              is_flag=True,
              help="Run selected solvers from database.")
@click.option("--solver",
              "-s",
              required=False,
              default=[],
              cls=CustomClickOptions.StringAsOption,
              help="Comma-separated list of solver ids in database.")
@click.option(
    "--benchmark",
    "-b",
    required=False,
    cls=CustomClickOptions.StringAsOption,
    default=[],
    help="Comma-separated list of benchmark ids or names in database.")
@click.option("--task",
              cls=CustomClickOptions.StringAsOption,
              required=False,
              #callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--save_to", required=False, help="Path for storing results.")
@click.option("--timeout",
              "-t",
              required=False,
              default=600,
              help="Instance timeout in seconds")
@click.option("--dry",
              is_flag=True,
              help="Print the results to the commandline without saving.")
@click.option(
    "--track",
    cls=CustomClickOptions.TrackToProblemClass,
    default="",
    type=click.types.STRING,
    is_eager=True,
    help="Solve the EE,SE,DC,DS problems for a semantics. Supported: CO,ST,GR,PR"
)
@click.option(
    "--tag",
    required=True,
    help=
    "Specify tag for individual experiments.This tag is used to identify the experiment."
)
@click.option(
    "--notify",
    "-n",
    help=
    "Send a notification to the email address provided as soon as the experiments are finished."
)
@click.option("--report", is_flag=True)
@click.pass_context
def run(ctx,all, select, benchmark, task, save_to, solver, timeout, dry, tag,
        notify, report, track):
    """[summary]

    Args:
        all ([type]): [description]
        select ([type]): [description]
        benchmark ([type]): [description]
        task ([type]): [description]
        save_to ([type]): [description]
        solver ([type]): [description]
        timeout ([type]): [description]
        dry ([type]): [description]
        tag ([type]): [description]
        notify ([type]): [description]
        report ([type]): [description]
        track ([type]): [description]
    """
    run_parameter = ctx.params.copy()
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = DatabaseHandler.get_benchmarks(session, benchmark)
    if track:
        tasks = DatabaseHandler.get_tasks(session, track)
    else:

        tasks = DatabaseHandler.get_tasks(session, task)

    if DatabaseHandler.tag_in_database(session, tag):
        print("Tag is already used. Please use another tag.")
        sys.exit()

    Status.init_status_file(tasks, benchmarks, tag)
    run_parameter['benchmark'] = benchmarks
    run_parameter['task'] = tasks
    run_parameter['session'] = session

    if select:
        run_parameter['solver'] = DatabaseHandler.get_solvers(session, solver)

    utils.run_experiment(run_parameter)

    if notify:
        notification = Notification(notify)
        notification.send()


@click.command()
@click.option("--par",
              "-p",
              type=click.types.INT,
              help="Penalty multiplier for PAR score")
@click.option("--coverage",
              "-cov",
              is_flag=True,
              help="Calculate instance coverage")
@click.option("--average",
              "-avg",
              is_flag=True,
              help="Calculate average runtimes")
@click.option("--total", is_flag=True)
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
              "-pf",
              type=click.Choice([
                  "plain", "simple", "github", "grid", "fancy_grid", "pipe",
                  "orgtbl", "jira", "presto", "pretty", "psql", "rst",
                  "mediawiki", "moinmoin", "youtrack", "html", "unsafehtml"
                  "latex", "latex_raw", "latex_booktabs", "textile"
              ]))
@click.option("--tag", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--validated", is_flag=True)
@click.option("--iccma", is_flag=True)
@click.option("--combine",
              "-c",
              cls=CustomClickOptions.StringAsOption,
              default=[])
# @click.option("--vbs", is_flag=True, help="Create virtual best solver")
@click.option("--save_to", "-st", help="Directory to store tables")
# @click.option("--export",type=click.Choice(["html","latex","png","jpeg","svg"]),default="png")
# @click.option("--css",default="styled-table.css",help="CSS file for table style.")
def calculate(par, coverage, average, total, solver, task, benchmark,
              print_format, tag, filter, combine, validated,iccma, save_to):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    grouping = ['tag', 'task_id', 'benchmark_id', 'solver_id']

    if combine:
        grouping = [x for x in grouping if x not in combine]


    df = stats.prepare_data(
        DatabaseHandler.get_results(session, solver, task, benchmark, tag,
                                    filter))
    res = []
    if par:
        par_scores = (df
                        .groupby(grouping,as_index=False)
                        .apply(lambda group: stats.calculate_par_score(group, par))
                     )

        res.append(par_scores)


    if average:
        res.append(stats.calculate_average_runtimes(df))
    if total:
        total_runtimes = (df
                        .groupby(grouping,as_index=False)
                        .apply(lambda group: stats.calculate_total_runtime(group))
                        )
        res.append(total_runtimes)

    if coverage:
        coverages = (df
                        .groupby(grouping,as_index=False)
                        .apply(lambda group: stats.calculate_coverage(group))
                        )
        res.append(coverages)
    if iccma:
        iccma_scores = (df
                        .groupby(grouping,as_index=False)
                        .apply(lambda group: stats.calculate_iccma_score(group))
                        )
        res.append(iccma_scores)

        #res.append(stats.calculate_coverages(df))

    merged = stats.merge_dataframes(
        res, ['tag', 'task_id', 'benchmark_id', 'solver_id','task','solver','benchmark'])

    #merged = merged[merged.columns[::-1]]  # change column order

    grouping.remove('solver_id')
    merged.groupby(grouping).apply(lambda df: print(
        tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    ))
    #css_file = definitions.CSS_TEMPLATES_PATH +"/tables/styled-table.css"
    #merged[['solver','task','PAR10','total_runtime','coverage','iccma_score']].groupby(['task']).apply(lambda df: utils.export_html(df,save_to,css_file=css_file))


@click.command()
@click.pass_context
@click.option("--tag", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--problem",
              "-p",
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
@click.option("--plot_type",
              "-pt",
              type=click.Choice(['cactus', 'scatter', 'distribution']),
              help="Specifiy plot type")
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
def plot(ctx, tag, problem, benchmark, solver, save_to, filter, vbs, plot_type,
         x_max, y_max, alpha, backend, no_grid):
    with open(definitions.PLOT_JSON_DEFAULTS, 'r') as fp:
        options = json.load(fp)['settings']
        options['def_path'] = definitions.PLOT_JSON_DEFAULTS

    for key, value in ctx.params.items():
        if value is not None:
            options[key] = value

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    task = problem
    grid = True
    if plot_type == 'cactus':
        group = pl_util.prepare_data_cactus_plot(
            DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        benchmark,
                                        tag,
                                        filter,
                                        only_solved=True))
        if grid:
            grid_data = pl_util.prepare_grid(DatabaseHandler.get_results(session,
                                        solver,
                                        task,
                                        benchmark,
                                        tag,
                                        filter,
                                        only_solved=True))
            grid_data = grid_data.rename(columns={'rank': 'Instance','solver_full_name': 'Solver','task':'Task','runtime':'Runtime'})
            grid_plot = sns.relplot(x="Instance",y="Runtime", hue="Solver", col="Task",
                data=grid_data, kind="line",markers=True,
                height=4, aspect=.9)
            figure = grid_plot.fig
            figure.savefig(f"{save_to}_grid.png",
                     bbox_inches='tight',
                   transparent=True)
        else:
            pl_util.cactus_plot_group(group, save_to, options)


@click.command()
def benchmarks():
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = session.query(Benchmark).all()
    tabulate_data = []
    for benchmark in benchmarks:
        tabulate_data.append([
            benchmark.id, benchmark.benchmark_name, benchmark.format_instances
        ])

    print(
        tabulate.tabulate(tabulate_data,
                          headers=["ID", "Name", "Format"],
                          tablefmt=format))
    session.close()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False, required=False)
def solvers(verbose):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    solvers = session.query(Solver).all()
    tabulate_data = []
    if verbose:
        for solver in solvers:
            tasks = [t.symbol for t in solver.supported_tasks]
            tabulate_data.append(
                [solver.id, solver.solver_name, solver.solver_format, tasks])

        print(
            tabulate.tabulate(tabulate_data,
                              headers=["ID", "Name", "Format", "Tasks"],
                              tablefmt=format))
    else:
        for solver in solvers:
            tabulate_data.append(
                [solver.solver_id, solver.solver_name, solver.solver_format])

        print(
            tabulate.tabulate(tabulate_data,
                              headers=["ID", "Name", "Format"],
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
    print(result_df.tag.unique())

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
    Status.print_status_summary()


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
                  ['json', 'latex', 'excel', 'html', 'heatmap','pie-chart','count-plot' 'csv']),
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



        # validation.count_plot(val,save_to,title="Summary",grid=False)
        # validation.count_plot(val,save_to,title="Summary-Grid",grid=True)
        # validation.pie_chart(analyse,save_to,title="Summary")

        # analyse.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.pie_chart(df,save_to))
        # val.groupby(['tag', 'task_id', 'benchmark_id']).apply(lambda df: validation.count_plot(df,save_to))
        validation.export_styled_table(analyse[['solver','task','benchmark_name','correct_solved','incorrect_solved','no_reference','total','percentage_validated']],save_to)



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
    sig_data = [x['post_hoc_result'] for x in result_dict["non_parametric"]]
    test_data = []
    data_list = []
    for test in sig_data:
        for index, row in test.iterrows():
            for i, value in row.items():
                test_data.append({'x': index, 'y': i, 'value': value})
        data_list.append(json.dumps(test_data))
    env = Environment(loader=FileSystemLoader(
        os.path.join(definitions.ROOT_DIR, "src", "html_templates",
                     "reports")))
    template = env.get_template('heatmap_template.html')
    html = template.render(data_list=data_list)
    with open('html_report_jinja.html', 'w') as f:
        f.write(html)

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


cli.add_command(add_solver)
cli.add_command(add_benchmark)
cli.add_command(benchmarks)
cli.add_command(solvers)
cli.add_command(results)
cli.add_command(run)
cli.add_command(calculate)
cli.add_command(plot)
cli.add_command(export)
cli.add_command(status)
cli.add_command(validate)
cli.add_command(significance)
cli.add_command(delete_solver)
