from src.database_models.Base import Supported_Tasks
import click
import os
import sys
import json
import seaborn
import sqlalchemy
from sqlalchemy.sql.expression import false
import tabulate
import pandas as pd

import src.plotting.ScatterPlot as ScatterPlot

import src.plotting.CactusPlot as CactusPlot

import src.plotting.DistributionPlot as DistributionPlot

import src.Stats as Stats


from sqlalchemy import engine
from sqlalchemy import and_, or_

import src.DatabaseHandler as DatabaseHandler
import src.CustomClickOptions as CustomClickOptions
#from src.Models import Result, Solver, Benchmark, Task
from src.database_models.Solver import Solver
from src.database_models.Result import Result
from src.database_models.Task import Task
from src.database_models.Benchmark import Benchmark
from src.Notification import Notification
import src.Status as Status

import src.definitions as Definitions
# TEST COMMIT
@click.group()
def cli():
    if not os.path.exists(Definitions.DATABASE_DIR):
        os.makedirs(Definitions.DATABASE_DIR)
        engine = DatabaseHandler.get_engine()
        DatabaseHandler.init_database(engine)

@click.command(cls=CustomClickOptions.command_required_option_from_option('guess'))
@click.option("--name", "-n", required=True, help="Name of the solver")
@click.option("--path", required=True, callback=CustomClickOptions.check_path, type=str,
              help="Full path to solver executable")
@click.option("--format", "-f", type=click.Choice(['apx', 'tgf'], case_sensitive=False), required=False,
              help="Supported format")
@click.option('--tasks', "-t", required=False, default=[], callback=CustomClickOptions.check_problems,
              help="Supported computational problems")
@click.option("--version", "-v", type=click.types.STRING, required=True, help="Version of solver")
@click.option("--author", "-a", type=click.types.STRING, required=False, default='', help="Author of solver")
@click.option("--competition", "-c", type=click.types.STRING, required=False, default="",
              help="ICCMA competition the solver paticipated in.")
@click.option("--guess", is_flag=True, help="Pull supported file format and computational problems from solver")
def add_solver(name, path, format, tasks, version, author, competition, guess):
    """ Adds a solver to the database.
    Adding has to be confirmed by user.
      Args:
          guess: Pull supported file format and computational problems from solver.
          name: Name of the Solver as string.
          path: Full path to the executable of the solver as string.
          format: Supported file format of the solver as string.
          problems: Supported problems of the solver as a list of strings.
          version: Version of solver.
          author: Author of solver.
          competition: ICCMA competition this solver was first committed to.
     Returns:
         None
     Raises:
          None
      """
    new_solver = Solver(solver_name=name, solver_path=path,solver_author=author, solver_version=version, solver_competition=competition,solver_format=format)
    if guess:
        if format:
            new_solver.solver_format = format
        else:
            new_solver.solver_format = new_solver.guess("formats")[0]  # select first supported format
        
        if not tasks:
            problems_output = new_solver.guess("problems")
        
            for p in problems_output:
                p = p.strip(" ")
                if p in Definitions.SUPPORTED_TASKS:
                    tasks.append(p)
    
    # TODO: Check if solver is working
    #new_solver.check_solver(tasks)       
        
    try:
        engine = DatabaseHandler.get_engine()
        session = DatabaseHandler.create_session(engine)
        
        click.confirm("Are you sure you want to add this solver to the database?", abort=True)
        new_solver_id = DatabaseHandler.add_solver(session, new_solver, tasks)
        session.commit()
        new_solver.print_summary()
        print("Solver {0} added to database with ID: {1}".format(name, new_solver_id))
    except ValueError as e:
        session.rollback()
        print(e)
    finally:
        session.close()

@click.command()
@click.option("--name", "-n", type=click.types.STRING, required=True, help="Name of benchmark/fileset")
@click.option("--path", "-p", type=click.types.STRING, required=True, help="Path to instances",
              callback=CustomClickOptions.check_path)
@click.option("--graph_type", "-gt", type=click.types.STRING, help="Graph type of instances")
@click.option("--format", "-f", required=True, help="Supported formats of benchmark/fileset")
@click.option("--hardness", "-h", type=click.types.STRING, help="Hardness of benchmark/fileset")
@click.option("--competition", "-c", type=click.types.STRING, help="Competition benchmark was used in")
@click.option("--extension_arg_files", "-ext", type=click.types.STRING, default='arg',
              help="Extension of additional argument parameter for DC/DS problems")
@click.option("--no_check", is_flag=True, help="Checks if the benchmark is complete.")
@click.option("--no_args", is_flag=True,
              help="Flag to indicate that there are no additional argument files for the DC/DS problems.")
@click.option("--generate", "-g", type=click.types.Choice(['apx', 'tgf']),
              help="Generate instances in specified format")
@click.option("--random_arguments","-rnd", is_flag=True, help="Generate additional argument files with a random argument.")
def add_benchmark(name, path, graph_type,
                  format, hardness, competition,
                  extension_arg_files, no_check, no_args, generate, random_arguments):
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
    meta_data = {'graph_type': graph_type, 'hardness': hardness,
                 'benchmark_competition': competition}
    new_benchmark = Benchmark(benchmark_name=name, benchmark_path=path,
                                  format_instances=format,
                                  extension_arg_files=extension_arg_files, **meta_data)
    
    if generate:
        new_benchmark.generate_files(generate)
    if random_arguments:
        new_benchmark.generate_argument_files(extension_arg_files)


    if not no_check:
        if not new_benchmark.is_complete():
            if click.confirm("Some files are missing. Do you want to create the missing files?"):
                new_benchmark.generate_missing_files()
                if not no_args:
                    new_benchmark.generate_argument_files(extension_arg_files)
                if not new_benchmark.is_complete():
                    sys.exit("Something went wrong when generating the missing instances.")
    
    click.confirm("Are you sure you want to add this benchmark to the database?", abort=True)
    try:
        engine = DatabaseHandler.get_engine()
        session = DatabaseHandler.create_session(engine)
        new_benchmark_id = DatabaseHandler.add_benchmark(session, new_benchmark)
        session.commit()
        print("Benchmark {0} added to database with ID: {1}.".format(new_benchmark.benchmark_name, new_benchmark_id))
    except ValueError as e:
        session.rollback()
        print(e)
    finally:
        session.close()

@click.command()
@click.option("--all", "-a",
              required=False, is_flag=True, help="Run all solvers in database on specified problems and instances.")
@click.option("--select", "-slct", is_flag=True, help="Run selected solvers from database.")
@click.option("--solver","-s", required=False, default=[], cls=CustomClickOptions.StringAsOption,
              help="Comma-separated list of solver ids in database.")
@click.option("--benchmark", "-b", required=False, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of benchmark ids or names in database.")
@click.option("--problem", "-p", required=False, callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--save_to", required=False, help="Path for storing results.")
@click.option("--timeout","-t", required=False, default=600, help="Instance timeout in seconds")
@click.option("--dry", is_flag=True, help="Print the results to the commandline without saving.")
@click.option("--track", cls=CustomClickOptions.TrackToProblemClass,default="",type=click.types.STRING, is_eager=True,help="Solve the EE,SE,DC,DS problems for a semantics. Supported: CO,ST,GR,PR" )
@click.option("--tag", required=True, help="Specify tag for individual experiments.This tag is used to identify the experiment.")
@click.option("--notify", "-n",help="Send a notification to the email address provided as soon as the experiments are finished.")
@click.option("--report", is_flag=True)
@click.pass_context
def run(ctx, all, select, benchmark,
        problem, save_to, solver, timeout,
        dry, tag, notify, report,track):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = DatabaseHandler.get_benchmarks(session,benchmark)
    print(track)
    if track:
        print(track)
        tasks = DatabaseHandler.get_tasks(session,track)
    else:
        tasks = DatabaseHandler.get_tasks(session,problem)
    sys.exit()
    if DatabaseHandler.tag_in_database(session,tag):
        print("Tag is already used. Please use another tag.")
        sys.exit()

    Status.init_status_file(tasks,benchmarks,tag)
    
    if all:
        for task in tasks:
            print("**********{}***********".format(task.symbol.upper()))
            for bench in benchmarks:
                print("Benchmark: ",bench.benchmark_name)
                for solver in task.solvers:
                    click.echo(solver.fullname,nl=False)
                    solver.run(task,bench,timeout,save_db=True, tag=tag, session=session)
                    print("---FINISHED")
            Status.increment_task_counter()
    if select:
        queried_solver = DatabaseHandler.get_solvers(session,solver)
        for task in tasks:
            print("**********{}***********".format(task.symbol.upper()))
            for bench in benchmarks:
                print("Benchmark: ",bench.benchmark_name)
                solvers_to_run = set(task.solvers).intersection(set(queried_solver))
                for solver in solvers_to_run:
                    click.echo(solver.fullname,nl=False)
                    solver.run(task,bench,timeout,save_db=True, tag=tag, session=session)
                    print("---FINISHED")
    if notify:
        notification = Notification(notify)
        notification.send()

@click.command()
@click.option("--par", "-p", type=click.types.INT, help="Penalty multiplier for PAR score")
@click.option("--coverage", "-cov", is_flag=True, help="Calculate instance coverage")
@click.option("--average", "-avg", is_flag=True, help="Calculate average runtimes")
@click.option("--total", is_flag=True)
@click.option("--solver", "-s", required=True, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of solver ids")
@click.option("--filter", "-f", cls=CustomClickOptions.FilterAsDictionary, multiple=True)
@click.option("--task", "-t", required=False, callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--benchmark", "-b", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--print_format", "-pf", type=click.Choice(["plain",
                                                          "simple",
                                                          "github",
                                                          "grid",
                                                          "fancy_grid",
                                                          "pipe",
                                                          "orgtbl",
                                                          "jira",
                                                          "presto",
                                                          "pretty",
                                                          "psql",
                                                          "rst",
                                                          "mediawiki",
                                                          "moinmoin",
                                                          "youtrack",
                                                          "html",
                                                          "unsafehtml"
                                                          "latex",
                                                          "latex_raw",
                                                          "latex_booktabs",
                                                          "textile"]))
@click.option("--tag")
#@click.option("--vbs", is_flag=True, help="Create virtual best solver")
#@click.option("--save_to", "-st", help="Directory to store tables")
#@click.option("--export",type=click.Choice(["html","latex","png","jpeg","svg"]),default="png")
#@click.option("--css",default="styled-table.css",help="CSS file for table style.")
def calculate(par, coverage, average, total, solver, task, benchmark, print_format, tag,filter):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    df = DatabaseHandler.get_results(session,solver,task,benchmark,tag,filter)
    res = []
    if par:
        res.append(Stats.calculate_par_scores(df,par))
    if average:
        res.append(Stats.calculate_average_runtime(df))
    if total:
        res.append(Stats.calculate_total_runtime(df))
    if coverage:
        res.append(Stats.calculate_coverage(df))
    
    merged = Stats.merge_dataframes(res)
    merged['Solver'] = merged.index.to_series().map(lambda index: DatabaseHandler.get_full_name_solver(session,index[2]))
    merged = merged[merged.columns[::-1]] # change column order
    

    grouped = merged.groupby(['task_id','benchmark_id'])
    for key, item in grouped:
        task_symbol = DatabaseHandler.get_task(session,int(key[0])).symbol
        benchmark_name = DatabaseHandler.get_benchmark(session,int(key[1])).benchmark_name
        
        print("Task:",task_symbol)
        print("Benchmark:", benchmark_name)
        print(tabulate.tabulate( grouped.get_group(key),headers='keys', tablefmt='psql', showindex=False))
        print("")

@click.command()
@click.pass_context
@click.option("--tag", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--problem", "-p", required=False, callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--benchmark", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--solver", "-s", required=True, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of solver ids")
@click.option("--filter", "-f", cls=CustomClickOptions.FilterAsDictionary, multiple=True,
              help="Filter results in database. Format: [column:value]")
@click.option("--save_to", "-st", required=True,
              help="Directory to store plots in. Filenames will be generated automatically.")
@click.option("--vbs", is_flag=True, help="Create virtual best solver")
@click.option("--plot_type", "-pt", type=click.Choice(['cactus', 'scatter', 'distribution']), help="Specifiy plot type")
@click.option("--x_max", "-xm", type=click.types.INT)
@click.option("--y_max", "-ym", type=click.types.INT)
@click.option("--alpha", "-a", type=click.FloatRange(0, 1), help="Alpha value (only for scatter plots)")
@click.option("--backend", "-b", type=click.Choice(['pdf', 'pgf', 'png', 'ps', 'svg']), default='png', help="Backend to use")
@click.option("--no_grid", "-ng", is_flag=True, help="Do not show a grid.")
def plot(ctx, tag, problem, benchmark, solver, save_to, filter, vbs, plot_type, x_max, y_max, alpha, backend, no_grid):
    with open(Definitions.PLOT_JSON_DEFAULTS, 'r') as fp:
        options = json.load(fp)['settings']
        options['def_path'] = Definitions.PLOT_JSON_DEFAULTS

    for key, value in ctx.params.items():
        if value is not None:
            options[key] = value

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)

    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    task = problem
    df = DatabaseHandler.get_results(session,solver,task,benchmark,tag,filter,only_solved=True)
    group = df.groupby(['tag','task_id','benchmark_id'])
    for key, item in group:
        cur = group.get_group(key)
        cur['Solver'] = cur['solver_name']+"_" + cur['solver_version']
        prepared = Stats.prepare_data_seaborn_plot(cur)

        task_symbol = DatabaseHandler.get_task(session,int(key[1])).symbol
        benchmark_name = DatabaseHandler.get_benchmark(session,int(key[2])).benchmark_name
        curr_tag = key[0]
        
        options['title'] = task_symbol + " " + benchmark_name
        save_file_name = os.path.join(save_to, "{}_{}_{}_".format(task_symbol, benchmark_name,curr_tag))
        options['save_to'] = save_file_name

        if plot_type =='cactus':
                plotter = CactusPlot.Cactus(options)
                plotter.create(prepared)
        
@click.command()
def benchmarks():
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = session.query(Benchmark).all()
    tabulate_data = []
    for benchmark in benchmarks:
        tabulate_data.append([benchmark.id, benchmark.benchmark_name, benchmark.format_instances])

    print(tabulate.tabulate(tabulate_data, headers=["ID", "Name", "Format"], tablefmt=format))
    session.close()

@click.command()
@click.option("--verbose","-v",is_flag=True, default=False, required=False)
def solvers(verbose):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    solvers = session.query(Solver).all()
    tabulate_data = []
    if verbose:
        for solver in solvers:
            tasks = [t.symbol for t in solver.supported_tasks]
            tabulate_data.append([solver.id, solver.solver_name, solver.solver_format, tasks])

        print(tabulate.tabulate(tabulate_data, headers=["ID", "Name", "Format","Tasks"], tablefmt=format))
    else:
        for solver in solvers:
           tabulate_data.append([solver.id, solver.solver_name, solver.solver_format])

        print(tabulate.tabulate(tabulate_data, headers=["ID", "Name", "Format"], tablefmt=format))
    

    session.close()



@click.command()
@click.option("--verbose","-v",is_flag=True, default=False, required=False)
def results(verbose):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    results = session.query(Result).all()
    tabulate_data = []
    if verbose:
        for result in results:
            tabulate_data.append([result.instance, result.task.symbol,result.runtime,result.solver.fullname,result.tag,result.exit_with_error,result.error_code,result.benchmark.benchmark_name])
        print(tabulate.tabulate(tabulate_data, headers=["INSTANCE", "TASK", "RUNTIME", "SOLVER", "TAG", "EXIT_WITH_ERROR", "ERROR_CODE","BENCHMARK"], tablefmt=format))
    else:
        for result in results:
            tabulate_data.append([result.instance, result.task.symbol,result.runtime,result.solver.fullname,result.tag])
        print(tabulate.tabulate(tabulate_data, headers=["INSTANCE", "TASK", "RUNTIME", "SOLVER", "TAG"], tablefmt=format))
    session.close()


@click.command()
@click.option("--save_to", "-st", required=True)
@click.option("--solver", "-s", required=False, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of solver ids")
@click.option("--filter", "-f", cls=CustomClickOptions.FilterAsDictionary, multiple=True,
              help="Filter results in database. Format: [column:value]")
@click.option("--problem", "-p", required=False, callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--task", "-t", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--benchmark", "-b", cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--tag", "-t", required=False, cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--format", type=click.Choice(['csv','json','xlsx']),default='csv')
@click.option("--group_by", "-g", type=click.Choice(['tag','solver_name','benchmark_name','symbol']))
@click.option("--file_name",required=False, default='data')
@click.option("--include_column", required=False, cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--exclude_column", required=False, cls=CustomClickOptions.StringAsOption, default=[])
@click.option("--only_solved",is_flag=True, default=False)

def export(save_to, solver, filter, problem, benchmark, tag,task,format, group_by, file_name, include_column, exclude_column, only_solved):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    colums_to_export = ['solver_name','solver_version','benchmark_name','instance','timed_out','runtime','symbol','additional_argument','tag']
    
    result_df = DatabaseHandler.get_results(session,solver,task,benchmark,tag,filter,only_solved=only_solved)
    
    if include_column:
        colums_to_export.extend(include_column)
    if exclude_column:
        colums_to_export = [ele for ele in colums_to_export if ele not in exclude_column]

    export_df = result_df[colums_to_export]
  
    if group_by:
        grouped = export_df.groupby(group_by)
        for name, group in grouped:
            group.to_csv(os.path.join(save_to,'{}.zip'.format(name)), index=False)
    
    file = os.path.join(save_to,file_name)
    if format == 'xlsx':
        export_df.to_excel("{}.{}".format(file,'xlsx'),index=False)
    if format == 'csv':
        export_df.to_csv("{}.{}".format(file,'csv'),index=False)
    if format == 'json':
        export_df.to_json("{}.{}".format(file,'json'))
   
@click.command()           
def status():
    Status.print_status_summary()

  







cli.add_command(add_solver)
cli.add_command(add_benchmark)
cli.add_command(benchmarks)
cli.add_command(solvers)
cli.add_command(results)
cli.add_command(run)
#cli.add_command(calculate)
cli.add_command(plot)
cli.add_command(export)
cli.add_command(status)
