import click
import os
import sys
import tabulate
import pandas

import src.Stats as Stats


from sqlalchemy import engine
from sqlalchemy import and_, or_

import src.DatabaseHandler as DatabaseHandler
import src.CustomClickOptions as CustomClickOptions
from src.Models import Result, Solver, Benchmark, Task
import src.definitions as Definitions
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
    new_solver = Solver(name=name, path=path,author=author, version=version, competition=competition,format=format)
    if guess:
        new_solver.format = new_solver.guess("formats")[0]  # select first supported format
        problems_output = new_solver.guess("problems")
        
        for p in problems_output:
            p = p.strip(" ")
            if p in Definitions.SUPPORTED_TASKS:
                tasks.append(p)
            
        
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
                 'competition': competition}
    new_benchmark = Benchmark(name=name, path=path,
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
    
    click.confirm("Are you sure you want to add this benchmark to the db?", abort=True)
    try:
        engine = DatabaseHandler.get_engine()
        session = DatabaseHandler.create_session(engine)
        new_benchmark_id = DatabaseHandler.add_benchmark(session, new_benchmark)
        session.commit()
        print("Benchmark {0} added to database with ID: {1}.".format(new_benchmark.name, new_benchmark_id))
    except ValueError as e:
        session.rollback()
        print(e)
    finally:
        session.close()

@click.command()
@click.option("--all", "-a",
              required=False, is_flag=True, help="Run all solvers in database on specified problems and instances.")
@click.option("--select", "-slct", is_flag=True, help="Run selected solvers from database.")
@click.option("--solver","-s", required=False, default=[], cls=CustomClickOptions.StringToInteger,
              help="Comma-separated list of solver ids in database.")
@click.option("--benchmark", "-b", required=False, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of benchmark ids or names in database.")
@click.option("--problem", "-p", required=False, callback=CustomClickOptions.check_problems,
              help="Computational problems")
@click.option("--save_to", required=False, help="Path for storing results.")
@click.option("--timeout","-t", required=False, default=600, help="Instance timeout in seconds")
@click.option("--dry", is_flag=True, help="Print the results to the commandline without saving.")
#@click.option("--track", cls=CustomClickOptions.TrackToProblemClass, is_eager=True,help="Solve the EE,SE,DC,DS problems for a semantics. Supported: CO,ST,GR,PR" )
@click.option("--tag", required=True, help="Specify tag for individual experiments.This tag is used to identify the experiment.")
@click.option("--notify", "-n",help="Send a notification to the email address provided as soon as the experiments are finished.")
@click.option("--report", is_flag=True)
@click.pass_context
def run(ctx, all, select, benchmark,
        problem, save_to, solver, timeout,
        dry, tag, notify, report):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = DatabaseHandler.get_benchmark(session,benchmark)
    tasks = DatabaseHandler.get_task(session,problem)
    
    if DatabaseHandler.tag_in_database(session,tag):
        print("Tag is already used. Please use another tag.")
        sys.exit()


    if all:
        for task in tasks:
            for bench in benchmarks:
                for solver in task.solvers:
                    res = solver.run(task,bench,timeout)
                    DatabaseHandler.insert_results(session,task,bench,solver,timeout,tag,res)
    if select:
        queried_solver = DatabaseHandler.get_solver(session,solver)
        for task in tasks:
            for bench in benchmarks:
                solvers_to_run = set(task.solvers).intersection(set(queried_solver))
                for solver in solvers_to_run:
                    res = solver.run(task,bench,timeout)
                    DatabaseHandler.insert_results(session,task,bench,solver,timeout,tag,res)

@click.command()
@click.option("--par", "-p", type=click.types.INT, help="Penalty multiplier for PAR score")
@click.option("--coverage", "-cov", is_flag=True, help="Calculate instance coverage")
@click.option("--average", "-avg", is_flag=True, help="Calculate average runtimes")
@click.option("--total", is_flag=True)
@click.option("--solver", "-s", required=True, cls=CustomClickOptions.StringAsOption, default=[],
              help="Comma-separated list of solver ids")
#@click.option("--filter", "-f", cls=CustomClickOptions.FilterAsDictionary, multiple=True,
              #help="Filter results in database. Format: [column:value]")
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
def calculate(par, coverage, average, total, solver, task, benchmark, print_format, tag):
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    res = session.query(Result).filter(Result.tag == tag)
    if solver:
        res = res.join(Solver).filter(or_(Solver.name.in_(solver), Solver.id.in_(solver)))
    if benchmark:
        res = res.join(Benchmark).filter(or_(Benchmark.name.in_(benchmark), Benchmark.id.in_(benchmark)))
    if task:
        res = res.join(Task).filter(or_(Task.symbol.in_(task), Task.id.in_(task)))
    df = pandas.read_sql(res.statement, res.session.bind)
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
    merged = merged[merged.columns[::-1]]
    

    grouped = merged.groupby(['task_id','benchmark_id'])
    for key, item in grouped:
        print(tabulate.tabulate( grouped.get_group(key),headers='keys', tablefmt='psql', showindex=False))
    
    #print(total_runtime)
    #print(Stats.get_count_timed_out(df))
    #print(Stats.calculate_par_score(df,10))

    # grouped_df = df.groupby(['task_id','benchmark_id','solver_id'])
    # print(grouped_df.apply(lambda g: g[g['timed_out'] == False])['runtime'].sum())
   


    
    # if solver:
    #     res = res.filter(or_(Result.solver.name.in_(solver),Result.solver.id.in_(solver)))


   
@click.command()
def benchmarks():
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    benchmarks = session.query(Benchmark).all()
    print(type(benchmarks))
    tabulate_data = []
    for benchmark in benchmarks:
        print(type(benchmark))
        tabulate_data.append([benchmark.id, benchmark.name, benchmark.format_instances])
    print(tabulate.tabulate(tabulate_data, headers=["ID", "Name", "Format"], tablefmt=format))
    session.close()

@click.command()
def results():
    engine = DatabaseHandler.get_engine()
    session = DatabaseHandler.create_session(engine)
    results = session.query(Result).all()
    tabulate_data = []
    for result in results:
        tabulate_data.append([result.instance, result.task.symbol,result.runtime,result.solver.fullname,result.tag])
    print(tabulate.tabulate(tabulate_data, headers=["INSTANCE", "TASK", "RUNTIME", "SOLVER", "TAG"], tablefmt=format))
    session.close()








cli.add_command(add_solver)
cli.add_command(add_benchmark)
cli.add_command(benchmarks)
cli.add_command(results)
cli.add_command(run)
cli.add_command(calculate)

