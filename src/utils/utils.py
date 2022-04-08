import os
import psutil
from pathlib import Path
from tabulate import tabulate
import json
import shutil

import click
import pandas
from src.database_models import DatabaseHandler

from src.utils import definitions as definitions
from src.utils import Status

from subprocess import Popen
from subprocess import TimeoutExpired
from subprocess import CalledProcessError
from subprocess import CompletedProcess

from src.database_models.Result import Result

def create_archive_from_files(save_to: str, name: str,saved_files_paths: list,compress: str, archive_type:str):
    temp_path = os.path.join(save_to,'_temp_files')
    if not os.path.exists(temp_path):
        os.mkdir(os.path.join(save_to,'_temp_files'))
    for file_to_copy in saved_files_paths:
        shutil.copy(file_to_copy,temp_path)
    archive_save_to = os.path.join(save_to,f'{name}_{archive_type}')
    directory_save_path =compress_directory(temp_path, archive_save_to,compress)
    shutil.rmtree(temp_path)
    return directory_save_path

def print_df(df, grouping=None, headers=None, format='fancy_grid'):
    if headers is None:
        headers = df.columns.values
    if grouping is None:
        print(tabulate(df[headers],headers='keys',tablefmt=format,showindex=False))

    else:
        (df
        .groupby(grouping)
        [headers]
        .apply(lambda df: print(tabulate(df,headers='keys',tablefmt=format,showindex=False))))


def compress_directory(from_dir: str, to_dir: str, compression: str):
    from_dir = from_dir.rstrip(os.path.sep)
    to_dir = to_dir.rstrip(os.path.sep)
    shutil.make_archive(to_dir, compression,from_dir)
    return to_dir

def export(df: pandas.DataFrame,
               formats,
               save_to: str,
               file_name=None,
               columns=None,
               index=False,
               css_file=None
               ) -> None:
    """[summary]

    Args:
        df (pandas.DataFrame): [description]
        formats ([type]): [description]
        save_to (str): [description]
        columns ([type], optional): [description]. Defaults to None.
        index (bool, optional): [description]. Defaults to False.
    """



    if not columns:
        columns = list(df.columns)

    if 'html' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_html(df,save_to,css_file=css_file,index=index,file_name=file_name)))
    if 'csv' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_csv(df,save_to,index=index,file_name=file_name)))
    if 'latex' in formats:
        (df
        [columns]
        .groupby(['task'])
        .apply(lambda df: export_latex(df,save_to,index=index,file_name=file_name)))
def export_latex(df: pandas.DataFrame,
               save_to: str,
               columns=None,
               index=False,
               encoding='utf-8',
               file_name=None) -> None:
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)
    if columns:
        df = df[columns]

    save_path = os.path.join(save_to, f'{file_name}.tex')
    tasks = list(df.task.unique())
    benchs = list(df.benchmark.unique())

    task_string = ",".join(tasks)
    bench_string = ",".join(benchs)

    caption_text = f'Results for tasks {task_string} and benchmark {bench_string}'

    df = df.drop(columns=['tag','benchmark','task'])
    df.to_latex(save_path,
              index=index,
              caption=caption_text,
              encoding=encoding)





def export_csv(df: pandas.DataFrame,
               save_to: str,
               columns=None,
               index=False,
               encoding='utf-8',
               file_name=None,
               compression=None) -> None:
    """[summary]

    Args:
        df (pandas.DataFrame): [description]
        save_to (str): [description]
        columns ([type], optional): [description]. Defaults to None.
        index (bool, optional): [description]. Defaults to False.
        encoding (str, optional): [description]. Defaults to 'utf-8'.
        file_name ([type], optional): [description]. Defaults to None.
        compression ([type], optional): [description]. Defaults to None.

    Returns:
        bool: [description]
    """
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)
    if columns:
        df = df[columns]

    save_path = os.path.join(save_to, f'{file_name}.csv')

    df.to_csv(save_path,
              index=index,
              compression=compression,
              encoding=encoding)


def export_html(df: pandas.DataFrame,
                save_to: str,
                columns=None,
                index=False,
                encoding='utf-8',
                file_name="html_data.html",
                css_file=None):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        save_to (str): [description]
    """
    Path(save_to).mkdir(parents=True, exist_ok=True)
    if not file_name:
        file_name = create_file_name(df)

    if columns:
        df = df[columns]
    save_path = os.path.join(save_to, f'{file_name}.html')

    if css_file:
        css_file_path = os.path.join(str(definitions.CSS_TEMPLATES_PATH),"tables",css_file)
        name_css_file = Path(css_file_path).stem
        table_html_string = df.to_html(escape=False,
                                       index=index,
                                       classes=name_css_file)

        with open(css_file_path, encoding="utf-8") as file:
            content_css_file = file.read()
        content_css_file = "<style>\n" + content_css_file + "\n</style>"

        with open(save_path, "w", encoding=encoding) as file_to_write:
            file_to_write.write(content_css_file + table_html_string)
    else:
        df.to_html(save_path, index=index, encoding=encoding)

import multiprocessing as mp

def _multiprocess_run_experiment(parameters: dict):
    """[summary]

    Args:
        parameters (dict): [description]
    """
    tag = parameters['tag']
    tasks = parameters['task']
    benchmarks = parameters['benchmark']
    timeout = parameters['timeout']
    dry = parameters['dry']
    session = parameters['session']
    select = parameters['select']
    n = parameters['n_times']
    first_n_instances = parameters['subset']
    runs = []
    for task in tasks:
        task_symbol = task.symbol.upper()
        click.echo(f"**********{task_symbol}***********")
        for bench in benchmarks:

            click.echo(f"Benchmark: {bench.benchmark_name}")
            if select:
                solvers_to_run = set(task.solvers).intersection(
                    set(parameters['solver']))
            else:

                solvers_to_run = task.solvers

            for solver in solvers_to_run:
                #click.echo(solver.solver_full_name, nl=False)
                runs.append((solver,task,bench,timeout,(not dry),tag,n,first_n_instances,True))
    if runs:
        # TODO splitting runs in num_cores - 1 batches
        pool = mp.Pool((psutil.cpu_count(logical=False) - 1) or 1) #  to avoid locking up the system
        res_paths_1 = pool.starmap(_single_run,runs)
        _bulk_instert_results(res_paths_1,session)
        session.commit()
        try:
            shutil.rmtree('_temp_results')
        except:
            print('Error while deleting _temp_results directory')




def _create_result_obj_from_json(file_path,session):

    with open(file_path,'r') as result_json:
        data = json.load(result_json)
    benchmark = DatabaseHandler.get_benchmark(session,data['benchmark_id'])
    solver = DatabaseHandler.get_solver(session,data['solver_id'])
    task = DatabaseHandler.get_task(session,data['task_id'])

    result_obj = Result(tag=data['tag'],
                    solver_id=data['solver_id'],
                    benchmark_id = data['benchmark_id'],
                    task_id = data['task_id'],
                    instance=data['instance'],
                    cut_off=data['timeout'],
                    timed_out = data['timed_out'],
                    runtime=data['runtime'],
                    result=data['result'],
                    additional_argument = data['additional_argument'],
                    benchmark=benchmark,
                    solver=solver,
                    task=task,
                    exit_with_error=
                    data['exit_with_error'],
                    error_code=data['error_code'])
    return result_obj

def _bulk_instert_results(result_paths,session):
    for solver_result_paths in result_paths:
        for result_file in solver_result_paths:
            session.add(_create_result_obj_from_json(result_file,session))

    #session.commit()



def _single_run(solver,task,bench,timeout,save_db,tag,n,first_n_instances,multi):

    paths = solver.run(task,
                           bench,
                           timeout,
                           save_db=save_db,
                           tag=tag,
                           session=None,
                           n=n,
                           first_n_instances=first_n_instances,
                           multi=multi)

    click.echo(f"{solver.solver_full_name} Task: {task.symbol}---FINISHED")
    return paths

def run_experiment(parameters: dict):
    """[summary]

    Args:
        parameters (dict): [description]
    """
    tag = parameters['tag']
    tasks = parameters['task']
    benchmarks = parameters['benchmark']
    timeout = parameters['timeout']
    dry = parameters['dry']
    session = parameters['session']
    select = parameters['select']
    n = parameters['n_times']
    first_n_instances = parameters['subset']
    if n > 1:
        multiple_runs = True
    else:
        multiple_runs = False

    for task in tasks:
        task_symbol = task.symbol.upper()
        click.echo(f"**********{task_symbol}***********")
        for bench in benchmarks:
            click.echo(f"Benchmark: {bench.benchmark_name}")
            if select:
                solvers_to_run = set(task.solvers).intersection(
                    set(parameters['solver']))
            else:

                solvers_to_run = task.solvers
            if task.symbol.endswith("-D"):

                for solver in solvers_to_run:
                    click.echo(solver.solver_full_name, nl=False)
                    for i in range(1,n+1):
                        solver.run_dynamic(task,
                                    bench,
                                    timeout,
                                    save_db=(not dry),
                                    tag=tag,
                                    session=session,
                                    n=i,
                                    first_n_instances=first_n_instances, multiple_runs=multiple_runs)
                    click.echo("---FINISHED")
            else:
                for solver in solvers_to_run:
                    click.echo(solver.solver_full_name, nl=False)
                    for i in range(1,n+1):
                        solver.run(task,
                            bench,
                            timeout,
                            save_db=(not dry),
                            tag=tag,
                            session=session,
                            n=i,
                            first_n_instances=first_n_instances, multiple_runs=multiple_runs)
                    click.echo("---FINISHED")
        Status.increment_task_counter()

def create_file_name(df: pandas.DataFrame) -> str:
    """[summary]

    Args:
        df ([type]): [description]
        post_hoc ([type]): [description]

    Returns:
        [type]: [description]
    """
    print(df)
    unique_tags = "_".join(list(df['tag'].unique()))
    unique_benchmarks = "_".join(list(df['benchmark'].unique()))
    unique_tasks = "_".join(list(df['task'].unique()))
    file_name = f'{unique_tags}-{unique_benchmarks}-{unique_tasks}'
    return file_name

def dispatch_on_value(func):
    """
    Value-dispatch function decorator.

    Transforms a function into a value-dispatch function,
    which can have different behaviors based on the value of the first argument.
    """

    registry = {}

    def dispatch(value):

        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):

        if func is None:
            return lambda f: register(value, f)

        registry[value] = func

        return func

    def wrapper(*args, **kw):
        return dispatch(args[0])(*args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


PIPE = -1
STDOUT = -2
DEVNULL = -3

_mswindows = False

def run_process(*popenargs,
        input=None, capture_output=False, timeout=None, check=False, **kwargs):
    """Run command with arguments and return a CompletedProcess instance.

    The returned instance will have attributes args, returncode, stdout and
    stderr. By default, stdout and stderr are not captured, and those attributes
    will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them.

    If check is True and the exit code was non-zero, it raises a
    CalledProcessError. The CalledProcessError object will have the return code
    in the returncode attribute, and output & stderr attributes if those streams
    were captured.

    If timeout is given, and the process takes too long, a TimeoutExpired
    exception will be raised.

    There is an optional argument "input", allowing you to
    pass bytes or a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's "stdin" argument, as
    it will be used internally.

    By default, all communication is in bytes, and therefore any "input" should
    be bytes, and the stdout and stderr will be bytes. If in text mode, any
    "input" should be a string, and stdout and stderr will be strings decoded
    according to locale encoding, or by "encoding" if set. Text mode is
    triggered by setting any of text, encoding, errors or universal_newlines.

    The other arguments are the same as for the Popen constructor.
    """
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = PIPE
        kwargs['stderr'] = PIPE

    with Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    continue
            parent.kill()
            process.kill()
            if _mswindows:
                # Windows accumulates the output in a single blocking
                # read() call run on child threads, with the timeout
                # being done in a join() on those threads.  communicate()
                # _after_ kill() is required to collect that and add it
                # to the exception.
                exc.stdout, exc.stderr = process.communicate()
            else:
                # POSIX _communicate already populated the output so
                # far into the TimeoutExpired exception.
                process.wait()
            raise
        except:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr)

def get_from_last_experiment(key):
    with open(definitions.LAST_EXPERIMENT_JSON_PATH,'r') as file:
        json_string = file.read()
    json_obj = json.loads(json_string)

    if key in json_obj.keys():
        return json_obj[key]

def get_last_experiment_json():
    with open(definitions.LAST_EXPERIMENT_JSON_PATH,'r') as file:
        json_string = file.read()
    json_obj = json.loads(json_string)

    return json_obj

def set_run_parameters(run_parameter, config):
    run_parameter['track'] = config['Track']
    run_parameter['task'] = config['Tasks']
    run_parameter['benchmark'] = config['Benchmark IDs']
    run_parameter['solver'] = config['Solver IDs']
    run_parameter['timeout'] = config['Timeout']
    run_parameter['notify'] = config['Notify']
    run_parameter['n_times'] = config['n_times']

    if config['Mode'] == 'select':
        run_parameter['select'] = True
        run_parameter['all'] = False
    else:
        run_parameter['select'] = False
        run_parameter['all'] = True

    last_tag = config['Tag']
    run_parameter['tag'] = f'{last_tag}_rerun'

def check_solver_paths(solvers):
    valid_solver_paths = []
    for solver in solvers:
        if os.path.isfile(solver.solver_path):
            valid_solver_paths.append(solver)
        else:
            print(f'Solver {solver.solver_full_name} was excluded because of a invalid path to the executable.')

    return valid_solver_paths

def init_files():
    pass






