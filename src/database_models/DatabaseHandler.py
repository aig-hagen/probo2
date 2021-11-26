import pandas


from sqlalchemy import create_engine
from sqlalchemy import and_, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import label
from sqlalchemy.util.langhelpers import symbol



from src.database_models.Base import Base, Supported_Tasks
import src.utils.definitions as definitions
from src.database_models.Solver import Solver
from src.database_models.Result import Result
from src.database_models.Task import Task
from src.database_models.Benchmark import Benchmark

def get_engine():
    engine = None
    try:
        engine = create_engine(f"sqlite:///{definitions.TEST_DATABASE_PATH}")
    except Exception as err:
        print(err)
    return engine

def init_database(engine):
    Base.metadata.create_all(engine)
    session = create_session(engine)
    task_objs = []
    for task in definitions.SUPPORTED_TASKS:
        task_objs.append(Task(symbol=task))
    add_objects(session,task_objs)
    session.close()


def add_objects(session, objects: list):
    session.bulk_save_objects(objects)
    session.commit()




def create_session(engine):
    """
        Adds a new solver into the solvers table
        :param engine: active engine
        :return: new_session: active sql session
    """
    Session = sessionmaker()
    Session.configure(bind=engine)
    new_session = Session()
    return new_session

def delete_solver(session, solver_id):
    """
           Deletes a  solver from the solvers table

    """
    solver_to_delete = (
        session.query(Solver).filter(Solver.solver_id == solver_id
                                      ).one_or_none()
    )

    if solver_to_delete is None:
        raise ValueError("Solver does not exist")
    else:
        session.delete(solver_to_delete)

def delete_benchmark(session, benchmark_id):
    """
           Deletes a benchmark from the benchmarks table

    """
    benchmark_to_delete = (
        session.query(Benchmark).filter(Benchmark.id == benchmark_id
                                      ).one_or_none()
    )

    if benchmark_to_delete is None:
        raise ValueError("Solver does not exist")
    else:
        session.delete(benchmark_to_delete)

def add_solver(session, solver, tasks):
    new_solver = (
        session.query(Solver)
            .filter(
            and_(
                Solver.solver_name == solver.solver_name, Solver.solver_path == solver.solver_path, Solver.solver_version == solver.solver_version
            )
        )
            .one_or_none()
    )

    if new_solver is None:
        new_solver = solver
        for task in tasks:
            current_task = session. \
            query(Task). \
            filter(Task.symbol == task).one_or_none()
            new_solver.supported_tasks.append(current_task)
        session.add(new_solver)
        session.flush()
        session.refresh(new_solver)

    else:
        raise ValueError("Solver already in Database!")
    return new_solver.solver_id
def get_supported_tasks(session):
    supported_tasks = session.query(Task).all()
    symbols = [x.symbol for x in supported_tasks]
    return symbols
def add_task(session,task_symbol):
    new_task = (
        session.query(Task)
            .filter(
                Task.symbol == task_symbol
            )
            .one_or_none()
    )
        # create solver object
    if new_task is None:
        new_task = Task(symbol=task_symbol)
        session.add(new_task)
        session.flush()
        session.refresh(new_task)
    else:
        raise ValueError(f"Task {task_symbol} already in Database!")

    return new_task.id



def add_benchmark(session, benchmark):
    """
        Adds a new solver into the solvers table
        """
    # check if solver already exists
    new_benchmark = (
        session.query(Benchmark)
            .filter(
            and_(
                Benchmark.benchmark_name == benchmark.benchmark_name, Benchmark.benchmark_path == benchmark.benchmark_path
            )
        )
            .one_or_none()
    )
    # create solver object
    if new_benchmark is None:
        new_benchmark = benchmark
        session.add(new_benchmark)
        session.flush()
        session.refresh(new_benchmark)
    else:
        raise ValueError("Solver already in Database!")

    return new_benchmark.id

def add_result(session,result):
    new_result =  (
        session.query(Result)
            .filter(
            and_(
                Result.tag == result.tag, Result.solver_id == result.solver_id, Result.task_id == result.task_id, Result.benchmark_id == result.benchmark_id, Result.instance == result.instance
            )
        )
            .one_or_none()
    )
    if new_result is None:
        new_result = result
        session.add(new_result)
    else:
         raise ValueError("Result already in Database!")

def get_results(session,solver,task, benchmark,tag,filter,only_solved=False,validated=False) -> pandas.DataFrame:

    res  = session.query(Result.id,
                        Solver.solver_id,
                        Solver.solver_full_name,
                        Result.instance,
                        Solver.solver_format,
                        Result.runtime,
                        Result.task_id,
                        Task.symbol,
                        Result.result,
                        Result.cut_off,
                        Result.timed_out,
                        Result.exit_with_error,
                        Result.error_code,
                        Result.additional_argument,
                        Result.benchmark_id,
                        Result.correct,
                        Result.correct_solved,
                        Result.incorrect_solved,
                        Result.no_reference,
                        Result.validated,
                        Benchmark.benchmark_name,
                        Result.tag).join(Solver,Solver.solver_id == Result.solver_id).join(Benchmark,Benchmark.id==Result.benchmark_id).join(Task, Task.id == Result.task_id)

    if tag:
        res = res.filter(Result.tag.in_(tag))
    if solver:
        res = res.filter(or_(Solver.solver_name.in_(solver), Solver.solver_id.in_(solver)))
    if benchmark:
        res = res.filter(or_(Benchmark.benchmark_name.in_(benchmark), Benchmark.id.in_(benchmark)))
    if task:
        res = res.filter(or_(Task.symbol.in_(task), Task.id.in_(task)))
    if filter:
        res = res.filter_by(**filter)

    result_df = pandas.read_sql(res.statement,res.session.bind)
    if only_solved:
        return result_df[(result_df['timed_out'] == False) & (result_df['exit_with_error'] == False)]
    else:
        return result_df



def get_solvers(session, to_get):
    # Get solver by id or name
    queried_solver = (session.query(Solver).filter(or_(Solver.solver_id.in_(to_get),Solver.solver_name.in_(to_get)))).all()
    return queried_solver

def get_solver(session, to_get):
    queried_solver = (session.query(Solver).filter(or_(Solver.solver_id == to_get,Solver.solver_name == to_get))).one()
    return queried_solver


def get_benchmarks(session, to_get):
    queried_benchmark = (session.query(Benchmark).filter(or_(Benchmark.id.in_(to_get),Benchmark.benchmark_name.in_(to_get)))).all()
    return queried_benchmark

def get_benchmark(session, to_get):
    queried_benchmark = (session.query(Benchmark).filter(or_(Benchmark.id == to_get,Benchmark.benchmark_name == to_get))).first()
    return queried_benchmark

def get_tasks(session, to_get):
    queried_task = (session.query(Task).filter(or_(Task.id.in_(to_get),Task.symbol.in_(to_get)))).all()
    return queried_task

def get_task(session, to_get):
    queried_task = (session.query(Task).filter(or_(Task.id == to_get,Task.symbol == to_get))).first()
    return queried_task

def insert_results(session,task,benchmark,solver,timeout, tag, results: dict):
    for instance, data in results.items():
        result = Result(tag=tag,solver_id=solver.id,benchmark_id = benchmark.id,task_id = task.id,
                            instance=instance,cut_off=timeout, timed_out = data['timed_out'],
                            runtime=data['runtime'], result=data['result'], additional_argument = data['additional_argument'],
                            benchmark=benchmark, solver=solver, task=task, exit_with_error=data['exit_with_error'], error_code=data['error_code'])
        session.add(result)

    session.commit()

def tag_in_database(session, tag):
    tag = (session.query(Result)
            .filter(Result.tag ==tag)).first()
    if tag is None:
        return False
    else:
        return True

def get_full_name_solver(session,id):
    return session.query(Solver).filter(Solver.solver_id == id).one().solver_full_name

def map_ids_to_name(ids):
    print(ids)










