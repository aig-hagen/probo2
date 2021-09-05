from sqlalchemy import create_engine
from sqlalchemy import and_, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.util.langhelpers import symbol

import src.Models as Models
import src.definitions as Definitions
from src.Models import Result, Solver, Task, Benchmark

def get_engine():
    engine = None
    try:
        engine = create_engine(f"sqlite:///{Definitions.TEST_DATABASE_PATH}")
    except Exception as err:
        print(err)
    return engine

def init_database(engine):
    Models.Base.metadata.create_all(engine)
    session = create_session(engine)
    task_objs = []
    for task in Definitions.SUPPORTED_TASKS:
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


def add_solver(session, solver, tasks):
    new_solver = (
        session.query(Solver)
            .filter(
            and_(
                Solver.name == solver.name, Solver.path == solver.path, Solver.version == solver.version
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
    print(new_solver.supported_tasks)
    return new_solver.id

def add_benchmark(session, benchmark):
    """
        Adds a new solver into the solvers table
        """
    # check if solver already exists
    new_benchmark = (
        session.query(Benchmark)
            .filter(
            and_(
                Benchmark.name == benchmark.name, Benchmark.path == benchmark.path
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
    # print(len(session.query(Result).all()))
    # print(result.__dict__)
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

def get_solver(session, to_get):
    queried_solver = (session.query(Solver).filter(or_(Solver.id.in_(to_get),Solver.name.in_(to_get)))).all()
    return queried_solver

def get_benchmark(session, to_get):
    queried_benchmark = (session.query(Benchmark).filter(or_(Benchmark.id.in_(to_get),Benchmark.name.in_(to_get)))).all()
    return queried_benchmark

def get_task(session, to_get):
    queried_task = (session.query(Task).filter(or_(Task.id.in_(to_get),Task.symbol.in_(to_get)))).all()
    return queried_task

def insert_results(session,task,benchmark,solver,timeout, tag, results: dict):
    for instance, data in results.items():
        result = Result(tag=tag,solver_id=solver.id,benchmark_id = benchmark.id,task_id = task.id,
                            instance=instance,cut_off=timeout, timed_out = data['timed_out'],
                            runtime=data['runtime'], result=data['result'], additional_argument = data['additional_argument'],
                            benchmark=benchmark, solver=solver, task=task)
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
    return session.query(Solver).filter(Solver.id == id).one().fullname

def map_ids_to_name(ids):
    print(ids)






   


