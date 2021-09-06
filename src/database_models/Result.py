
from sqlalchemy import Column, ForeignKey, Integer, String,Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import Float

from src.database_models.Base import Base, Supported_Tasks

class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    tag = Column(String, nullable=False)
    solver_id = Column(Integer, ForeignKey('solvers.id'))
    benchmark_id = Column(Integer, ForeignKey('benchmarks.id'))
    task_id = Column(Integer,ForeignKey('tasks.id'))
    instance = Column(String,nullable=False)
    cut_off = Column(Integer, nullable=False)
    timed_out = Column(Boolean, nullable=False)
    exit_with_error = Column(Boolean, nullable=False)
    error_code = Column(Integer,nullable=True )
    runtime = Column(Float,nullable=True)
    result = Column(String,nullable=True)
    additional_argument= Column(String, nullable=True)
    benchmark = relationship("Benchmark", back_populates="results")
    solver = relationship("Solver", back_populates="results")
    task = relationship("Task", back_populates="results")
