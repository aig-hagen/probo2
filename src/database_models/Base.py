from sqlalchemy import Column, ForeignKey, Integer, String, Table, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

Supported_Tasks = Table(
  "Supported_Tasks",
  Base.metadata,
  Column("solver_id", Integer, ForeignKey("solvers.id")),
  Column("problem_id", Integer, ForeignKey("tasks.id")),
)