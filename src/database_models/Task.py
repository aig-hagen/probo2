
from src.database_models.Base import Base, Supported_Tasks
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    solvers = relationship("Solver", secondary=Supported_Tasks, back_populates="supported_tasks")
    results = relationship('Result')