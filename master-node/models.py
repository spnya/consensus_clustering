import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from config import get_database_url

Base = declarative_base()

class Experiment(Base):
    """Model for clustering experiments"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    parameters = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tasks = relationship("Task", back_populates="experiment", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Task(Base):
    """Model for tasks sent to workers"""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    task_data = Column(JSONB, nullable=False)
    status = Column(String(50), default='pending', nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="tasks")
    results = relationship("Result", back_populates="task", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "task_data": self.task_data,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Result(Base):
    """Model for task results"""
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id'))
    worker_id = Column(String(255), nullable=False)
    result_data = Column(JSONB, nullable=False)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="results")
    
    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "result_data": self.result_data,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Worker(Base):
    """Model for worker nodes"""
    __tablename__ = 'workers'
    
    id = Column(String(255), primary_key=True)
    hostname = Column(String(255))
    ip_address = Column(String(50))
    status = Column(String(50), default='active', nullable=False)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    worker_metadata = Column(JSONB)
    
    def to_dict(self):
        return {
            "id": self.id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "metadata": self.worker_metadata
        }

# Using the imported get_database_url function from config.py

def init_db():
    """Initialize database connection and create tables if they don't exist"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def get_db_session():
    """Get a new database session"""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session()