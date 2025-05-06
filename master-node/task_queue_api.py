from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import logging
from contextlib import contextmanager

from models import Task, Result, Worker, Experiment, get_db_session
from task_queue_service import TaskQueueService

logger = logging.getLogger(__name__)

# Initialize task queue service
task_queue = TaskQueueService()

task_api = Blueprint('task_api', __name__)

# Context manager for database sessions
@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = get_db_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

@task_api.route('/tasks/request', methods=['GET'])
def request_task():
    """Endpoint for workers to request tasks based on capacity"""
    worker_id = request.args.get('worker_id')
    
    if not worker_id:
        return jsonify({"error": "worker_id is required"}), 400
    
    # Update worker heartbeat in database
    try:
        with session_scope() as session:
            worker = session.query(Worker).filter_by(id=worker_id).first()
            
            if not worker:
                return jsonify({"error": "Worker not found"}), 404
            
            # Update worker status
            worker.last_heartbeat = datetime.utcnow()
            worker.status = 'active'
            
            # Get worker metadata
            metadata = worker.worker_metadata or {}
            max_tasks = metadata.get('max_tasks', 2)
            active_tasks = metadata.get('active_tasks', 0)
            
            # Update Redis with worker stats
            task_queue.update_worker_stats(worker_id, {
                'max_tasks': max_tasks,
                'active_tasks': active_tasks
            })
    
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        return jsonify({"error": "Database error"}), 500
    
    # Assign task to worker based on capacity
    task = task_queue.assign_task(worker_id)
    
    if not task:
        # Try to steal work if no regular tasks available
        task = task_queue.steal_work(worker_id)
        
        if not task:
            # No tasks available
            return "", 204
    
    return jsonify(task), 200

@task_api.route('/tasks/submit', methods=['POST'])
def submit_task_result():
    """Submit a completed task result"""
    data = request.json
    
    if not data or not data.get('task_id') or not data.get('worker_id') or 'result_data' not in data:
        return jsonify({"error": "task_id, worker_id, and result_data are required"}), 400
    
    task_id = data.get('task_id')
    worker_id = data.get('worker_id')
    result_data = data.get('result_data')
    processing_time = data.get('processing_time')
    
    # Mark task as completed in task queue
    if not task_queue.complete_task(task_id, worker_id, result_data):
        return jsonify({"error": "Task not assigned to this worker"}), 403
    
    # Save result to database
    try:
        with session_scope() as session:
            # Update task status
            task = session.query(Task).filter_by(id=task_id).first()
            if not task:
                return jsonify({"error": "Task not found"}), 404
            
            task.status = 'completed'
            task.updated_at = datetime.utcnow()
            
            # Create result
            new_result = Result(
                task_id=task_id,
                worker_id=worker_id,
                result_data=result_data,
                processing_time=processing_time
            )
            
            session.add(new_result)
            session.flush()
            
            result_dict = new_result.to_dict()
        
        return jsonify(result_dict), 201
    
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        return jsonify({"error": "Database error"}), 500

@task_api.route('/tasks/enqueue', methods=['POST'])
def enqueue_task():
    """Enqueue a new task"""
    data = request.json
    
    if not data or not data.get('experiment_id') or 'task_data' not in data:
        return jsonify({"error": "experiment_id and task_data are required"}), 400
    
    try:
        # Save task to database
        with session_scope() as session:
            new_task = Task(
                experiment_id=data.get('experiment_id'),
                task_data=data.get('task_data'),
                status='pending'
            )
            
            session.add(new_task)
            session.flush()
            
            # Convert to dict to avoid SQLAlchemy object issues
            task_dict = new_task.to_dict()
        
        # Add to task queue
        task_queue.enqueue_task(task_dict)
        
        return jsonify(task_dict), 201
    
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        return jsonify({"error": "Database error"}), 500

@task_api.route('/array-sum-task', methods=['POST'])
def create_array_sum_task():
    """Create a simple array sum task"""
    data = request.json
    
    if not data or 'array' not in data or not isinstance(data['array'], list):
        return jsonify({"error": "An array of numbers is required"}), 400
    
    experiment_id = data.get('experiment_id', 1)  # Default to first experiment if not specified
    
    task_data = {
        'operation': 'sum',
        'array': data['array']
    }
    
    try:
        # Save task to database
        with session_scope() as session:
            new_task = Task(
                experiment_id=experiment_id,
                task_data=task_data,
                status='pending'
            )
            
            session.add(new_task)
            session.flush()
            
            # Convert to dict to avoid SQLAlchemy object issues
            task_dict = new_task.to_dict()
        
        # Add to task queue
        task_queue.enqueue_task(task_dict)
        
        return jsonify(task_dict), 201
    
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        return jsonify({"error": "Database error"}), 500

@task_api.route('/queue/stats', methods=['GET'])
def get_queue_stats():
    """Get statistics about the task queue"""
    stats = task_queue.get_queue_stats()
    return jsonify(stats), 200

def init_app(app):
    """Initialize the task API blueprint"""
    app.register_blueprint(task_api, url_prefix='/api')