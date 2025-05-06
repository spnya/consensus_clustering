import json
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

from flask import Flask, request, jsonify
import redis
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc, func
from task_queue_api import init_app as init_task_api

from config import REDIS_HOST, REDIS_PORT
from models import Experiment, Task, Result, Worker, init_db, get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Database initialization
with app.app_context():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

# Initialize Redis connection
def get_redis_connection():
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        redis_client.ping()  # Check connection
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

redis_client = get_redis_connection()

# Initialize task queue API
init_task_api(app)

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

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running"""
    redis_healthy = False
    if redis_client:
        try:
            redis_client.ping()
            redis_healthy = True
        except:
            pass
    
    # Check DB connection
    db_healthy = False
    try:
        with session_scope() as session:
            session.execute("SELECT 1")
            db_healthy = True
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
    
    status = "healthy" if redis_healthy and db_healthy else "unhealthy"
    
    return jsonify({
        "status": status,
        "redis": "connected" if redis_healthy else "disconnected",
        "database": "connected" if db_healthy else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

# Create a new experiment
@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    """Create a new experiment"""
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({"error": "Experiment name is required"}), 400
    
    try:
        new_experiment = Experiment(
            name=data.get('name'),
            description=data.get('description'),
            parameters=data.get('parameters', {})
        )
        
        with session_scope() as session:
            session.add(new_experiment)
            session.flush()  # To get the ID before committing
            
            # Convert to dict to avoid SQLAlchemy object issues
            result = new_experiment.to_dict()
        
        return jsonify(result), 201
    
    except SQLAlchemyError as e:
        logger.error(f"Database error creating experiment: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return jsonify({"error": str(e)}), 500

# Get all experiments
@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Get all experiments"""
    try:
        with session_scope() as session:
            experiments = session.query(Experiment).order_by(desc(Experiment.created_at)).all()
            result = [exp.to_dict() for exp in experiments]
        
        return jsonify(result), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting experiments: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        return jsonify({"error": str(e)}), 500

# Register a worker
@app.route('/api/workers', methods=['POST'])
def register_worker():
    """Register a new worker or update existing one"""
    data = request.json
    
    if not data or not data.get('worker_id'):
        return jsonify({"error": "worker_id is required"}), 400
    
    try:
        with session_scope() as session:
            worker = session.query(Worker).filter_by(id=data.get('worker_id')).first()
            
            if worker:
                # Update existing worker
                worker.hostname = data.get('hostname')
                worker.ip_address = data.get('ip_address')
                worker.status = 'active'
                worker.last_heartbeat = datetime.utcnow()
                worker.worker_metadata = data.get('metadata', {})
            else:
                # Create new worker
                worker = Worker(
                    id=data.get('worker_id'),
                    hostname=data.get('hostname'),
                    ip_address=data.get('ip_address'),
                    status='active',
                    worker_metadata=data.get('metadata', {})
                )
                session.add(worker)
            
            session.flush()
            worker_dict = worker.to_dict()
        
        return jsonify(worker_dict), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error registering worker: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error registering worker: {e}")
        return jsonify({"error": str(e)}), 500

# Get all workers
@app.route('/api/workers', methods=['GET'])
def get_workers():
    """Get all registered workers"""
    try:
        with session_scope() as session:
            workers = session.query(Worker).all()
            result = [worker.to_dict() for worker in workers]
        
        return jsonify(result), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting workers: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting workers: {e}")
        return jsonify({"error": str(e)}), 500

# Get worker status (checks if workers are active based on heartbeat)
@app.route('/api/workers/status', methods=['GET'])
def get_workers_status():
    """Get status of all workers"""
    try:
        # Consider a worker inactive if no heartbeat in last 2 minutes
        heartbeat_threshold = datetime.utcnow() - timedelta(minutes=2)
        
        with session_scope() as session:
            workers = session.query(Worker).all()
            
            result = {
                "total": len(workers),
                "active": 0,
                "inactive": 0,
                "workers": []
            }
            
            for worker in workers:
                is_active = worker.last_heartbeat and worker.last_heartbeat > heartbeat_threshold
                
                # Update status in database if needed
                if is_active and worker.status != 'active':
                    worker.status = 'active'
                elif not is_active and worker.status == 'active':
                    worker.status = 'inactive'
                
                worker_info = worker.to_dict()
                worker_info['is_active'] = is_active
                
                result['workers'].append(worker_info)
                if is_active:
                    result['active'] += 1
                else:
                    result['inactive'] += 1
        
        return jsonify(result), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting worker status: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        return jsonify({"error": str(e)}), 500

# Get worker statistics
@app.route('/api/workers/<worker_id>/stats', methods=['GET'])
def get_worker_stats(worker_id):
    """Get statistics for a specific worker"""
    try:
        with session_scope() as session:
            # Check if the worker exists
            worker = session.query(Worker).filter_by(id=worker_id).first()
            if not worker:
                return jsonify({"error": "Worker not found"}), 404
            
            # Get task processing statistics
            results = session.query(Result).filter_by(worker_id=worker_id).all()
            
            # Calculate stats
            total_tasks = len(results)
            avg_processing_time = 0
            if total_tasks > 0:
                avg_processing_time = sum(r.processing_time or 0 for r in results) / total_tasks
            
            # Group by experiment
            experiments_processed = session.query(Experiment.id, Experiment.name, func.count(Result.id))\
                .join(Task, Task.id == Result.task_id)\
                .join(Experiment, Experiment.id == Task.experiment_id)\
                .filter(Result.worker_id == worker_id)\
                .group_by(Experiment.id, Experiment.name)\
                .all()
            
            experiments = [
                {
                    "id": exp_id, 
                    "name": exp_name, 
                    "tasks_processed": task_count
                } 
                for exp_id, exp_name, task_count in experiments_processed
            ]
            
            # Calculate tasks over time
            tasks_over_time = session.query(
                    func.date_trunc('hour', Result.created_at).label('hour'),
                    func.count(Result.id)
                )\
                .filter(Result.worker_id == worker_id)\
                .group_by('hour')\
                .order_by('hour')\
                .all()
            
            hourly_stats = [
                {
                    "hour": hour.isoformat() if hour else None,
                    "tasks_processed": count
                }
                for hour, count in tasks_over_time
            ]
            
            # Prepare result
            stats = {
                "worker_id": worker_id,
                "total_tasks_processed": total_tasks,
                "average_processing_time": avg_processing_time,
                "first_task": results[0].created_at.isoformat() if total_tasks > 0 else None,
                "last_task": results[-1].created_at.isoformat() if total_tasks > 0 else None,
                "experiments": experiments,
                "hourly_stats": hourly_stats
            }
        
        return jsonify(stats), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting worker stats: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting worker stats: {e}")
        return jsonify({"error": str(e)}), 500

# Get a specific worker
@app.route('/api/workers/<worker_id>', methods=['GET'])
def get_worker(worker_id):
    """Get a specific worker by ID"""
    try:
        with session_scope() as session:
            worker = session.query(Worker).filter_by(id=worker_id).first()
            
            if not worker:
                return jsonify({"error": "Worker not found"}), 404
            
            result = worker.to_dict()
        
        return jsonify(result), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting worker: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting worker: {e}")
        return jsonify({"error": str(e)}), 500

# Worker heartbeat
@app.route('/api/workers/<worker_id>/heartbeat', methods=['POST'])
def worker_heartbeat(worker_id):
    """Update worker heartbeat"""
    try:
        # Get additional data from request if available
        data = request.json or {}
        
        with session_scope() as session:
            worker = session.query(Worker).filter_by(id=worker_id).first()
            
            if not worker:
                return jsonify({"error": "Worker not found"}), 404
            
            worker.last_heartbeat = datetime.utcnow()
            worker.status = 'active'
            
            # Update metadata with active tasks if provided
            if worker.worker_metadata is None:
                worker.worker_metadata = {}
                
            if 'active_tasks' in data:
                # Create or update the active_tasks field in metadata
                metadata = worker.worker_metadata.copy() if worker.worker_metadata else {}
                metadata['active_tasks'] = data['active_tasks']
                metadata['max_tasks'] = data.get('max_tasks', 1)  # Default to 1 if not provided
                metadata['last_updated'] = datetime.utcnow().isoformat()
                worker.worker_metadata = metadata
        
        return jsonify({"status": "success", "worker_id": worker_id}), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error updating worker heartbeat: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error updating worker heartbeat: {e}")
        return jsonify({"error": str(e)}), 500


# Get workers for an experiment
@app.route('/api/experiments/<int:experiment_id>/workers', methods=['GET'])
def get_experiment_workers(experiment_id):
    """Get all workers that processed tasks for a specific experiment"""
    try:
        with session_scope() as session:
            # Check if the experiment exists
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return jsonify({"error": "Experiment not found"}), 404
            
            # Query for unique worker IDs that have processed tasks for this experiment
            worker_ids = session.query(Result.worker_id)\
                .join(Task, Result.task_id == Task.id)\
                .filter(Task.experiment_id == experiment_id)\
                .distinct()\
                .all()
            
            worker_ids = [w[0] for w in worker_ids]  # Flatten the result
            
            # Get full worker details
            workers = []
            if worker_ids:
                workers_data = session.query(Worker)\
                    .filter(Worker.id.in_(worker_ids))\
                    .all()
                workers = [w.to_dict() for w in workers_data]
            
            result = {
                "experiment_id": experiment_id,
                "worker_count": len(workers),
                "workers": workers
            }
        
        return jsonify(result), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting experiment workers: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting experiment workers: {e}")
        return jsonify({"error": str(e)}), 500

# Get results for an experiment
@app.route('/api/experiments/<int:experiment_id>/results', methods=['GET'])
def get_experiment_results(experiment_id):
    """Get all results for a specific experiment"""
    try:
        with session_scope() as session:
            # Check if the experiment exists
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return jsonify({"error": "Experiment not found"}), 404
            
            # Query for results with task data
            results = []
            for result in session.query(Result).join(Task).filter(Task.experiment_id == experiment_id).order_by(desc(Result.created_at)).all():
                result_dict = result.to_dict()
                # Add task_data to the result
                result_dict['task_data'] = result.task.task_data
                results.append(result_dict)
        
        return jsonify(results), 200
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting experiment results: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)