import json
import redis
import logging
import time
from datetime import datetime
from models import Task, Worker, Experiment, get_db_session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

logger = logging.getLogger(__name__)

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

class TaskQueueService:
    """Intelligent task queue with worker capacity awareness"""
    
    def __init__(self, redis_host='redis', redis_port=6379, redis_db=0):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.queue_key = 'tasks:queue'
        self.processing_key = 'tasks:processing'
        self.worker_stats_key = 'workers:stats'
        
    def enqueue_task(self, task):
        """Add a task to the queue with priority"""
        task_id = task.get('id')
        task_json = json.dumps(task)
        
        # Store task data
        self.redis.hset('tasks:data', task_id, task_json)
        
        # Add to queue
        self.redis.zadd(self.queue_key, {str(task_id): int(time.time())})
        
        logger.info(f"Task {task_id} added to queue. Queue length: {self.redis.zcard(self.queue_key)}")
        return True
    
    def _get_worker_capacity(self, worker_id):
        """Get current capacity info for a worker"""
        stats = self.redis.hget(self.worker_stats_key, worker_id)
        if not stats:
            return 0, 0
        
        stats = json.loads(stats)
        max_tasks = stats.get('max_tasks', 1)
        active_tasks = stats.get('active_tasks', 0)
        available = max(0, max_tasks - active_tasks)
        
        return available, max_tasks
    
    def _get_best_worker(self):
        """Find the worker with most available capacity"""
        workers = self.redis.hgetall(self.worker_stats_key)
        if not workers:
            return None
        
        best_worker = None
        max_available = -1
        
        for worker_id, stats in workers.items():
            worker_id = worker_id.decode('utf-8')
            stats = json.loads(stats)
            
            # Check if worker is active (last heartbeat within 30 seconds)
            last_heartbeat = stats.get('last_heartbeat', 0)
            if time.time() - last_heartbeat > 30:
                continue
            
            max_tasks = stats.get('max_tasks', 1)
            active_tasks = stats.get('active_tasks', 0)
            available = max(0, max_tasks - active_tasks)
            
            # Factor in load percentage to avoid overloading nearly-full workers
            capacity_percentage = available / max_tasks if max_tasks > 0 else 0
            
            # Use a scoring formula that considers both absolute capacity and percentage
            # This helps balance between large workers with little free capacity
            # and small workers with more free capacity
            score = (available * 0.7) + (capacity_percentage * 0.3 * max_tasks)
            
            if score > max_available:
                max_available = score
                best_worker = worker_id
        
        return best_worker
    
    def update_worker_stats(self, worker_id, stats):
        """Update worker stats"""
        # Add timestamp to stats
        stats['last_heartbeat'] = int(time.time())
        
        # Store in Redis
        self.redis.hset(self.worker_stats_key, worker_id, json.dumps(stats))
        
        # Set expiration (worker considered dead after 60 seconds)
        self.redis.expire(f"{self.worker_stats_key}:{worker_id}", 60)
    
    def assign_task(self, worker_id=None):
        """Assign a task to a worker, respecting capacity"""
        # If no worker_id provided, find the best worker
        if not worker_id:
            worker_id = self._get_best_worker()
            if not worker_id:
                logger.warning("No active workers available to process tasks")
                return None
        
        # Check worker capacity
        available, max_tasks = self._get_worker_capacity(worker_id)
        if available <= 0:
            logger.info(f"Worker {worker_id} is at capacity ({max_tasks}/{max_tasks})")
            return None
        
        # Get the next task
        task_id = self.redis.zpopmin(self.queue_key, 1)
        if not task_id or len(task_id) == 0:
            return None
        
        task_id = task_id[0][0].decode('utf-8')
        
        # Get task data
        task_json = self.redis.hget('tasks:data', task_id)
        if not task_json:
            logger.error(f"Task {task_id} data not found")
            return None
        
        task = json.loads(task_json)
        
        # Mark as processing by this worker
        self.redis.hset(self.processing_key, task_id, worker_id)
        
        # Update worker stats
        stats = self.redis.hget(self.worker_stats_key, worker_id)
        if stats:
            stats = json.loads(stats)
            stats['active_tasks'] = stats.get('active_tasks', 0) + 1
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(stats))
        
        # Update task status in database
        try:
            with session_scope() as session:
                db_task = session.query(Task).filter_by(id=int(task_id)).first()
                if db_task:
                    db_task.status = 'processing'
                    db_task.updated_at = datetime.utcnow()
        except SQLAlchemyError as e:
            logger.error(f"Database error updating task status: {e}")
        
        logger.info(f"Task {task_id} assigned to worker {worker_id}")
        return task
    
    def complete_task(self, task_id, worker_id, result_data):
        """Mark a task as completed"""
        # Verify this worker owns the task
        assigned_worker = self.redis.hget(self.processing_key, str(task_id))
        if not assigned_worker or assigned_worker.decode('utf-8') != worker_id:
            logger.warning(f"Worker {worker_id} attempted to complete task {task_id} not assigned to it")
            return False
        
        # Remove from processing
        self.redis.hdel(self.processing_key, str(task_id))
        
        # Update worker stats
        stats = self.redis.hget(self.worker_stats_key, worker_id)
        if stats:
            stats = json.loads(stats)
            stats['active_tasks'] = max(0, stats.get('active_tasks', 0) - 1)
            stats['completed_tasks'] = stats.get('completed_tasks', 0) + 1
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(stats))
        
        # Task data can be kept for history
        logger.info(f"Task {task_id} completed by worker {worker_id}")
        return True
    
    def requeue_lost_tasks(self):
        """Requeue tasks from workers that haven't sent heartbeats"""
        processing_tasks = self.redis.hgetall(self.processing_key)
        requeued = 0
        
        for task_id, worker_id in processing_tasks.items():
            task_id = task_id.decode('utf-8')
            worker_id = worker_id.decode('utf-8')
            
            # Check if worker is still active
            stats = self.redis.hget(self.worker_stats_key, worker_id)
            if not stats:
                # Worker disappeared, requeue task
                self._requeue_task(task_id)
                requeued += 1
            else:
                stats = json.loads(stats)
                last_heartbeat = stats.get('last_heartbeat', 0)
                
                # If no heartbeat in 30 seconds, requeue task
                if time.time() - last_heartbeat > 30:
                    self._requeue_task(task_id)
                    requeued += 1
        
        return requeued
    
    def _requeue_task(self, task_id):
        """Requeue a single task"""
        # Remove from processing
        self.redis.hdel(self.processing_key, task_id)
        
        # Add back to queue with current timestamp
        self.redis.zadd(self.queue_key, {task_id: int(time.time())})
        
        # Update task status in database
        try:
            with session_scope() as session:
                db_task = session.query(Task).filter_by(id=int(task_id)).first()
                if db_task:
                    db_task.status = 'pending'
                    db_task.updated_at = datetime.utcnow()
        except SQLAlchemyError as e:
            logger.error(f"Database error updating task status: {e}")
        
        logger.info(f"Task {task_id} requeued")
    
    def steal_work(self, worker_id):
        """Steal work from overloaded workers"""
        # Check if requester has capacity
        available, max_tasks = self._get_worker_capacity(worker_id)
        if available <= 0:
            return None
        
        # Find overloaded workers
        overloaded_workers = []
        workers = self.redis.hgetall(self.worker_stats_key)
        
        for w_id, stats in workers.items():
            w_id = w_id.decode('utf-8')
            if w_id == worker_id:
                continue
                
            stats = json.loads(stats)
            max_w_tasks = stats.get('max_tasks', 1)
            active_w_tasks = stats.get('active_tasks', 0)
            
            # Worker is considered overloaded if it has more than 75% capacity used
            if active_w_tasks / max_w_tasks > 0.75:
                overloaded_workers.append(w_id)
        
        if not overloaded_workers:
            return None
        
        # Try to steal a task from an overloaded worker
        for w_id in overloaded_workers:
            # Find a task assigned to this worker
            for task_id, assigned_worker in self.redis.hscan_iter(self.processing_key):
                task_id = task_id.decode('utf-8')
                assigned_worker = assigned_worker.decode('utf-8')
                
                if assigned_worker == w_id:
                    # Found a task to steal
                    task_json = self.redis.hget('tasks:data', task_id)
                    if task_json:
                        task = json.loads(task_json)
                        
                        # Reassign to the stealing worker
                        self.redis.hset(self.processing_key, task_id, worker_id)
                        
                        # Update worker stats for both workers
                        self._update_worker_task_count(w_id, -1)
                        self._update_worker_task_count(worker_id, 1)
                        
                        logger.info(f"Task {task_id} stolen from worker {w_id} by worker {worker_id}")
                        return task
        
        return None
    
    def _update_worker_task_count(self, worker_id, delta):
        """Update worker task count"""
        stats = self.redis.hget(self.worker_stats_key, worker_id)
        if stats:
            stats = json.loads(stats)
            stats['active_tasks'] = max(0, stats.get('active_tasks', 0) + delta)
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(stats))
    
    def get_queue_stats(self):
        """Get statistics about the queue"""
        stats = {
            'queued_tasks': self.redis.zcard(self.queue_key),
            'processing_tasks': self.redis.hlen(self.processing_key),
            'active_workers': 0,
            'total_capacity': 0,
            'used_capacity': 0,
            'worker_details': []
        }
        
        # Get worker stats
        workers = self.redis.hgetall(self.worker_stats_key)
        
        for worker_id, worker_stats in workers.items():
            worker_id = worker_id.decode('utf-8')
            worker_stats = json.loads(worker_stats)
            
            # Check if worker is active (last heartbeat within 30 seconds)
            last_heartbeat = worker_stats.get('last_heartbeat', 0)
            is_active = time.time() - last_heartbeat <= 30
            
            if is_active:
                stats['active_workers'] += 1
                
                max_tasks = worker_stats.get('max_tasks', 1)
                active_tasks = worker_stats.get('active_tasks', 0)
                
                stats['total_capacity'] += max_tasks
                stats['used_capacity'] += active_tasks
                
                stats['worker_details'].append({
                    'worker_id': worker_id,
                    'max_tasks': max_tasks,
                    'active_tasks': active_tasks,
                    'available_capacity': max_tasks - active_tasks,
                    'last_heartbeat': last_heartbeat,
                    'completed_tasks': worker_stats.get('completed_tasks', 0)
                })
        
        stats['available_capacity'] = stats['total_capacity'] - stats['used_capacity']
        if stats['total_capacity'] > 0:
            stats['capacity_percent'] = (stats['used_capacity'] / stats['total_capacity']) * 100
        else:
            stats['capacity_percent'] = 0
            
        return stats