import json
import redis
import logging
import time
import numpy as np
from datetime import datetime
from models import Task, Result, Worker, Experiment, get_db_session
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
            
            # Scoring: 70% absolute capacity, 30% percentage
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
        # Expire after 60s
        self.redis.expire(f"{self.worker_stats_key}:{worker_id}", 60)
    
    def assign_task(self, worker_id=None):
        """Assign a task to a worker, respecting capacity"""
        if not worker_id:
            worker_id = self._get_best_worker()
            if not worker_id:
                logger.warning("No active workers available to process tasks")
                return None
        
        available, max_tasks = self._get_worker_capacity(worker_id)
        if available <= 0:
            logger.info(f"Worker {worker_id} is at capacity ({max_tasks}/{max_tasks})")
            return None
        
        popped = self.redis.zpopmin(self.queue_key, 1)
        if not popped:
            return None
        
        task_id = popped[0][0].decode('utf-8')
        task_json = self.redis.hget('tasks:data', task_id)
        if not task_json:
            logger.error(f"Task {task_id} data not found")
            return None
        
        task = json.loads(task_json)
        
        # Mark as processing
        self.redis.hset(self.processing_key, task_id, worker_id)
        # Increment worker's active count
        raw = self.redis.hget(self.worker_stats_key, worker_id)
        if raw:
            s = json.loads(raw)
            s['active_tasks'] = s.get('active_tasks', 0) + 1
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(s))
        
        # Update DB status
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
        """Mark a task as completed, then trigger consensus if appropriate"""
        assigned = self.redis.hget(self.processing_key, str(task_id))
        if not assigned or assigned.decode('utf-8') != worker_id:
            logger.warning(f"Worker {worker_id} attempted to complete task {task_id} not assigned to it")
            return False
        
        # Remove from processing
        self.redis.hdel(self.processing_key, str(task_id))
        # Update stats
        raw = self.redis.hget(self.worker_stats_key, worker_id)
        if raw:
            s = json.loads(raw)
            s['active_tasks'] = max(0, s.get('active_tasks', 0) - 1)
            s['completed_tasks'] = s.get('completed_tasks', 0) + 1
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(s))
        
        logger.info(f"Task {task_id} completed by worker {worker_id}")
        
        # --- NEW: if this was a base run, and all base runs for that dataset are done, build & enqueue consensus ---
        try:
            with session_scope() as session:
                db_task = session.query(Task).filter_by(id=int(task_id)).first()
                if db_task and db_task.task_data.get('run_type') == 'base':
                    exp_id = db_task.experiment_id
                    ds_idx = db_task.task_data['dataset_index']
                    # fetch all base tasks for this exp + dataset
                    all_tasks = session.query(Task).filter_by(experiment_id=exp_id).all()
                    base = [
                        t for t in all_tasks
                        if t.task_data.get('run_type')=='base'
                        and t.task_data.get('dataset_index')==ds_idx
                    ]
                    done = [t for t in base if t.status=='completed']
                    if base and len(done)==len(base):
                        # collect their labels
                        labels_list = []
                        for t in done:
                            res = session.query(Result).filter_by(task_id=t.id).first()
                            if res and 'labels' in res.result_data:
                                labels_list.append(res.result_data['labels'])
                        if labels_list:
                            n = len(labels_list[0])
                            M = np.zeros((n,n), float)
                            for lbls in labels_list:
                                arr = np.array(lbls)
                                eq = (arr[:,None] == arr[None,:]).astype(float)
                                M += eq
                            M = (M / len(labels_list)).tolist()
                            # enqueue consensus tasks
                            expt = session.query(Experiment).get(exp_id)
                            for algo in (expt.parameters or {}).get('consensus_algorithms', []):
                                cd = {
                                    'operation': algo,
                                    'algorithm': algo,
                                    'ensemble_size': expt.parameters.get('ensemble_size', 1),
                                    'dataset_index': ds_idx,
                                    'consensus_matrix': M,
                                    'ground_truth': db_task.task_data.get('ground_truth'),
                                    'run_type': 'consensus'
                                }
                                new_t = Task(
                                    experiment_id=exp_id,
                                    task_data=cd,
                                    status='pending'
                                )
                                session.add(new_t)
                                session.flush()
                                self.enqueue_task(new_t.to_dict())
        except Exception:
            logger.exception("Error scheduling consensus tasks")
        
        return True
    
    def requeue_lost_tasks(self):
        """Requeue tasks from workers that haven't sent heartbeats"""
        processing = self.redis.hgetall(self.processing_key)
        requeued = 0
        for tidb, widb in processing.items():
            tid, wid = tidb.decode('utf-8'), widb.decode('utf-8')
            raw = self.redis.hget(self.worker_stats_key, wid)
            if not raw or time.time() - json.loads(raw).get('last_heartbeat', 0) > 30:
                self._requeue_task(tid)
                requeued += 1
        return requeued
    
    def _requeue_task(self, task_id):
        """Requeue a single task"""
        self.redis.hdel(self.processing_key, task_id)
        self.redis.zadd(self.queue_key, {task_id: int(time.time())})
        try:
            with session_scope() as session:
                db_t = session.query(Task).filter_by(id=int(task_id)).first()
                if db_t:
                    db_t.status = 'pending'
                    db_t.updated_at = datetime.utcnow()
        except SQLAlchemyError as e:
            logger.error(f"Database error requeuing task: {e}")
        logger.info(f"Task {task_id} requeued")
    
    def steal_work(self, worker_id):
        """Steal work from overloaded workers"""
        available, _ = self._get_worker_capacity(worker_id)
        if available <= 0:
            return None
        
        # find overloaded
        overloaded = []
        for widb, raw in self.redis.hgetall(self.worker_stats_key).items():
            wid = widb.decode('utf-8')
            if wid == worker_id:
                continue
            s = json.loads(raw)
            mt, at = s.get('max_tasks',1), s.get('active_tasks',0)
            if at/mt > 0.75:
                overloaded.append(wid)
        
        for victim in overloaded:
            for tidb, widb in self.redis.hscan_iter(self.processing_key):
                tid, owner = tidb.decode('utf-8'), widb.decode('utf-8')
                if owner == victim:
                    raw_t = self.redis.hget('tasks:data', tid)
                    if raw_t:
                        task = json.loads(raw_t)
                        self.redis.hset(self.processing_key, tid, worker_id)
                        self._update_worker_task_count(victim, -1)
                        self._update_worker_task_count(worker_id, 1)
                        logger.info(f"Task {tid} stolen from {victim} by {worker_id}")
                        return task
        return None
    
    def _update_worker_task_count(self, worker_id, delta):
        """Update worker task count"""
        raw = self.redis.hget(self.worker_stats_key, worker_id)
        if raw:
            s = json.loads(raw)
            s['active_tasks'] = max(0, s.get('active_tasks',0) + delta)
            self.redis.hset(self.worker_stats_key, worker_id, json.dumps(s))
    
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
        for widb, raw in self.redis.hgetall(self.worker_stats_key).items():
            wid = widb.decode('utf-8')
            s = json.loads(raw)
            hb = s.get('last_heartbeat', 0)
            if time.time() - hb <= 30:
                stats['active_workers'] += 1
                mt, at = s.get('max_tasks',1), s.get('active_tasks',0)
                stats['total_capacity'] += mt
                stats['used_capacity'] += at
                stats['worker_details'].append({
                    'worker_id': wid,
                    'max_tasks': mt,
                    'active_tasks': at,
                    'available_capacity': mt - at,
                    'last_heartbeat': hb,
                    'completed_tasks': s.get('completed_tasks',0)
                })
        stats['available_capacity'] = stats['total_capacity'] - stats['used_capacity']
        stats['capacity_percent'] = (
            (stats['used_capacity']/stats['total_capacity'])*100
            if stats['total_capacity'] else 0
        )
        return stats
