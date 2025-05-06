import os
import json
import time
import uuid
import socket
import logging
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MASTER_API_URL = os.environ.get('MASTER_API_URL', 'http://localhost:5000/api')
HEARTBEAT_INTERVAL = int(os.environ.get('HEARTBEAT_INTERVAL', 30))
TASK_REQUEST_INTERVAL = float(os.environ.get('TASK_REQUEST_INTERVAL', 1.0))  # seconds

# Worker concurrency settings
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', 2))  # Default to 2 concurrent tasks

# Generate a unique worker ID
WORKER_ID = os.environ.get('WORKER_ID', f"worker-{uuid.uuid4()}")
HOSTNAME = socket.gethostname()
try:
    IP_ADDRESS = socket.gethostbyname(HOSTNAME)
except:
    IP_ADDRESS = "127.0.0.1"  # Fallback to localhost if hostname resolution fails

# Worker metadata
WORKER_METADATA = {
    'python_version': os.environ.get('PYTHON_VERSION', '3.x'),
    'startup_time': datetime.now().isoformat(),
    'max_concurrent_tasks': MAX_CONCURRENT_TASKS
}

# Task tracking
active_tasks = []
active_tasks_lock = threading.Lock()

def register_worker():
    """Register the worker with the master node"""
    try:
        response = requests.post(
            f"{MASTER_API_URL}/workers",
            json={
                'worker_id': WORKER_ID,
                'hostname': HOSTNAME,
                'ip_address': IP_ADDRESS,
                'metadata': WORKER_METADATA
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Worker registered successfully: {WORKER_ID}")
            return True
        else:
            logger.error(f"Failed to register worker: {response.status_code} - {response.text}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"Error registering worker: {e}")
        return False

def send_heartbeat():
    """Send heartbeat to master node"""
    while True:
        try:
            # Include current active tasks count in heartbeat
            with active_tasks_lock:
                current_active_tasks = len(active_tasks)
                available_capacity = MAX_CONCURRENT_TASKS - current_active_tasks
            
            response = requests.post(
                f"{MASTER_API_URL}/workers/{WORKER_ID}/heartbeat",
                json={
                    'active_tasks': current_active_tasks,
                    'max_tasks': MAX_CONCURRENT_TASKS,
                    'available_capacity': available_capacity
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Heartbeat sent successfully: {WORKER_ID}")
            else:
                logger.warning(f"Failed to send heartbeat: {response.status_code} - {response.text}")
        
        except requests.RequestException as e:
            logger.error(f"Error sending heartbeat: {e}")
        
        time.sleep(HEARTBEAT_INTERVAL)

def submit_result(task_id, result_data, processing_time):
    """Submit task result to master node"""
    try:
        response = requests.post(
            f"{MASTER_API_URL}/tasks/submit",
            json={
                'task_id': task_id,
                'worker_id': WORKER_ID,
                'result_data': result_data,
                'processing_time': processing_time
            },
            timeout=10
        )
        
        if response.status_code == 201:
            logger.info(f"Result submitted successfully for task {task_id}")
            return True
        else:
            logger.error(f"Failed to submit result: {response.status_code} - {response.text}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"Error submitting result: {e}")
        return False

def process_array_sum(array):
    """Process an array sum task"""
    if not array or not isinstance(array, list):
        return {"error": "Invalid array"}
    
    try:
        # Validate that all elements are numbers
        for i, val in enumerate(array):
            if not isinstance(val, (int, float)):
                return {"error": f"Element at index {i} is not a number: {val}"}
        
        result = sum(array)
        return {"sum": result}
    except Exception as e:
        return {"error": str(e)}

def process_task(task):
    """Process a task based on its type"""
    task_id = task['id']
    
    try:
        logger.info(f"Processing task: {task_id}")
        start_time = time.time()
        
        task_data = task['task_data']
        operation = task_data.get('operation', 'unknown')
        
        # Simulate work with sleep
        time.sleep(10)
        
        if operation == 'sum':
            result = process_array_sum(task_data.get('array', []))
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        processing_time = time.time() - start_time
        
        logger.info(f"Task {task_id} completed in {processing_time:.4f} seconds")
        
        # Submit result to master node
        submit_result(task_id, result, processing_time)
    
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
    
    finally:
        # Remove task from active tasks
        with active_tasks_lock:
            if task_id in active_tasks:
                active_tasks.remove(task_id)
            logger.info(f"Completed task {task_id}. Active tasks: {len(active_tasks)}/{MAX_CONCURRENT_TASKS}")

def can_accept_task():
    """Check if worker can accept more tasks"""
    with active_tasks_lock:
        return len(active_tasks) < MAX_CONCURRENT_TASKS

def request_task():
    """Request a task from the master node"""
    try:
        response = requests.get(
            f"{MASTER_API_URL}/tasks/request",
            params={
                'worker_id': WORKER_ID
            },
            timeout=10
        )
        
        if response.status_code == 200:
            task = response.json()
            logger.info(f"Received task: {task['id']}")
            return task
        elif response.status_code == 204:
            # No tasks available
            return None
        else:
            logger.warning(f"Failed to request task: {response.status_code} - {response.text}")
            return None
    
    except requests.RequestException as e:
        logger.error(f"Error requesting task: {e}")
        return None

def task_loop():
    """Main task processing loop"""
    while True:
        # Check if we can accept more tasks
        if not can_accept_task():
            # Wait before checking again
            time.sleep(TASK_REQUEST_INTERVAL)
            continue
        
        # Request a task
        task = request_task()
        
        if task:
            # Add to active tasks
            with active_tasks_lock:
                task_id = task['id']
                active_tasks.append(task_id)
                logger.info(f"Starting task {task_id}. Active tasks: {len(active_tasks)}/{MAX_CONCURRENT_TASKS}")
            
            # Process task in a separate thread
            threading.Thread(target=process_task, args=(task,), daemon=True).start()
        else:
            # No tasks available, wait before polling again
            time.sleep(TASK_REQUEST_INTERVAL)

def main():
    """Main worker function"""
    logger.info(f"Starting worker: {WORKER_ID}")
    logger.info(f"Hostname: {HOSTNAME}")
    logger.info(f"IP Address: {IP_ADDRESS}")
    logger.info(f"Maximum concurrent tasks: {MAX_CONCURRENT_TASKS}")
    
    # Register worker with master node
    if not register_worker():
        logger.error("Failed to register worker. Retrying in 10 seconds...")
        time.sleep(10)
        if not register_worker():
            logger.error("Failed to register worker again. Exiting.")
            return
    
    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
    heartbeat_thread.start()
    
    # Start task processing loop
    try:
        task_loop()
    except KeyboardInterrupt:
        logger.info("Worker shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()