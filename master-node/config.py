import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))

# PostgreSQL configuration
POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'consensus')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'zX9cV8bN7mA6sD5f')
POSTGRES_DB = os.environ.get('POSTGRES_DB', 'consensus_clustering')

# Worker configuration
HEARTBEAT_INTERVAL = int(os.environ.get('HEARTBEAT_INTERVAL', '30'))
TASK_REQUEST_INTERVAL = float(os.environ.get('TASK_REQUEST_INTERVAL', '1.0'))
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', '2'))

def get_database_url():
    """Get database URL from environment variables"""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"