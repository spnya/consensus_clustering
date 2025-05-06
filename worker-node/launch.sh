#!/bin/bash

# This script helps to set up and launch a worker node on a remote server

# Function to show usage
show_usage() {
    echo "Usage: $0 -m MASTER_HOST [-i WORKER_ID] [-k KAFKA_PORT] [-a API_PORT] [-h]"
    echo "  -m MASTER_HOST  The hostname or IP address of the master node"
    echo "  -i WORKER_ID    Optional: Custom worker ID (default: auto-generated)"
    echo "  -k KAFKA_PORT   Optional: Kafka port (default: 9092)"
    echo "  -a API_PORT     Optional: API port (default: 5000)"
    echo "  -h              Show this help message"
    exit 1
}

# Parse command line arguments
MASTER_HOST=""
WORKER_ID=""
KAFKA_PORT="9092"
API_PORT="5000"

while getopts "m:i:k:a:h" opt; do
    case ${opt} in
        m )
            MASTER_HOST=$OPTARG
            ;;
        i )
            WORKER_ID=$OPTARG
            ;;
        k )
            KAFKA_PORT=$OPTARG
            ;;
        a )
            API_PORT=$OPTARG
            ;;
        h )
            show_usage
            ;;
        \? )
            show_usage
            ;;
    esac
done

# Check if master host is provided
if [ -z "$MASTER_HOST" ]; then
    echo "Error: Master host is required."
    show_usage
fi

# Generate Worker ID if not provided
if [ -z "$WORKER_ID" ]; then
    WORKER_ID="worker-$(hostname)-$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)"
fi

# Create .env file
cat > .env << EOF
# Kafka Settings
KAFKA_BOOTSTRAP_SERVERS=${MASTER_HOST}:${KAFKA_PORT}
KAFKA_TOPIC=clustering-tasks
KAFKA_GROUP_ID=clustering-workers

# Master Node API
MASTER_API_URL=http://${MASTER_HOST}:${API_PORT}/api

# Worker Settings
WORKER_ID=${WORKER_ID}
HEARTBEAT_INTERVAL=30
EOF

echo "Worker configuration created. Worker ID: ${WORKER_ID}"
echo "Master node: ${MASTER_HOST}:${API_PORT}"
echo "Kafka: ${MASTER_HOST}:${KAFKA_PORT}"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker is installed. Starting worker with Docker..."
    docker-compose up -d
    echo "Worker started. Check logs with: docker-compose logs -f"
else
    echo "Docker not found. Starting worker directly with Python..."
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        echo "Installing Python requirements..."
        pip3 install -r requirements.txt
        
        echo "Starting worker..."
        python3 worker.py
    else
        echo "Error: Neither Docker nor Python is available. Cannot start worker."
        exit 1
    fi
fi