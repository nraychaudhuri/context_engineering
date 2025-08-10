#!/bin/bash

# Deployment script for attention_recitation.py agent
# Usage: ./deploy.sh [build|run|stop|logs|clean]

set -e

IMAGE_NAME="attention-recitation-agent"
CONTAINER_NAME="attention-recitation-agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOF
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_api_key_here
EXA_API_KEY=your_exa_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
EOF
        print_error "Please edit .env file with your API keys before running the agent"
        exit 1
    fi
}

# Build the Docker image
build() {
    print_status "Building Docker image..."
    docker build -t $IMAGE_NAME .
    print_status "Build completed successfully!"
}

# Run the container
run() {
    check_env_file

    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    print_status "Starting the attention recitation agent..."
    docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        $IMAGE_NAME

    print_status "Agent started! Use './deploy.sh logs' to view output"
}

# Stop the container
stop() {
    print_status "Stopping the agent..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    print_status "Agent stopped!"
}

# View logs
logs() {
    print_status "Showing agent logs (Ctrl+C to exit)..."
    docker logs -f $CONTAINER_NAME
}

# Clean up
clean() {
    print_status "Cleaning up containers and images..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME 2>/dev/null || true
    docker system prune -f
    print_status "Cleanup completed!"
}

# Show usage
usage() {
    echo "Usage: $0 [build|run|stop|logs|clean|help]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Build and run the agent"
    echo "  stop        Stop the running agent"
    echo "  logs        Show agent logs"
    echo "  clean       Clean up containers and images"
    echo "  help        Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker installed"
    echo "  - .env file with API keys configured"
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    run)
        build
        run
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    help|*)
        usage
        ;;
esac
