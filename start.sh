#!/bin/bash

# Intelligent FAQ System Startup Script
# ====================================
# This script handles the complete initialization and startup of the FAQ system
# with proper error handling, health checks, and user experience optimization.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_PORT=8000
ARANGO_PORT=8529
MAX_WAIT_TIME=120
HEALTH_CHECK_INTERVAL=5

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required commands are available
check_dependencies() {
    log "Checking system dependencies..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
    
    success "All dependencies are available"
}

# Check if Docker is running
check_docker() {
    log "Checking Docker daemon..."
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        echo "Please start Docker and try again."
        exit 1
    fi
    
    success "Docker daemon is running"
}

# Clean up any existing containers
cleanup_existing() {
    log "Cleaning up existing containers..."
    
    # Stop and remove existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove any dangling containers
    docker container prune -f 2>/dev/null || true
    
    success "Cleanup completed"
}

# Build and start services
start_services() {
    log "Building and starting services..."
    
    # Build the application image
    log "Building FAQ application image..."
    if ! docker-compose build --no-cache; then
        error "Failed to build application image"
        exit 1
    fi
    
    # Start services in background
    log "Starting services..."
    if ! docker-compose up -d; then
        error "Failed to start services"
        exit 1
    fi
    
    success "Services started successfully"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local max_wait=$3
    
    log "Waiting for $service_name to be healthy..."
    
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        
        printf "."
        sleep $HEALTH_CHECK_INTERVAL
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done
    
    echo
    error "$service_name failed to become healthy within ${max_wait}s"
    return 1
}

# Check service logs for errors
check_service_logs() {
    local service_name=$1
    
    log "Checking $service_name logs for errors..."
    
    local error_count=$(docker-compose logs --tail=50 "$service_name" 2>/dev/null | grep -i error | wc -l)
    
    if [ "$error_count" -gt 0 ]; then
        warning "Found $error_count error(s) in $service_name logs"
        echo "Recent logs:"
        docker-compose logs --tail=10 "$service_name"
        echo
    else
        success "$service_name logs look clean"
    fi
}

# Display system status
show_status() {
    log "System Status:"
    echo
    docker-compose ps
    echo
    
    # Show resource usage
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo
}

# Open browser automatically
open_browser() {
    local url="http://localhost:$APP_PORT"
    
    log "Opening browser to $url..."
    
    # Detect OS and open browser accordingly
    if command -v xdg-open > /dev/null; then
        # Linux
        xdg-open "$url" 2>/dev/null &
    elif command -v open > /dev/null; then
        # macOS
        open "$url" 2>/dev/null &
    elif command -v start > /dev/null; then
        # Windows (Git Bash/WSL)
        start "$url" 2>/dev/null &
    else
        warning "Could not automatically open browser"
        echo "Please manually open: $url"
        return 1
    fi
    
    success "Browser opened successfully"
}

# Display helpful information
show_info() {
    echo
    echo "=============================================="
    echo "🤖 Intelligent FAQ System is Ready!"
    echo "=============================================="
    echo
    echo "📊 Application: http://localhost:$APP_PORT"
    echo "🗄️  ArangoDB:   http://localhost:$ARANGO_PORT"
    echo
    echo "🔧 Management Commands:"
    echo "  View logs:     docker-compose logs -f"
    echo "  Stop system:   docker-compose down"
    echo "  Restart:       docker-compose restart"
    echo "  Status:        docker-compose ps"
    echo
    echo "📝 API Endpoints:"
    echo "  Chat:          POST /chat"
    echo "  Health:        GET /health"
    echo "  Database Info: GET /database-info"
    echo "  Statistics:    GET /stats"
    echo "  Categories:    GET /categories"
    echo "  Feedback:      POST /feedback"
    echo
    echo "🚀 Ready to answer your questions!"
    echo
    echo "🧪 Test ArangoDB Integration:"
    echo "  python3 test_db_integration.py"
    echo
    echo "🗄️  Access ArangoDB Web Interface:"
    echo "  http://localhost:8529 (root / faq_system_2024)"
    echo "=============================================="
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Startup failed with exit code $exit_code"
    
    echo
    echo "Troubleshooting steps:"
    echo "1. Check Docker is running: docker info"
    echo "2. Check ports are available: netstat -an | grep ':8000\\|:8529'"
    echo "3. View logs: docker-compose logs"
    echo "4. Clean restart: docker-compose down && $0"
    echo
    
    # Show recent logs if containers exist
    if docker-compose ps -q | head -1 | grep -q .; then
        echo "Recent logs:"
        docker-compose logs --tail=20
    fi
    
    exit $exit_code
}

# Set up error handling
trap handle_error ERR

# Main execution
main() {
    echo
    echo "🚀 Starting Intelligent FAQ System..."
    echo "======================================"
    
    # Pre-flight checks
    check_dependencies
    check_docker
    
    # Setup
    cleanup_existing
    start_services
    
    # Health checks
    wait_for_service "ArangoDB" "http://localhost:$ARANGO_PORT/_api/version" $MAX_WAIT_TIME
    
    # Give ArangoDB a moment to fully initialize
    log "Waiting for ArangoDB to fully initialize..."
    sleep 5
    
    wait_for_service "FAQ Application" "http://localhost:$APP_PORT/health" $MAX_WAIT_TIME
    
    # Post-startup checks
    check_service_logs "arangodb"
    check_service_logs "faq-app"
    
    # Display status
    show_status
    
    # Open browser
    sleep 2  # Give services a moment to fully initialize
    open_browser
    
    # Show final information
    show_info
}

# Handle script interruption
cleanup_on_exit() {
    echo
    warning "Startup interrupted"
    echo "To clean up: docker-compose down"
    exit 130
}

trap cleanup_on_exit INT

# Run main function
main "$@" 