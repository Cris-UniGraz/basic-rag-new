#!/bin/bash

# Production deployment script for RAG API
# This script handles production deployment with health checks, rollbacks, and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/../docker-compose.production.yml"
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-latest}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker and docker-compose are installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log_error ".env file not found. Please copy from .env.production.example and configure."
        exit 1
    fi
    
    # Check if required directories exist
    mkdir -p "${PROJECT_ROOT}/../logs" "${PROJECT_ROOT}/../data" "${PROJECT_ROOT}/backups"
    
    log_success "Prerequisites check passed"
}

# Validate environment configuration
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Use Python to validate environment
    python3 -c "
import sys
sys.path.append('${PROJECT_ROOT}')
try:
    from app.core.environment_manager import check_deployment_readiness, get_environment_summary
    
    print('Environment Summary:')
    print(get_environment_summary())
    print()
    
    readiness = check_deployment_readiness()
    if not readiness['ready']:
        print('❌ Environment validation failed!')
        for issue in readiness['issues']:
            print(f'  - {issue}')
        sys.exit(1)
    else:
        print('✅ Environment validation passed!')
        
except Exception as e:
    print(f'❌ Environment validation error: {e}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Environment validation passed"
    else
        log_error "Environment validation failed"
        exit 1
    fi
}

# Build images
build_images() {
    log_info "Building production images..."
    
    export BUILD_DATE VERSION VCS_REF ENVIRONMENT
    
    docker-compose -f "$COMPOSE_FILE" build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg ENVIRONMENT="$ENVIRONMENT"
    
    log_success "Images built successfully"
}

# Health check function
wait_for_health() {
    local service_name=$1
    local max_attempts=${2:-30}
    local attempt=1
    
    log_info "Waiting for $service_name to be healthy..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service_name" | grep -q "healthy"; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not healthy yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service_name failed to become healthy within timeout"
    return 1
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    # Start infrastructure services first
    log_info "Starting infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d etcd minio mongodb redis
    
    # Wait for infrastructure to be healthy
    wait_for_health "etcd" 15
    wait_for_health "minio" 15
    wait_for_health "mongodb" 20
    wait_for_health "redis" 10
    
    # Start Milvus
    log_info "Starting Milvus..."
    docker-compose -f "$COMPOSE_FILE" up -d milvus
    wait_for_health "milvus" 30
    
    # Start application services
    log_info "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" up -d backend
    wait_for_health "backend" 30
    
    docker-compose -f "$COMPOSE_FILE" up -d frontend
    wait_for_health "frontend" 20
    
    log_success "All services deployed successfully"
}

# Start monitoring (optional)
deploy_monitoring() {
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        log_info "Deploying monitoring stack..."
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d
        
        wait_for_health "prometheus" 15
        wait_for_health "grafana" 15
        
        log_success "Monitoring stack deployed"
    fi
}

# Start reverse proxy (optional)
deploy_proxy() {
    if [[ "${ENABLE_PROXY:-false}" == "true" ]]; then
        log_info "Deploying reverse proxy..."
        docker-compose -f "$COMPOSE_FILE" --profile proxy up -d
        
        wait_for_health "nginx" 10
        
        log_success "Reverse proxy deployed"
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test backend health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Backend health check passed"
    else
        log_error "Backend health check failed"
        return 1
    fi
    
    # Test frontend
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        log_success "Frontend health check passed"
    else
        log_error "Frontend health check failed"
        return 1
    fi
    
    # Test database connectivity (through backend)
    if curl -s http://localhost:8000/health/detailed | grep -q '"status":"healthy"'; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    log_success "All smoke tests passed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    log_info "Service URLs:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - Frontend: http://localhost:8501"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo "  - Health Check: http://localhost:8000/health"
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
    fi
    
    if [[ "${ENABLE_PROXY:-false}" == "true" ]]; then
        echo "  - Nginx Proxy: http://localhost"
    fi
    
    echo
    log_success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from backup if needed
    if [[ -d "${PROJECT_ROOT}/backups/last_good_deployment" ]]; then
        log_info "Restoring from last good deployment backup..."
        # Add restoration logic here
    fi
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    docker system prune -f
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local command=${1:-deploy}
    
    case $command in
        deploy)
            log_info "Starting production deployment..."
            check_prerequisites
            validate_environment
            build_images
            deploy_services
            deploy_monitoring
            deploy_proxy
            run_smoke_tests
            show_status
            ;;
        rollback)
            rollback
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        health)
            run_smoke_tests
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|cleanup|health}"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy the application to production"
            echo "  rollback - Rollback to previous deployment"
            echo "  status   - Show current deployment status"
            echo "  cleanup  - Clean up unused Docker resources"
            echo "  health   - Run health checks"
            exit 1
            ;;
    esac
}

# Trap for cleanup on script exit
trap 'echo "Deployment script interrupted"' INT TERM

# Run main function
main "$@"