#!/bin/bash
# =============================================================================
# Google Cloud Run Deployment Script
# Financial Market Intelligence Platform
#
# Usage:
#   ./scripts/deploy-gcp.sh [command]
#
# Commands:
#   deploy    - Deploy all services
#   start     - Start all Cloud Run services
#   stop      - Stop all Cloud Run services (scale to zero)
#   status    - Check status of all services
#   destroy   - Delete all GCP resources
# =============================================================================

set -e

# Configuration - UPDATE THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
REPO="market-intelligence"

# Service names
AIRFLOW_SERVICE="market-intel-airflow"
MLFLOW_SERVICE="market-intel-mlflow"
API_SERVICE="market-intel-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is configured
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not installed. Please install Google Cloud SDK."
        exit 1
    fi

    if [ "$PROJECT_ID" = "your-project-id" ]; then
        log_error "Please set GCP_PROJECT_ID environment variable or update PROJECT_ID in this script."
        exit 1
    fi

    gcloud config set project "$PROJECT_ID"
    log_info "Using project: $PROJECT_ID"
}

# Enable required APIs (excluding Cloud SQL which requires special permissions)
enable_apis() {
    log_info "Enabling required GCP APIs..."
    gcloud services enable \
        run.googleapis.com \
        storage.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        --quiet
}

# Create Artifact Registry
create_artifact_registry() {
    log_info "Creating Artifact Registry repository..."
    gcloud artifacts repositories create "$REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Market Intelligence containers" \
        --quiet 2>/dev/null || log_warn "Repository already exists"
}

# Build and push containers
build_and_push() {
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

    # Note: Airflow not deployed to Cloud Run - use local Docker
    log_warn "Skipping Airflow build (use local Docker for Airflow)"
    log_warn "Skipping MLflow (use local Docker or deploy later)"

    # Build API container only
    log_info "Building and pushing API container..."
    docker build -f docker/Dockerfile.api -t "${REGISTRY}/api:latest" .
    docker push "${REGISTRY}/api:latest"
}

# Deploy services to Cloud Run
deploy_services() {
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

    # Note: Airflow not deployed to Cloud Run (use local Docker)
    
    log_info "Deploying API to Cloud Run..."
    gcloud run deploy "$API_SERVICE" \
        --image="${REGISTRY}/api:latest" \
        --region="$REGION" \
        --platform=managed \
        --memory=2Gi \
        --cpu=2 \
        --port=8080 \
        --min-instances=0 \
        --max-instances=10 \
        --timeout=600 \
        --set-env-vars="DEMO_MODE=true" \
        --allow-unauthenticated \
        --quiet

    log_info "Deployment complete!"
    show_urls
}

# Start services (set min-instances to 1)
start_services() {
    log_info "Starting all Cloud Run services..."

    # Note: Airflow not deployed to Cloud Run (use local Docker)
    log_info "Starting MLflow..."
    gcloud run services update "$MLFLOW_SERVICE" \
        --region="$REGION" \
        --min-instances=1 \
        --quiet

    log_info "Starting API..."
    gcloud run services update "$API_SERVICE" \
        --region="$REGION" \
        --min-instances=1 \
        --quiet

    log_info "All services started!"
    show_urls
}

# Stop services (scale to zero)
stop_services() {
    log_info "Stopping all Cloud Run services (scaling to zero)..."

    # Note: Airflow not deployed to Cloud Run (use local Docker)
    log_info "Stopping MLflow..."
    gcloud run services update "$MLFLOW_SERVICE" \
        --region="$REGION" \
        --min-instances=0 \
        --quiet

    log_info "Stopping API..."
    gcloud run services update "$API_SERVICE" \
        --region="$REGION" \
        --min-instances=0 \
        --quiet

    log_info "All services stopped (will scale to zero when idle)."
    log_info "Cost savings: Services only incur charges when handling requests."
}

# Show service URLs
show_urls() {
    echo ""
    log_info "Service URLs:"
    echo "  API:     $(gcloud run services describe $API_SERVICE --region=$REGION --format='value(status.url)' 2>/dev/null || echo 'Not deployed')"
    echo ""
    log_info "Note: Airflow and MLflow are not deployed to Cloud Run."
    log_info "Use local Docker for those: docker compose up -d airflow mlflow"
}

# Check status of services
check_status() {
    log_info "Checking Cloud Run service status..."
    echo ""

    for service in "$API_SERVICE"; do
        status=$(gcloud run services describe "$service" \
            --region="$REGION" \
            --format='value(status.conditions[0].status)' 2>/dev/null || echo "Not deployed")

        min_instances=$(gcloud run services describe "$service" \
            --region="$REGION" \
            --format='value(spec.template.metadata.annotations["autoscaling.knative.dev/minScale"])' 2>/dev/null || echo "0")

        echo "  $service:"
        echo "    Status: $status"
        echo "    Min Instances: $min_instances"
        echo ""
    done

    echo "  Note: Airflow and MLflow use local Docker"
    echo ""

    show_urls
}

# Destroy all resources
destroy_resources() {
    log_warn "This will delete all GCP resources. Are you sure? (yes/no)"
    read -r confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Aborted."
        exit 0
    fi

    log_info "Deleting Cloud Run services..."
    gcloud run services delete "$API_SERVICE" --region="$REGION" --quiet 2>/dev/null || true
    # Note: Airflow and MLflow not deployed to Cloud Run

    log_info "Deleting GCS bucket..."
    gsutil rm -r "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || true

    log_info "Deleting Artifact Registry repository..."
    gcloud artifacts repositories delete "$REPO" --location="$REGION" --quiet 2>/dev/null || true

    log_info "All resources deleted."
}

# Create GCS bucket (optional - only needed for MLflow)
create_mlflow_bucket() {
    log_info "Creating GCS bucket for MLflow artifacts (optional)..."
    gsutil mb -l "$REGION" "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || log_warn "Bucket may already exist or not needed"
}

# Full deployment
full_deploy() {
    check_gcloud
    enable_apis
    create_artifact_registry
    create_mlflow_bucket
    build_and_push
    deploy_services
    show_urls
    log_info "Deployment complete!"
    log_info "TIP: Use './scripts/deploy-gcp.sh stop' to scale to zero when not in use."
}

# Main command handler
case "${1:-deploy}" in
    deploy)
        full_deploy
        ;;
    start)
        check_gcloud
        start_services
        ;;
    stop)
        check_gcloud
        stop_services
        ;;
    status)
        check_gcloud
        check_status
        ;;
    destroy)
        check_gcloud
        destroy_resources
        ;;
    urls)
        check_gcloud
        show_urls
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|status|destroy|urls}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Build containers and deploy all services"
        echo "  start   - Start all services (min-instances=1)"
        echo "  stop    - Stop all services (scale to zero for cost savings)"
        echo "  status  - Check status of all services"
        echo "  destroy - Delete all GCP resources"
        echo "  urls    - Show service URLs"
        exit 1
        ;;
esac
