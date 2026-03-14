#!/bin/bash
# =============================================================================
# Google Cloud Run Deployment Script
# Financial Market Intelligence Platform
#
# Usage:
#   ./scripts/deploy-gcp.sh [command]
#
# Commands:
#   deploy    - Deploy all services (API, MLflow, Airflow webserver + scheduler)
#   start     - Start all Cloud Run services
#   stop      - Stop all Cloud Run services (scale to zero)
#   status    - Check status of all services
#   destroy   - Delete all GCP resources
#   urls      - Show service URLs
#   setup-db  - Create Cloud SQL instance and databases
# =============================================================================

set -e

# Enable BuildKit for cross-platform builds
export DOCKER_BUILDKIT=1

# Configuration - UPDATE THESE VALUES or set env vars
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-west1}"
REPO="market-intelligence"

# Service names
AIRFLOW_WEB_SERVICE="market-intel-airflow"
AIRFLOW_SCHEDULER_SERVICE="market-intel-airflow-scheduler"
MLFLOW_SERVICE="market-intel-mlflow"
API_SERVICE="market-intel-api"

# Cloud SQL
CLOUD_SQL_INSTANCE="market-intel-db"
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

# All services list
ALL_SERVICES=("$API_SERVICE" "$MLFLOW_SERVICE" "$AIRFLOW_WEB_SERVICE" "$AIRFLOW_SCHEDULER_SERVICE")

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

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."
    gcloud services enable \
        run.googleapis.com \
        sqladmin.googleapis.com \
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

# Setup Cloud SQL instance
setup_cloud_sql() {
    log_info "Setting up Cloud SQL instance..."

    # Check if instance already exists
    if gcloud sql instances describe "$CLOUD_SQL_INSTANCE" --quiet 2>/dev/null; then
        log_warn "Cloud SQL instance $CLOUD_SQL_INSTANCE already exists"
    else
        log_info "Creating Cloud SQL instance (this may take a few minutes)..."
        gcloud sql instances create "$CLOUD_SQL_INSTANCE" \
            --database-version=POSTGRES_15 \
            --tier=db-f1-micro \
            --region="$REGION" \
            --storage-size=10GB \
            --quiet

        # Set root password
        gcloud sql users set-password postgres \
            --instance="$CLOUD_SQL_INSTANCE" \
            --password="postgres" \
            --quiet
    fi

    # Create databases
    log_info "Creating databases..."
    gcloud sql databases create airflow --instance="$CLOUD_SQL_INSTANCE" --quiet 2>/dev/null || log_warn "Database 'airflow' already exists"
    gcloud sql databases create mlflow --instance="$CLOUD_SQL_INSTANCE" --quiet 2>/dev/null || log_warn "Database 'mlflow' already exists"

    log_info "Cloud SQL setup complete: $CLOUD_SQL_CONNECTION"
}

# Build and push all containers
build_and_push() {
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

    log_info "Building and pushing all containers..."

    # Build Airflow container (used by both webserver and scheduler)
    log_info "Building Airflow container..."
    docker build --platform linux/amd64 -f docker/Dockerfile.airflow -t "${REGISTRY}/airflow:latest" .
    docker push "${REGISTRY}/airflow:latest"

    # Build MLflow container
    log_info "Building MLflow container..."
    docker build --platform linux/amd64 -f docker/Dockerfile.mlflow -t "${REGISTRY}/mlflow:latest" .
    docker push "${REGISTRY}/mlflow:latest"

    # Build API container
    log_info "Building API container..."
    docker build --platform linux/amd64 -f docker/Dockerfile.api -t "${REGISTRY}/api:latest" .
    docker push "${REGISTRY}/api:latest"

    log_info "All containers built and pushed."
}

# Deploy all services to Cloud Run
deploy_services() {
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

    # Common Airflow env vars
    AIRFLOW_ENV_VARS="AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://postgres:postgres@/airflow?host=/cloudsql/${CLOUD_SQL_CONNECTION}"
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},AIRFLOW__CORE__EXECUTOR=LocalExecutor"
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},AIRFLOW__CORE__LOAD_EXAMPLES=False"
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},AIRFLOW__WEBSERVER__SECRET_KEY=market-intel-secret-key-change-me"
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},AIRFLOW_WWW_USER_USERNAME=admin"
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},AIRFLOW_WWW_USER_PASSWORD=admin"

    # Get MLflow URL (deployed first so Airflow can reference it)
    log_info "Deploying MLflow to Cloud Run..."
    gcloud run deploy "$MLFLOW_SERVICE" \
        --image="${REGISTRY}/mlflow:latest" \
        --region="$REGION" \
        --platform=managed \
        --memory=1Gi \
        --cpu=1 \
        --min-instances=0 \
        --max-instances=1 \
        --set-env-vars="MLFLOW_BACKEND_URI=postgresql+psycopg2://postgres:postgres@/mlflow?host=/cloudsql/${CLOUD_SQL_CONNECTION},MLFLOW_ARTIFACT_ROOT=gs://${PROJECT_ID}-ml-artifacts" \
        --add-cloudsql-instances="${CLOUD_SQL_CONNECTION}" \
        --allow-unauthenticated \
        --quiet

    MLFLOW_URL=$(gcloud run services describe "$MLFLOW_SERVICE" --region="$REGION" --format='value(status.url)')
    log_info "MLflow deployed at: $MLFLOW_URL"

    # Add MLflow URL to Airflow env vars
    AIRFLOW_ENV_VARS="${AIRFLOW_ENV_VARS},MLFLOW_TRACKING_URI=${MLFLOW_URL}"

    # Deploy Airflow webserver
    log_info "Deploying Airflow webserver to Cloud Run..."
    gcloud run deploy "$AIRFLOW_WEB_SERVICE" \
        --image="${REGISTRY}/airflow:latest" \
        --region="$REGION" \
        --platform=managed \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=0 \
        --max-instances=1 \
        --timeout=3600 \
        --set-env-vars="AIRFLOW_MODE=webserver,${AIRFLOW_ENV_VARS}" \
        --add-cloudsql-instances="${CLOUD_SQL_CONNECTION}" \
        --allow-unauthenticated \
        --quiet

    # Deploy Airflow scheduler (needs to run continuously)
    log_info "Deploying Airflow scheduler to Cloud Run..."
    gcloud run deploy "$AIRFLOW_SCHEDULER_SERVICE" \
        --image="${REGISTRY}/airflow:latest" \
        --region="$REGION" \
        --platform=managed \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=1 \
        --max-instances=1 \
        --timeout=3600 \
        --no-cpu-throttling \
        --set-env-vars="AIRFLOW_MODE=scheduler,${AIRFLOW_ENV_VARS}" \
        --add-cloudsql-instances="${CLOUD_SQL_CONNECTION}" \
        --no-allow-unauthenticated \
        --quiet

    # Deploy API
    log_info "Deploying API to Cloud Run..."
    gcloud run deploy "$API_SERVICE" \
        --image="${REGISTRY}/api:latest" \
        --region="$REGION" \
        --platform=managed \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=0 \
        --max-instances=10 \
        --timeout=300 \
        --set-env-vars="DEMO_MODE=true,MLFLOW_TRACKING_URI=${MLFLOW_URL}" \
        --allow-unauthenticated \
        --quiet

    log_info "All services deployed!"
    show_urls
}

# Start services (set min-instances to 1)
start_services() {
    log_info "Starting all Cloud Run services..."

    for service in "${ALL_SERVICES[@]}"; do
        log_info "Starting $service..."
        gcloud run services update "$service" \
            --region="$REGION" \
            --min-instances=1 \
            --quiet 2>/dev/null || log_warn "$service not found"
    done

    log_info "All services started!"
    show_urls
}

# Stop services (scale to zero)
stop_services() {
    log_info "Stopping all Cloud Run services (scaling to zero)..."

    for service in "${ALL_SERVICES[@]}"; do
        log_info "Stopping $service..."
        gcloud run services update "$service" \
            --region="$REGION" \
            --min-instances=0 \
            --quiet 2>/dev/null || log_warn "$service not found"
    done

    log_info "All services stopped (will scale to zero when idle)."
    log_info "Cost savings: Services only incur charges when handling requests."
}

# Show service URLs
show_urls() {
    echo ""
    log_info "Service URLs:"
    for service in "$API_SERVICE" "$MLFLOW_SERVICE" "$AIRFLOW_WEB_SERVICE"; do
        url=$(gcloud run services describe "$service" --region="$REGION" --format='value(status.url)' 2>/dev/null || echo "Not deployed")
        echo "  $service: $url"
    done
    echo "  $AIRFLOW_SCHEDULER_SERVICE: (no public URL - internal service)"
    echo ""
}

# Check status of services
check_status() {
    log_info "Checking Cloud Run service status..."
    echo ""

    for service in "${ALL_SERVICES[@]}"; do
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

    # Check Cloud SQL
    echo "  Cloud SQL ($CLOUD_SQL_INSTANCE):"
    sql_status=$(gcloud sql instances describe "$CLOUD_SQL_INSTANCE" --format='value(state)' 2>/dev/null || echo "Not created")
    echo "    Status: $sql_status"
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
    for service in "${ALL_SERVICES[@]}"; do
        gcloud run services delete "$service" --region="$REGION" --quiet 2>/dev/null || true
    done

    log_info "Deleting Cloud SQL instance..."
    gcloud sql instances delete "$CLOUD_SQL_INSTANCE" --quiet 2>/dev/null || true

    log_info "Deleting GCS bucket..."
    gsutil rm -r "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || true

    log_info "Deleting Artifact Registry repository..."
    gcloud artifacts repositories delete "$REPO" --location="$REGION" --quiet 2>/dev/null || true

    log_info "All resources deleted."
}

# Create GCS bucket for MLflow artifacts
create_mlflow_bucket() {
    log_info "Creating GCS bucket for MLflow artifacts..."
    gsutil mb -l "$REGION" "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || log_warn "Bucket may already exist"
}

# Full deployment
full_deploy() {
    check_gcloud
    enable_apis
    create_artifact_registry
    setup_cloud_sql
    create_mlflow_bucket
    build_and_push
    deploy_services
    echo ""
    log_info "=== Deployment complete! ==="
    log_info "All services deployed to Cloud Run with Cloud SQL backend."
    log_info "TIP: Use './scripts/deploy-gcp.sh stop' to scale to zero when not in use."
}

# Main command handler
case "${1:-deploy}" in
    deploy)
        full_deploy
        ;;
    setup-db)
        check_gcloud
        setup_cloud_sql
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
        echo "Usage: $0 {deploy|setup-db|start|stop|status|destroy|urls}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Build containers and deploy all services to Cloud Run"
        echo "  setup-db - Create Cloud SQL instance and databases only"
        echo "  start    - Start all services (min-instances=1)"
        echo "  stop     - Stop all services (scale to zero for cost savings)"
        echo "  status   - Check status of all services and Cloud SQL"
        echo "  destroy  - Delete all GCP resources"
        echo "  urls     - Show service URLs"
        exit 1
        ;;
esac
