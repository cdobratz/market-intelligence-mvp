#!/bin/bash
# =============================================================================
# GCP Cloud Scheduler Setup for Cost Optimization
# Automatically start/stop services during business hours
#
# Usage:
#   ./scripts/gcp-scheduler.sh setup    - Create scheduled jobs
#   ./scripts/gcp-scheduler.sh remove   - Remove scheduled jobs
# =============================================================================

set -e

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"

# Service names
AIRFLOW_SERVICE="market-intel-airflow"
AIRFLOW_SCHEDULER_SERVICE="market-intel-airflow-scheduler"
MLFLOW_SERVICE="market-intel-mlflow"
API_SERVICE="market-intel-api"

# Schedule (cron format)
# Start at 6 AM UTC (Mon-Fri)
START_SCHEDULE="0 6 * * 1-5"
# Stop at 10 PM UTC (every day)
STOP_SCHEDULE="0 22 * * *"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

setup_scheduler() {
    log_info "Setting up Cloud Scheduler for cost optimization..."

    # Enable Cloud Scheduler API
    gcloud services enable cloudscheduler.googleapis.com --quiet

    # Get service account for Cloud Run invoker
    SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"

    # Create start jobs for each service
    for service in "$AIRFLOW_SERVICE" "$AIRFLOW_SCHEDULER_SERVICE" "$MLFLOW_SERVICE" "$API_SERVICE"; do
        SERVICE_URL=$(gcloud run services describe "$service" --region="$REGION" --format='value(status.url)')

        log_info "Creating start scheduler for $service..."
        gcloud scheduler jobs create http "start-${service}" \
            --location="$REGION" \
            --schedule="$START_SCHEDULE" \
            --uri="https://${REGION}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${PROJECT_ID}/services/${service}" \
            --http-method=PATCH \
            --headers="Content-Type=application/json" \
            --message-body='{"spec":{"template":{"metadata":{"annotations":{"autoscaling.knative.dev/minScale":"1"}}}}}' \
            --oauth-service-account-email="$SA_EMAIL" \
            --quiet 2>/dev/null || log_warn "Job start-${service} already exists"

        log_info "Creating stop scheduler for $service..."
        gcloud scheduler jobs create http "stop-${service}" \
            --location="$REGION" \
            --schedule="$STOP_SCHEDULE" \
            --uri="https://${REGION}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${PROJECT_ID}/services/${service}" \
            --http-method=PATCH \
            --headers="Content-Type=application/json" \
            --message-body='{"spec":{"template":{"metadata":{"annotations":{"autoscaling.knative.dev/minScale":"0"}}}}}' \
            --oauth-service-account-email="$SA_EMAIL" \
            --quiet 2>/dev/null || log_warn "Job stop-${service} already exists"
    done

    log_info "Scheduler setup complete!"
    log_info "Services will:"
    echo "  - Start at 6 AM UTC (Mon-Fri)"
    echo "  - Stop at 10 PM UTC (every day)"
    echo ""
    echo "Estimated cost savings: 50-70% reduction in Cloud Run costs"
}

remove_scheduler() {
    log_info "Removing Cloud Scheduler jobs..."

    for service in "$AIRFLOW_SERVICE" "$AIRFLOW_SCHEDULER_SERVICE" "$MLFLOW_SERVICE" "$API_SERVICE"; do
        gcloud scheduler jobs delete "start-${service}" --location="$REGION" --quiet 2>/dev/null || true
        gcloud scheduler jobs delete "stop-${service}" --location="$REGION" --quiet 2>/dev/null || true
    done

    log_info "Scheduler jobs removed."
}

list_jobs() {
    log_info "Current scheduler jobs:"
    gcloud scheduler jobs list --location="$REGION" 2>/dev/null || echo "No jobs found"
}

case "${1:-setup}" in
    setup)
        setup_scheduler
        ;;
    remove)
        remove_scheduler
        ;;
    list)
        list_jobs
        ;;
    *)
        echo "Usage: $0 {setup|remove|list}"
        exit 1
        ;;
esac
