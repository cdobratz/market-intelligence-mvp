#!/bin/bash
# =============================================================================
# Airflow Cloud Run Entrypoint
# Handles DB initialization, user creation, and service startup
#
# MODE environment variable controls which service to run:
#   - webserver (default): Runs Airflow webserver on port 8080
#   - scheduler: Runs Airflow scheduler
#   - combined: Runs both webserver and scheduler (for single-container deploys)
# =============================================================================

set -e

# Wait for database to be ready
wait_for_db() {
    echo "[entrypoint] Waiting for database connection..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if python -c "
import sqlalchemy
engine = sqlalchemy.create_engine('${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN}')
engine.connect().close()
print('Database is ready')
" 2>/dev/null; then
            return 0
        fi
        retries=$((retries - 1))
        echo "[entrypoint] Database not ready, retrying in 2s... ($retries attempts left)"
        sleep 2
    done
    echo "[entrypoint] ERROR: Could not connect to database after 60s"
    exit 1
}

# Initialize the database
init_db() {
    echo "[entrypoint] Running database migrations..."
    airflow db migrate

    echo "[entrypoint] Creating admin user (if not exists)..."
    airflow users create \
        --username "${AIRFLOW_WWW_USER_USERNAME:-admin}" \
        --password "${AIRFLOW_WWW_USER_PASSWORD:-admin}" \
        --firstname "Admin" \
        --lastname "User" \
        --role "Admin" \
        --email "admin@example.com" 2>/dev/null || true
}

MODE="${AIRFLOW_MODE:-webserver}"

echo "[entrypoint] Starting Airflow in ${MODE} mode..."

# Always wait for DB and run migrations
wait_for_db
init_db

case "$MODE" in
    webserver)
        echo "[entrypoint] Starting Airflow webserver on port 8080..."
        exec airflow webserver --port 8080
        ;;
    scheduler)
        echo "[entrypoint] Starting Airflow scheduler in background..."
        airflow scheduler &
        SCHEDULER_PID=$!

        # Cloud Run requires a listening port - start a simple health check server
        echo "[entrypoint] Starting health check server on port 8080..."
        python -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import os, signal

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'scheduler running')
    def log_message(self, format, *args):
        pass  # Suppress request logs

HTTPServer(('0.0.0.0', 8080), HealthHandler).serve_forever()
" &
        HEALTH_PID=$!

        # Wait for scheduler to exit
        wait $SCHEDULER_PID
        EXIT_CODE=$?
        kill $HEALTH_PID 2>/dev/null || true
        exit $EXIT_CODE
        ;;
    combined)
        echo "[entrypoint] Starting Airflow scheduler in background..."
        airflow scheduler &
        SCHEDULER_PID=$!

        echo "[entrypoint] Starting Airflow webserver on port 8080..."
        airflow webserver --port 8080 &
        WEBSERVER_PID=$!

        # Wait for either process to exit
        wait -n $SCHEDULER_PID $WEBSERVER_PID
        EXIT_CODE=$?
        echo "[entrypoint] A process exited with code $EXIT_CODE, shutting down..."
        kill $SCHEDULER_PID $WEBSERVER_PID 2>/dev/null || true
        exit $EXIT_CODE
        ;;
    *)
        echo "[entrypoint] Unknown mode: $MODE"
        echo "Valid modes: webserver, scheduler, combined"
        exit 1
        ;;
esac
