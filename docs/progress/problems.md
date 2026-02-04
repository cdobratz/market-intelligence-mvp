# Problems, Warnings, and Errors Log

## Date: 2025-11-21

### 1. MLflow Service Not Accessible on localhost:5000

**Problem:**
- MLflow service was running but not accessible from host machine at localhost:5000
- Port mapping was correct (5000:5000) but service was unreachable

**Root Cause:**
- MLflow 3.6.0 has security middleware that enforces localhost-only binding by default
- Despite `--host 0.0.0.0` flag, service bound to `127.0.0.1:5000` inside container
- Security middleware warning: "Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins."

**Warnings in Logs:**
```
[MLflow] Security middleware enabled with default settings (localhost-only)
INFO: Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
FutureWarning: Filesystem tracking backend (e.g., './mlruns') is deprecated
FutureWarning: Filesystem model registry backend (e.g., './mlruns') is deprecated
WARNING: Running pip as the 'root' user can result in broken permissions
```

**Solution:**
- Downgraded to MLflow 2.9.2 which doesn't have strict security middleware
- Changed image from `ghcr.io/mlflow/mlflow:v2.18.0` to `python:3.11-slim`
- Used pip to install `mlflow==2.9.2 psycopg2-binary boto3`
- Service now binds correctly to 0.0.0.0:5000

**Files Modified:**
- `docker-compose.yml` (mlflow service configuration)

---

### 2. PostgreSQL Database Out of Disk Space

**Problem:**
- PostgreSQL container in restart loop
- All services depending on Postgres (Airflow scheduler, MLflow) failing to start

**Error Messages:**
```
FATAL: could not write lock file "postmaster.pid": No space left on device
PostgreSQL Database directory appears to contain a database; Skipping initialization
```

**Root Cause:**
- Docker build cache accumulated 25.4GB of unused data
- System had 246GB available, but Docker internal disk allocation was exhausted
- Prevented Postgres from writing lock files and starting properly

**Solution:**
- Ran `docker system prune -f` to remove unused containers, networks, and images
- Ran `docker volume prune -f` to remove unused volumes
- Reclaimed 25.99GB from images/cache and 2MB from volumes
- Restarted Postgres service successfully

**Impact:**
- Postgres status changed from "Restarting (1)" to "Up (healthy)"
- Enabled Airflow scheduler to start successfully

---

### 3. Airflow Scheduler Unable to Connect to Database

**Problem:**
- Airflow scheduler not starting
- Logs showing database connection errors
- No scheduler activity visible

**Error Messages:**
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not translate host name "postgres" to address: Name or service not known
```

**Root Cause:**
- PostgreSQL service was unhealthy/not running due to disk space issue (Problem #2)
- Scheduler couldn't resolve "postgres" hostname because service was down
- Dependency check prevented scheduler from starting

**Warnings:**
```
cannot record scheduled_duration for task fetch_data.* because previous state change time has not been saved
Marked 1 SchedulerJob instances as failed (after restart)
```

**Solution:**
- Fixed PostgreSQL disk space issue first
- Recreated airflow-scheduler container with `docker-compose up -d airflow-scheduler`
- Service started successfully after Postgres became healthy

**Current Status:**
- Scheduler running and healthy
- Processing DAGs successfully (data_ingestion_pipeline executed with state=success)
- Health check returning 200

---

### 4. Official MLflow Docker Image Missing psycopg2

**Problem:**
- Attempted to use official MLflow image `ghcr.io/mlflow/mlflow:v2.18.0`
- Container immediately crashed with ModuleNotFoundError

**Error Messages:**
```
ModuleNotFoundError: No module named 'psycopg2'
ERROR mlflow.cli: Error initializing backend store
ERROR mlflow.cli: No module named 'psycopg2'
```

**Root Cause:**
- Official MLflow Docker image doesn't include PostgreSQL driver (psycopg2)
- Backend store URI configured for PostgreSQL: `postgresql://airflow:airflow@postgres/mlflow`
- Image assumes SQLite or requires custom installation of database drivers

**Attempted Solutions:**
1. Added command to install psycopg2-binary before starting server - failed (command not executed properly)
2. Tried modifying entrypoint with shell script - failed

**Final Solution:**
- Reverted to `python:3.11-slim` base image
- Install dependencies explicitly in bash command: `pip install --no-cache-dir mlflow==2.9.2 psycopg2-binary boto3`
- Gives full control over package versions and dependencies

---

## Summary of Current Service Status

| Service | Status | Port | Notes |
|---------|--------|------|-------|
| Airflow Webserver | ✅ Healthy | 8080 | Accessible at localhost:8080 |
| Airflow Scheduler | ✅ Healthy | - | Logs accessible, processing DAGs |
| MLflow | ✅ Running | 5000 | Accessible at localhost:5000 |
| Jupyter Notebook | ✅ Healthy | 8888 | Accessible at localhost:8888 |
| PostgreSQL | ✅ Healthy | 5432 | Database for Airflow and MLflow |
| Redis | ✅ Healthy | 6379 | Cache and message broker |

---

## Recommendations for Future

1. **Docker Maintenance:**
   - Set up periodic `docker system prune` to prevent disk issues
   - Monitor Docker disk usage with `docker system df`
   - Consider setting Docker Desktop disk limit appropriately

2. **MLflow Version:**
   - Current version (2.9.2) is stable but older
   - If upgrading to newer versions (3.x), need to configure security middleware properly:
     - Use `--app-name` flag
     - Configure `--allowed-hosts` and `--cors-allowed-origins`
     - Or use official Docker image with custom Dockerfile for dependencies

3. **Package Management:**
   - Project uses `uv` package manager, but Docker containers still use `pip`
   - Consider standardizing on `uv` in Docker containers for consistency
   - Would require installing uv in containers and updating command syntax

4. **Monitoring:**
   - Set up health check monitoring for all services
   - Add alerting for disk space thresholds
   - Consider logging aggregation for easier debugging

5. **Documentation:**
   - Keep this problems.md updated with new issues
   - Document all configuration changes in comments
   - Maintain service dependency map
