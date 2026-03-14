# Troubleshooting

## Docker Issues

```bash
# Reset everything
docker-compose down -v
docker-compose up -d

# Check logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart airflow-scheduler
```

## Airflow Issues

```bash
# Reset Airflow database
docker-compose exec airflow-scheduler airflow db reset

# Clear task instances
docker-compose exec airflow-scheduler airflow tasks clear data_ingestion_pipeline

# DAG not appearing - check for syntax errors
docker-compose exec airflow-scheduler python /opt/airflow/dags/data_ingestion.py

# Refresh DAGs
docker-compose restart airflow-scheduler
```

## Permission Issues

```bash
# Fix Airflow directory permissions
sudo chown -R $(id -u):$(id -g) airflow/logs airflow/plugins data/
```

## API Rate Limits

- **Alpha Vantage**: 25 calls/day, 5 calls/minute
- **News API**: 100 calls/day
- **CoinGecko**: No API key required for free tier
- Wait between calls or use demo mode

## Cloud Run Issues

```bash
# Check service status
./scripts/deploy-gcp.sh status

# View logs for a service
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=market-intel-airflow" \
  --project=n8nclyt --limit=20 --format="value(textPayload)"

# Restart a service (redeploy same image)
gcloud run services update market-intel-airflow --region=us-west1 --project=n8nclyt

# Scheduler OOM - increase memory
gcloud run services update market-intel-airflow-scheduler --memory=8Gi --region=us-west1 --project=n8nclyt
```

### Cloud SQL Connection Issues

- Verify Cloud SQL instance is RUNNABLE: `gcloud sql instances list --project=n8nclyt`
- Check `--add-cloudsql-instances` flag is set on Cloud Run services
- Connection string format: `postgresql+psycopg2://postgres:postgres@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE`

### SQLAlchemy Version Conflict

Airflow 2.8.x requires SQLAlchemy 1.4.x. Do NOT add `sqlalchemy>=2.0` to `requirements-airflow.txt` — it breaks Airflow's `create_engine()` call with `Invalid argument(s) 'encoding'`.

## Known Issues

1. **XGBoost on Mac**: Requires `brew install libomp`
2. **MLflow 3.x Security**: Downgraded to 2.9.2 due to localhost binding issues
3. **tslearn**: Removed due to Python 3.11 incompatibility
4. **Docker disk space**: Run `docker system prune -f` periodically
5. **Airflow scheduler memory**: Requires 8Gi on Cloud Run due to ML library imports in DAGs
6. **SQLAlchemy**: Do not upgrade past 1.4.x in Airflow container
