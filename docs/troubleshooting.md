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

## Known Issues

1. **XGBoost on Mac**: Requires `brew install libomp`
2. **MLflow 3.x Security**: Downgraded to 2.9.2 due to localhost binding issues
3. **tslearn**: Removed due to Python 3.11 incompatibility
4. **Docker disk space**: Run `docker system prune -f` periodically
