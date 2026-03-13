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

## MLflow Issues

### MLflow Container Not Starting
- Check that the `mlflow` service in docker-compose.yml uses the correct image: `mlflow/mlflow:latest`
- Verify PostgreSQL is healthy before MLflow starts
- Ensure the backend-store-uri uses `postgresql+psycopg2://` prefix

### MLflow Database Not Created
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Connect to PostgreSQL and create mlflow database
docker-compose exec postgres psql -U airflow -c "CREATE DATABASE mlflow;"
```

### MLflow Cannot Connect to Backend
- Verify the PostgreSQL connection string format
- Check that the `mlflow` service depends on `postgres` with `condition: service_healthy`
- Ensure network connectivity between containers

## DAG Failures

### data_ingestion_pipeline Failures
- Check that ALPHA_VANTAGE_API_KEY and NEWS_API_KEY are set in .env
- Verify xcom_pull task_ids match the actual task_ids in the DAG
- For demo mode, the DAG should still work without API keys

### model_training_pipeline Failures
- Ensure training data exists in data/processed/
- Check that XGBoost is installed in the Airflow container
- Verify MLflow is accessible for model logging
- The DAG now uses actual parquet files from data/processed/

### feature_engineering_pipeline Failures
- Ensure raw data exists in data/raw/ or processed data exists in data/processed/
- Check that feature engineering imports work correctly

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

## Prediction Service Issues

### Demo Mode Always Active
- Check MLFLOW_TRACKING_URI environment variable
- Ensure model is registered in MLflow Model Registry
- Verify models/production/ directory exists and contains .pkl files

### Incorrect Predictions
- The prediction service now uses technical indicator signals when no model is loaded
- Check that technical indicators (RSI, MACD, SMA) are being calculated
- Verify target_price calculation: current_price * (1 + prediction)

## Known Issues

1. **XGBoost on Mac**: Requires `brew install libomp`
2. **MLflow 3.x Security**: Downgraded to 2.9.2 due to localhost binding issues
3. **tslearn**: Removed due to Python 3.11 incompatibility
4. **Docker disk space**: Run `docker system prune -f` periodically
5. **Empty data directory**: Run the sample data generator to create training data
