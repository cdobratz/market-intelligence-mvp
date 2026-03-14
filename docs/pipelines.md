# Running the Pipelines

## 1. Data Ingestion Pipeline

The data ingestion DAG fetches data from multiple sources daily at 2 AM UTC.

**Manual Trigger**:
```bash
# Via Airflow UI: Navigate to DAGs > data_ingestion_pipeline > Trigger DAG
# Local: http://localhost:8080
# Cloud Run: https://market-intel-airflow-1001565765695.us-west1.run.app

# Via CLI (local):
docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline

# Via API (Cloud Run):
curl -s -u admin:admin -X POST \
  "https://market-intel-airflow-1001565765695.us-west1.run.app/api/v1/dags/data_ingestion_pipeline/dagRuns" \
  -H "Content-Type: application/json" -d '{}'
```

**What it does**:
- Fetches stock data from Alpha Vantage
- Fetches forex data
- Fetches cryptocurrency data from CoinGecko
- Fetches financial news from News API
- Validates and stores data in Parquet format

---

## 2. Feature Engineering Pipeline

Runs daily at 4 AM UTC (after data ingestion).

**Manual Trigger**:
```bash
docker-compose exec airflow-scheduler airflow dags trigger feature_engineering_pipeline
```

**What it does**:
- Loads raw data
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Creates time-series features
- Processes news sentiment
- Compares Pandas vs Fireducks performance
- Validates and stores features

---

## 3. Model Training Pipeline

Runs weekly on Sundays at 6 AM UTC.

**Manual Trigger**:
```bash
docker-compose exec airflow-scheduler airflow dags trigger model_training_pipeline
```

**What it does**:
- Loads engineered features
- Trains multiple models (XGBoost, Random Forest, LSTM, etc.)
- Performs unsupervised learning (clustering, anomaly detection)
- Creates ensemble models
- Evaluates all models
- Registers champion model in MLflow
- Generates model cards
