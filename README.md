# Financial Market Intelligence Platform

> End-to-end machine learning pipeline for financial market analysis and prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8+-017CEE.svg)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Live Demo

**[Try the live dashboard](https://market-intel-api-1001565765695.us-west1.run.app/)** no setup required. Enter any stock symbol (AAPL, GOOGL, MSFT) to get instant price predictions with confidence scores, technical indicators, and pipeline status.

## What It Does

- **Data Ingestion** -- Automated collection from Alpha Vantage, CoinGecko, and News API
- **Feature Engineering** -- 50+ technical indicators, time-series features, sentiment analysis
- **ML Models** -- Price prediction, anomaly detection, asset clustering
- **Performance Benchmarking** -- Pandas vs Fireducks comparison
- **MLOps Pipeline** -- Automated training, evaluation, and deployment via Airflow + MLflow
- **API Serving** -- FastAPI endpoints for real-time predictions

## Architecture

```
Financial Data APIs (Alpha Vantage, CoinGecko, News API)
              |
              v
Apache Airflow (Orchestration)
  Data Ingest -> Feature Engineering -> Model Training
              |
              v
Data Processing (Pandas / Fireducks / Spark)
              |
              v
ML Models (XGBoost, RF, LSTM | K-Means, Isolation Forest)
              |
              v
MLflow (Experiment Tracking + Model Registry)
              |
              v
FastAPI (Model Serving)
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/cdobratz/market-intelligence-mvp.git
cd market-intelligence-mvp
uv venv && source .venv/bin/activate && uv pip install -e .

# 2. Configure API keys
cp .env.example .env   # then edit with your keys

# 3. Start services
docker-compose up -d
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions including API key registration and service verification.

### Demo Mode

Works without API keys using synthetic market data:

```bash
docker compose up -d
open http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo dashboard (web UI) |
| `/health` | GET | Health check |
| `/predict` | POST | Single stock prediction |
| `/predict/batch` | POST | Batch predictions |
| `/pipeline/status` | GET | Pipeline status |
| `/models` | GET | Available models |
| `/docs` | GET | OpenAPI documentation |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.11+ |
| Orchestration | Apache Airflow 2.8+ |
| Data Processing | Pandas, Fireducks, PySpark |
| ML/DL | Scikit-learn, XGBoost, LightGBM, TensorFlow |
| MLOps | MLflow 2.10+ |
| API | FastAPI |
| Infrastructure | Docker, Docker Compose, Google Cloud Run |
| Package Manager | uv |

## Services

### Cloud Run (Production)

| Service | URL | Credentials |
|---------|-----|-------------|
| **Live Dashboard** | [market-intel-api...run.app](https://market-intel-api-1001565765695.us-west1.run.app/) | None |
| Airflow | [market-intel-airflow...run.app](https://market-intel-airflow-1001565765695.us-west1.run.app/) | admin / admin |
| MLflow | [market-intel-mlflow...run.app](https://market-intel-mlflow-1001565765695.us-west1.run.app/) | None |

### Local Development (Docker Compose)

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | None |
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | None |
| Jupyter | http://localhost:8888 | Token: `jupyter` |

## Project Structure

```
market-intelligence-mvp/
├── airflow/dags/            # Airflow DAG definitions
├── src/
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   ├── models/              # ML model implementations
│   │   ├── supervised/
│   │   ├── unsupervised/
│   │   └── ensemble/
│   ├── evaluation/          # Model evaluation utilities
│   ├── api/                 # FastAPI application
│   └── utils/               # Common utilities
├── data/                    # Raw, processed, and feature data
├── models/                  # Trained model artifacts
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit and integration tests
├── benchmarks/              # Performance benchmarks
├── docs/                    # Documentation
├── docker-compose.yml
├── pyproject.toml
└── QUICKSTART.md
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start Guide](QUICKSTART.md) | Detailed setup instructions |
| [Pipelines](docs/pipelines.md) | Running the data/ML pipelines |
| [ML Models](docs/models.md) | Model details and metrics |
| [Benchmarks](docs/benchmarks.md) | Pandas vs Fireducks performance |
| [MLOps Workflow](docs/mlops.md) | Experiment tracking and model serving |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Contributing](docs/contributing.md) | Development workflow |
| [Roadmap](docs/roadmap.md) | Implementation plan and progress |

## Cloud Run Deployment

All services are deployed to Google Cloud Run with Cloud SQL (PostgreSQL 15) as the shared database.

```bash
# Deploy all services
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-west1
./scripts/deploy-gcp.sh deploy

# Check status
./scripts/deploy-gcp.sh status

# Scale to zero (cost savings)
./scripts/deploy-gcp.sh stop
```

**Architecture**: API + MLflow + Airflow webserver + Airflow scheduler on Cloud Run, PostgreSQL on Cloud SQL, artifacts on GCS.

## Roadmap

- [x] Project setup and infrastructure
- [x] Data ingestion pipeline
- [x] Feature engineering pipeline
- [x] Model training pipeline (in progress)
- [ ] Real-time streaming pipeline
- [ ] Production monitoring dashboard
- [ ] Comprehensive model suite

See [docs/roadmap.md](docs/roadmap.md) for the full implementation plan.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Cassidy** - [GitHub](https://github.com/cdobratz)

Project Link: [Live Dashboard](https://market-intel-api-1001565765695.us-west1.run.app/)
