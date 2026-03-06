# 📊 Financial Market Intelligence Platform 

> End-to-end machine learning pipeline for financial market analysis and prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8+-017CEE.svg)](https://airflow.apache.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5+-E25A1C.svg)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project demonstrates a production-ready machine learning pipeline for financial market intelligence, showcasing:

- **End-to-End ML Pipeline**: From data ingestion to model deployment
- **Multiple ML Techniques**: Supervised learning, unsupervised learning, deep learning, and ensemble methods
- **Performance Optimization**: Comparative analysis of Pandas vs Fireducks for data processing
- **MLOps Best Practices**: Workflow orchestration, experiment tracking, model versioning, and monitoring
- **Scalable Architecture**: Designed to handle both small and large-scale data processing

### Key Features

✅ **Data Ingestion**: Automated collection from multiple financial APIs  
✅ **Feature Engineering**: Technical indicators, time-series features, sentiment analysis  
✅ **ML Models**: Price prediction, anomaly detection, asset clustering  
✅ **Performance Benchmarking**: Pandas vs Fireducks vs Spark comparison  
✅ **MLOps Pipeline**: Automated training, evaluation, and deployment  
✅ **Model Tracking**: MLflow integration for experiment management  
✅ **API Serving**: FastAPI endpoints for model predictions  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Financial Data APIs                         │
│   Alpha Vantage │ CoinGecko │ News API │ Economic Indicators    │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Apache Airflow (Orchestration)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Data Ingest  │→ │  Feature Eng │→ │Model Training│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │  Pandas  │    │ Fireducks│    │  Spark   │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Models                                  │
│  Supervised: XGBoost, RF, LSTM │ Unsupervised: K-Means, IF     │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow + Model Registry                       │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Model Serving                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **uv**: Modern Python package manager
- **Git**: For version control

### Demo Mode (No API Keys Required)

The platform includes a **demo mode** that works without API keys, using synthetic market data for demonstrations.

```bash
# Start all services with Docker
docker compose up -d

# Access the demo dashboard
open http://localhost:8000
```

**Demo Features:**
- Enter any stock symbol (AAPL, GOOGL, MSFT, etc.)
- Get instant price predictions with confidence scores
- View technical indicators (RSI, MACD, SMA)
- See pipeline status and model information

> **Note**: Demo mode uses synthetic data. For real market predictions, add your API keys (see below).

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/market-intelligence-mvp.git
cd market-intelligence-mvp
```

2. **Install uv (if not already installed)**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. **Create virtual environment and install dependencies**
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

4. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required API Keys** (all free):
- **Alpha Vantage**: [Get API Key](https://www.alphavantage.co/support/#api-key)
- **News API**: [Get API Key](https://newsapi.org/register)
- **CoinGecko**: No API key required for basic tier

5. **Start Docker services**
```bash
# Start all services (Airflow, MLflow, Postgres, Redis, Jupyter)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

6. **Access the services**

| Service | URL | Credentials |
|---------|-----|-------------|
| **Live Dashboard** | https://market-intel-api-1001565765695.us-west1.run.app/ | No authentication |
| Airflow Web UI | http://localhost:8080 | Username: `airflow`<br>Password: `airflow` |
| MLflow UI | http://localhost:5000 | No authentication |
| Jupyter Lab | http://localhost:8888 | Token: `jupyter` |
| API Documentation | http://localhost:8000/docs | (when FastAPI is running) |

---

## 📂 Project Structure

```
market-intelligence-mvp/
├── airflow/
│   ├── dags/                      # Airflow DAG definitions
│   │   ├── data_ingestion.py     # Data collection pipeline
│   │   ├── feature_engineering.py # Feature creation pipeline
│   │   └── model_training.py      # Model training pipeline
│   ├── logs/                      # Airflow execution logs
│   ├── plugins/                   # Custom Airflow plugins
│   └── config/                    # Airflow configuration
├── src/
│   ├── data/                      # Data processing modules
│   │   ├── ingestion.py          # API clients and data fetching
│   │   ├── processing_pandas.py  # Pandas processing
│   │   ├── processing_fireducks.py # Fireducks processing
│   │   └── processing_spark.py   # Spark processing
│   ├── features/                  # Feature engineering
│   ├── models/                    # ML model implementations
│   │   ├── supervised/
│   │   ├── unsupervised/
│   │   └── ensemble/
│   ├── evaluation/                # Model evaluation utilities
│   ├── api/                       # FastAPI application
│   └── utils/                     # Common utilities
├── data/
│   ├── raw/                       # Raw data from APIs
│   ├── processed/                 # Processed data
│   └── features/                  # Feature store
├── models/                        # Trained model artifacts
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
├── benchmarks/                    # Performance benchmarks
├── docker-compose.yml             # Docker services definition
├── pyproject.toml                 # Project dependencies (uv)
├── warp.md                        # Project roadmap and goals
└── README.md                      # This file
```

---

## 🔄 Running the Pipelines

### 1. Data Ingestion Pipeline

The data ingestion DAG fetches data from multiple sources daily at 2 AM UTC.

**Manual Trigger**:
```bash
# Via Airflow UI: Navigate to DAGs → data_ingestion_pipeline → Trigger DAG

# Via CLI:
docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline
```

**What it does**:
- Fetches stock data from Alpha Vantage
- Fetches forex data
- Fetches cryptocurrency data from CoinGecko
- Fetches financial news from News API
- Validates and stores data in Parquet format

### 2. Feature Engineering Pipeline

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

### 3. Model Training Pipeline

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

---

## 📊 Data Processing: Pandas vs Fireducks

One of the key features of this project is the performance comparison between Pandas and Fireducks.

### Why Fireducks?

Fireducks is a drop-in replacement for Pandas that provides:
- **Faster execution**: 30-50% speed improvement for many operations
- **Lower memory usage**: More efficient memory management
- **Same API**: Pandas-compatible interface
- **Parallel processing**: Automatic parallelization

### Running Benchmarks

```bash
# Run comprehensive benchmarks
python benchmarks/pandas_vs_fireducks.py

# Results will be saved to benchmarks/results/
```

### Expected Performance Gains

| Dataset Size | Pandas Time | Fireducks Time | Speedup |
|--------------|-------------|----------------|---------|
| 100K rows    | 2.3s        | 1.5s          | 1.5x    |
| 1M rows      | 23.1s       | 14.2s         | 1.6x    |
| 10M rows     | 245s        | 152s          | 1.6x    |

---

## 🤖 Machine Learning Models

### Supervised Learning

| Model | Use Case | Key Metrics |
|-------|----------|-------------|
| **XGBoost** | Price prediction | RMSE, MAE, R² |
| **Random Forest** | Direction classification | Accuracy, F1 |
| **LightGBM** | Fast training | RMSE, Training time |
| **LSTM** | Sequence prediction | RMSE, Directional accuracy |

### Unsupervised Learning

| Model | Use Case | Key Metrics |
|-------|----------|-------------|
| **Isolation Forest** | Anomaly detection | Precision, Recall |
| **K-Means** | Asset clustering | Silhouette score |
| **DBSCAN** | Outlier detection | Cluster quality |
| **Autoencoder** | Pattern recognition | Reconstruction error |

### Ensemble Methods

- **Stacking**: Combines multiple base models
- **Weighted Averaging**: Performance-based model combination
- **Meta-learner**: XGBoost meta-model on base predictions

---

## 📈 MLOps Workflow

### Experiment Tracking

All experiments are logged to MLflow:

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("rmse", 0.05)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

**View experiments**: http://localhost:5000

### Model Registry

Champion models are registered in the MLflow Model Registry:

```python
# Register model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="market-predictor"
)

# Transition to production
client.transition_model_version_stage(
    name="market-predictor",
    version=1,
    stage="Production"
)
```

### Model Serving

The FastAPI service provides real-time stock predictions:

```bash
# Start FastAPI server (or use Docker)
uvicorn src.api.main:app --reload

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Response:
# {
#   "symbol": "AAPL",
#   "prediction": 0.0234,
#   "direction": "bullish",
#   "confidence": 0.75,
#   "current_price": 175.50,
#   "target_price": 179.62,
#   "timestamp": "2026-03-05T12:00:00"
# }
```

**API Endpoints:**
- `GET /` - Demo dashboard (web UI)
- `GET /health` - Health check
- `POST /predict` - Single stock prediction
- `POST /predict/batch` - Batch predictions
- `GET /pipeline/status` - Pipeline status
- `GET /models` - Available models
- `GET /docs` - OpenAPI documentation

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_ingestion.py

# Run integration tests
pytest tests/integration/
```

---

## 📝 Development Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes and test**
```bash
# Run tests
pytest

# Run linting
ruff check .

# Format code
black .
```

3. **Commit and push**
```bash
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name
```

4. **Create Pull Request**

---

## 🐛 Troubleshooting

### Docker Issues

```bash
# Reset everything
docker-compose down -v
docker-compose up -d

# Check logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart airflow-scheduler
```

### Airflow Issues

```bash
# Reset Airflow database
docker-compose exec airflow-scheduler airflow db reset

# Clear task instances
docker-compose exec airflow-scheduler airflow tasks clear data_ingestion_pipeline
```

### Permission Issues

```bash
# Fix Airflow directory permissions
sudo chown -R $(id -u):$(id -g) airflow/logs airflow/plugins
```

---

## 📚 Documentation

- [Project Roadmap](warp.md) - Detailed implementation plan
- [API Documentation](http://localhost:8000/docs) - When FastAPI is running
- [Architecture Diagrams](docs/architecture.md)
- [Model Cards](docs/model_cards/)

---

## 🎓 Learning Resources

- [Apache Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Fireducks Documentation](https://fireducks-dev.github.io/)
- [Time Series Forecasting with Python](https://machinelearningmastery.com/time-series-forecasting/)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Alpha Vantage** for stock market data
- **CoinGecko** for cryptocurrency data
- **News API** for financial news
- **Apache Airflow** for workflow orchestration
- **MLflow** for experiment tracking
- **Fireducks** for optimized data processing

---

## 📧 Contact

**Cassidy** - [GitHub Profile](https://github.com/yourusername)

Project Link: [https://github.com/yourusername/market-intelligence-mvp](https://github.com/yourusername/market-intelligence-mvp)

---

## 🗺️ Roadmap

- [x] Project setup and infrastructure
- [x] Data ingestion pipeline
- [x] Feature engineering pipeline
- [x] Model training pipeline
- [ ] Model serving API
- [ ] Real-time streaming pipeline
- [ ] Dashboard with Streamlit
- [ ] Production deployment on AWS/Azure
- [ ] Comprehensive documentation
- [ ] Demo video and presentation

---

**Built with ❤️ for demonstrating end-to-end ML engineering skills**
