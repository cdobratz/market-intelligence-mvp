# CLAUDE.md - AI Assistant Guidelines & Optimization Recommendations

## Project Overview

Financial Market Intelligence Platform MVP - an end-to-end ML pipeline for market analysis and prediction using Apache Airflow, MLflow, and multiple ML frameworks.

**Current Status**: Phase 3 (ML Model Development) - 15% complete

---

## Quick Commands

```bash
# Start all services
docker-compose up -d

# Run tests
pytest tests/ -v --cov=src

# Run benchmarks
python benchmarks/pandas_vs_fireducks.py

# Trigger Airflow DAGs
docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline

# Access services
# Airflow: http://localhost:8080 (airflow/airflow)
# MLflow: http://localhost:5000
# Jupyter: http://localhost:8888 (token: jupyter)
```

---

## Optimization Recommendations

### 1. Code Efficiency Improvements

#### 1.1 Replace Keyword-Based Sentiment with Pre-trained Models
**Current**: `src/features/sentiment.py` uses simple keyword matching (412 lines)
**Problem**: Low accuracy, no context understanding, misses sarcasm/negation
**Recommendation**: Use FinBERT or DistilBERT-financial for sentiment

```python
# Suggested change in src/features/sentiment.py
from transformers import pipeline

class TransformerSentimentAnalyzer:
    def __init__(self):
        # FinBERT is specifically trained on financial text
        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1  # CPU, use 0 for GPU
        )

    def analyze(self, text: str) -> dict:
        result = self.classifier(text, truncation=True, max_length=512)
        return {"sentiment": result[0]["label"], "score": result[0]["score"]}
```

**Why Better**:
- 15-25% accuracy improvement on financial text
- Handles negation ("not bullish" = bearish)
- Context-aware embeddings
- Pre-trained on financial news corpus

**Cost**: Adds ~500MB model download, 10x slower per text but batching reduces overhead

---

#### 1.2 Vectorize Technical Indicator Calculations
**Current**: `src/features/technical_indicators.py` calculates indicators sequentially
**Problem**: Each indicator iterates over entire DataFrame separately
**Recommendation**: Use vectorized operations and combine calculations

```python
# Current approach (multiple passes)
df['rsi'] = calculate_rsi(df['close'])
df['macd'], df['signal'], df['hist'] = calculate_macd(df['close'])
df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])

# Optimized approach (single pass where possible)
def calculate_all_indicators_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Pre-compute common rolling windows once
    windows = {14: df['close'].rolling(14), 20: df['close'].rolling(20)}

    # RSI components
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Combine calculations sharing same windows
    df['sma_20'] = windows[20].mean()
    df['std_20'] = windows[20].std()
    df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
    df['bb_lower'] = df['sma_20'] - 2 * df['std_20']

    return df
```

**Why Better**:
- 30-40% faster on large datasets
- Reduces memory allocations
- NumPy vectorization is C-optimized

---

#### 1.3 Implement Feature Caching with Redis
**Current**: Features recalculated on every DAG run
**Problem**: Wastes compute for unchanged data
**Recommendation**: Cache computed features with TTL

```python
# Add to src/features/__init__.py
import redis
import hashlib
import pickle

class FeatureCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
        self.ttl = 86400  # 24 hours

    def get_or_compute(self, df: pd.DataFrame, feature_func: callable, **kwargs):
        # Create cache key from data hash + function name
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        cache_key = f"features:{feature_func.__name__}:{data_hash}"

        cached = self.client.get(cache_key)
        if cached:
            return pickle.loads(cached)

        result = feature_func(df, **kwargs)
        self.client.setex(cache_key, self.ttl, pickle.dumps(result))
        return result
```

**Why Better**:
- Redis already in docker-compose but unused
- Eliminates redundant computation
- Enables incremental processing

---

### 2. Google Cloud Run Deployment

#### 2.1 Container Architecture for Cloud Run
**Goal**: Deploy the full stack to Google Cloud Run with Docker containers

```
┌─────────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Cloud Run   │  │  Cloud Run   │  │  Cloud Run   │          │
│  │  (Airflow)   │  │  (MLflow)    │  │  (FastAPI)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         └────────────┬────┴──────────────────┘                   │
│                      │                                           │
│              ┌───────▼───────┐                                   │
│              │  Cloud SQL    │                                   │
│              │ (PostgreSQL)  │                                   │
│              └───────────────┘                                   │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Cloud Storage│  │  Memorystore │  │ Artifact     │          │
│  │ (Data/Models)│  │  (Redis)     │  │ Registry     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 2.2 Dockerfiles for Cloud Run

**Airflow Container** (`docker/Dockerfile.airflow`):
```dockerfile
FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install project dependencies
COPY requirements-airflow.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs and source code
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root src /opt/airflow/src

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"

# Cloud Run expects port 8080
EXPOSE 8080

# Airflow webserver for Cloud Run
CMD ["airflow", "webserver", "--port", "8080"]
```

**MLflow Container** (`docker/Dockerfile.mlflow`):
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    psycopg2-binary \
    google-cloud-storage \
    boto3

# Cloud Run expects port 8080
EXPOSE 8080

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend-store-uri", "${MLFLOW_BACKEND_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}"]
```

**FastAPI Inference Container** (`docker/Dockerfile.api`):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH="/app/src"

# Cloud Run expects port 8080
EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

#### 2.3 Cloud Run Deployment Scripts

**Deploy All Services** (`scripts/deploy-gcp.sh`):
```bash
#!/bin/bash
set -e

PROJECT_ID="your-project-id"
REGION="us-central1"
REPO="market-intelligence"

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudsql.googleapis.com \
    redis.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create $REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Market Intelligence containers"

# Build and push containers
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

# Build Airflow
docker build -f docker/Dockerfile.airflow -t ${REGISTRY}/airflow:latest .
docker push ${REGISTRY}/airflow:latest

# Build MLflow
docker build -f docker/Dockerfile.mlflow -t ${REGISTRY}/mlflow:latest .
docker push ${REGISTRY}/mlflow:latest

# Build FastAPI
docker build -f docker/Dockerfile.api -t ${REGISTRY}/api:latest .
docker push ${REGISTRY}/api:latest

# Deploy Cloud SQL (PostgreSQL)
gcloud sql instances create market-intel-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=$REGION \
    --storage-size=10GB

# Create databases
gcloud sql databases create airflow --instance=market-intel-db
gcloud sql databases create mlflow --instance=market-intel-db

# Deploy Memorystore (Redis)
gcloud redis instances create market-intel-redis \
    --size=1 \
    --region=$REGION \
    --redis-version=redis_7_0

# Create Cloud Storage bucket for artifacts
gsutil mb -l $REGION gs://${PROJECT_ID}-ml-artifacts

# Deploy Airflow to Cloud Run
gcloud run deploy airflow \
    --image=${REGISTRY}/airflow:latest \
    --region=$REGION \
    --platform=managed \
    --memory=2Gi \
    --cpu=2 \
    --timeout=3600 \
    --set-env-vars="AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://...,AIRFLOW__CORE__EXECUTOR=LocalExecutor" \
    --add-cloudsql-instances=${PROJECT_ID}:${REGION}:market-intel-db \
    --allow-unauthenticated

# Deploy MLflow to Cloud Run
gcloud run deploy mlflow \
    --image=${REGISTRY}/mlflow:latest \
    --region=$REGION \
    --platform=managed \
    --memory=1Gi \
    --cpu=1 \
    --set-env-vars="MLFLOW_BACKEND_URI=postgresql://...,MLFLOW_ARTIFACT_ROOT=gs://${PROJECT_ID}-ml-artifacts" \
    --add-cloudsql-instances=${PROJECT_ID}:${REGION}:market-intel-db \
    --allow-unauthenticated

# Deploy FastAPI to Cloud Run
gcloud run deploy api \
    --image=${REGISTRY}/api:latest \
    --region=$REGION \
    --platform=managed \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=10 \
    --allow-unauthenticated

echo "Deployment complete!"
echo "Airflow: $(gcloud run services describe airflow --region=$REGION --format='value(status.url)')"
echo "MLflow: $(gcloud run services describe mlflow --region=$REGION --format='value(status.url)')"
echo "API: $(gcloud run services describe api --region=$REGION --format='value(status.url)')"
```

---

#### 2.4 Cloud Run Configuration Files

**cloudbuild.yaml** (CI/CD with Cloud Build):
```yaml
steps:
  # Build Airflow container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.airflow', '-t', 'gcr.io/$PROJECT_ID/airflow:$COMMIT_SHA', '.']

  # Build MLflow container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.mlflow', '-t', 'gcr.io/$PROJECT_ID/mlflow:$COMMIT_SHA', '.']

  # Build API container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.api', '-t', 'gcr.io/$PROJECT_ID/api:$COMMIT_SHA', '.']

  # Push containers
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/airflow:$COMMIT_SHA']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/mlflow:$COMMIT_SHA']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/api:$COMMIT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'api'
      - '--image=gcr.io/$PROJECT_ID/api:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'

images:
  - 'gcr.io/$PROJECT_ID/airflow:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/mlflow:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/api:$COMMIT_SHA'

options:
  logging: CLOUD_LOGGING_ONLY
```

---

#### 2.5 Environment Configuration for GCP

**.env.gcp** (GCP-specific environment variables):
```bash
# Google Cloud Project
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Cloud SQL Connection
CLOUD_SQL_CONNECTION_NAME=${GCP_PROJECT_ID}:${GCP_REGION}:market-intel-db
DATABASE_URL=postgresql://airflow:password@/airflow?host=/cloudsql/${CLOUD_SQL_CONNECTION_NAME}

# MLflow
MLFLOW_TRACKING_URI=https://mlflow-xxxxx-uc.a.run.app
MLFLOW_ARTIFACT_ROOT=gs://${GCP_PROJECT_ID}-ml-artifacts

# Redis (Memorystore)
REDIS_HOST=10.0.0.3
REDIS_PORT=6379

# API Keys (store in Secret Manager)
ALPHA_VANTAGE_API_KEY=projects/${GCP_PROJECT_ID}/secrets/alpha-vantage-key/versions/latest
NEWS_API_KEY=projects/${GCP_PROJECT_ID}/secrets/news-api-key/versions/latest
```

---

#### 2.6 Cost Optimization for Cloud Run

**Estimated Monthly Costs**:
| Service | Configuration | Est. Cost |
|---------|--------------|-----------|
| Cloud Run (Airflow) | 2 vCPU, 2GB, always-on | $30-50 |
| Cloud Run (MLflow) | 1 vCPU, 1GB, always-on | $15-25 |
| Cloud Run (API) | 2 vCPU, 2GB, scale-to-zero | $5-20 |
| Cloud SQL | db-f1-micro | $10 |
| Memorystore | 1GB Redis | $35 |
| Cloud Storage | 10GB | $0.50 |
| **Total** | | **$95-140/mo** |

**Cost Reduction Tips**:
```yaml
# Use min-instances=0 for non-critical services
gcloud run deploy mlflow --min-instances=0

# Use Cloud Scheduler to stop services during off-hours
gcloud scheduler jobs create http stop-airflow \
    --schedule="0 22 * * *" \
    --uri="https://run.googleapis.com/..."

# Use committed use discounts for always-on services
# Use preemptible/spot instances for batch training jobs
```

---

### 3. Model Accuracy Improvements

#### 3.1 Add Walk-Forward Validation
**Current**: Simple train/test split with 80/20
**Problem**: Doesn't respect temporal nature of financial data
**Recommendation**: Implement expanding window walk-forward validation

```python
# Add to src/models/supervised/regression.py
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_validation(
    model: BaseRegressionModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 30  # Days
) -> dict:
    """
    Walk-forward validation that simulates real trading conditions.
    Train on expanding window, test on next N days.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    metrics_per_fold = []
    predictions = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.train(X_train, y_train)
        fold_metrics = model.evaluate(X_test, y_test, f"fold_{fold}")
        metrics_per_fold.append(fold_metrics)
        predictions.extend(model.predict(X_test))

    # Aggregate metrics across folds
    return {
        "mean_rmse": np.mean([m["rmse"] for m in metrics_per_fold]),
        "std_rmse": np.std([m["rmse"] for m in metrics_per_fold]),
        "mean_directional_accuracy": np.mean([m["directional_accuracy"] for m in metrics_per_fold]),
        "all_predictions": predictions
    }
```

**Why Better**:
- Prevents look-ahead bias
- More realistic performance estimates
- Identifies model stability over time
- Standard practice in quantitative finance

---

#### 3.2 Add Target Engineering for Better Signals
**Current**: Predicting raw returns (noisy)
**Problem**: Raw returns have low signal-to-noise ratio
**Recommendation**: Engineer better prediction targets

```python
# Add to src/data/sample_data_generator.py

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiple target variables for different prediction horizons."""

    # 1. Forward returns (various horizons)
    for horizon in [1, 5, 10, 20]:
        df[f'return_{horizon}d'] = df['close'].pct_change(horizon).shift(-horizon)

    # 2. Volatility-adjusted returns (Sharpe-like)
    df['vol_20d'] = df['close'].pct_change().rolling(20).std()
    df['risk_adj_return_5d'] = df['return_5d'] / df['vol_20d']

    # 3. Binary direction (smoothed)
    df['direction_5d'] = (df['return_5d'] > 0).astype(int)

    # 4. Triple barrier labeling (trend-following)
    df['triple_barrier'] = create_triple_barrier_labels(
        df['close'],
        take_profit=0.02,  # 2% profit target
        stop_loss=0.01,    # 1% stop loss
        max_holding=10     # Max 10 days
    )

    return df

def create_triple_barrier_labels(prices, take_profit, stop_loss, max_holding):
    """
    Labels based on which barrier is hit first:
    1 = take profit hit, -1 = stop loss hit, 0 = time expired
    """
    labels = []
    for i in range(len(prices) - max_holding):
        entry = prices.iloc[i]
        future = prices.iloc[i+1:i+max_holding+1]

        returns = (future - entry) / entry

        # Check barriers
        tp_hit = returns >= take_profit
        sl_hit = returns <= -stop_loss

        if tp_hit.any() and (not sl_hit.any() or tp_hit.idxmax() < sl_hit.idxmax()):
            labels.append(1)
        elif sl_hit.any():
            labels.append(-1)
        else:
            labels.append(0)

    return pd.Series(labels + [np.nan] * max_holding, index=prices.index)
```

**Why Better**:
- Triple barrier is industry-standard for ML trading signals
- Risk-adjusted returns reduce noise
- Multiple horizons capture different market dynamics
- Improves classification accuracy by 10-20%

---

#### 3.3 Feature Selection to Reduce Overfitting
**Current**: 56 features, no selection
**Problem**: Many features are redundant or noisy, causing overfitting
**Recommendation**: Implement feature selection pipeline

```python
# Add src/features/selection.py
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class FeatureSelector:
    def __init__(self, method: str = "mutual_info"):
        self.method = method
        self.selected_features = None
        self.importance_scores = None

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20
    ) -> list[str]:
        """Select top N features using specified method."""

        if self.method == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=42)
            self.importance_scores = dict(zip(X.columns, scores))

        elif self.method == "correlation":
            # Remove features with correlation > 0.95 to each other
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            X = X.drop(columns=to_drop)

            # Then rank by target correlation
            target_corr = X.corrwith(y).abs()
            self.importance_scores = target_corr.to_dict()

        elif self.method == "rfecv":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFECV(estimator, step=1, cv=5, scoring='neg_mean_squared_error')
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            return self.selected_features

        # Sort and select top N
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.selected_features = [f[0] for f in sorted_features[:n_features]]
        return self.selected_features
```

**Why Better**:
- Reduces overfitting on training data
- Faster training and inference
- More interpretable models
- Often improves test set performance

---

#### 3.4 Implement Proper Ensemble Methods
**Current**: Single XGBoost model (R²: 0.353)
**Problem**: Single model is unstable and prone to overfitting
**Recommendation**: Implement stacking ensemble

```python
# Add to src/models/ensemble/stacking.py
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

class StackingEnsemble:
    """
    Stacking ensemble that combines multiple base models
    with a meta-learner for improved predictions.
    """

    def __init__(self):
        self.base_models = [
            ('xgboost', XGBRegressor(n_estimators=100, max_depth=4)),
            ('lightgbm', LGBMRegressor(n_estimators=100, max_depth=4)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=6)),
            ('ridge', Ridge(alpha=1.0))
        ]

        # Use simple linear model as meta-learner to prevent overfitting
        self.meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])

        self.model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_learner,
            cv=5,  # 5-fold CV for base model predictions
            passthrough=False  # Don't include original features
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
```

**Expected Improvement**:
- R² improvement: 0.35 → 0.45-0.55
- More stable predictions across market regimes
- Reduces variance without increasing bias

---

#### 3.5 Add Market Regime Detection
**Current**: Single model for all market conditions
**Problem**: Markets behave differently in bull/bear/sideways regimes
**Recommendation**: Train separate models per regime or use regime as feature

```python
# Add src/features/regime_detection.py
from sklearn.mixture import GaussianMixture

class MarketRegimeDetector:
    """Detect market regimes using HMM or GMM."""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42
        )

    def fit(self, returns: pd.Series, volatility: pd.Series):
        """Fit regime model on returns and volatility."""
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'returns_lag1': returns.shift(1)
        }).dropna()

        self.model.fit(features)
        return self

    def predict_regime(self, returns: pd.Series, volatility: pd.Series) -> np.ndarray:
        """Predict regime for each observation."""
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'returns_lag1': returns.shift(1)
        }).dropna()

        return self.model.predict(features)

    def get_regime_labels(self) -> dict:
        """Map regime indices to interpretable labels."""
        means = self.model.means_[:, 0]  # Return means
        sorted_idx = np.argsort(means)

        return {
            sorted_idx[0]: 'bear',
            sorted_idx[1]: 'sideways',
            sorted_idx[2]: 'bull'
        }
```

**Why Better**:
- Captures non-stationarity in financial markets
- Models can specialize for different conditions
- Improves directional accuracy by 5-10%
- Provides trading signal confidence

---

### 4. Additional Quick Wins

#### 4.1 Enable XGBoost GPU Training
```python
# In src/models/supervised/xgboost_model.py
# Change tree_method for 10-50x speedup on GPU

params = {
    'tree_method': 'hist',  # Change to 'gpu_hist' if GPU available
    'device': 'cuda',       # Add this line
    # ... other params
}
```

#### 4.2 Add Data Augmentation for Small Datasets
```python
# Add noise injection and synthetic sample generation
def augment_financial_data(df: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
    """Add Gaussian noise to create augmented samples."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    noise = np.random.normal(0, noise_level, df[numeric_cols].shape)
    augmented = df.copy()
    augmented[numeric_cols] += noise * df[numeric_cols].std()
    return pd.concat([df, augmented], ignore_index=True)
```

#### 4.3 Implement Early Stopping Properly
```python
# Current XGBoost doesn't use early stopping effectively
# Add to training:
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)
```

---

## Architecture Decision Records

### ADR-001: Fireducks vs Pandas
**Decision**: Support both via factory pattern
**Rationale**: Fireducks shows 25-50% speedup but is less mature
**Trade-off**: Increased code complexity for performance gains

### ADR-002: MLflow for Experiment Tracking
**Decision**: Use MLflow over alternatives (W&B, Neptune)
**Rationale**: Open-source, self-hosted, good Airflow integration
**Trade-off**: Less polished UI than commercial alternatives

### ADR-003: Multiple ML Frameworks
**Decision**: Include XGBoost, LightGBM, CatBoost, scikit-learn
**Rationale**: Different algorithms excel on different data characteristics
**Trade-off**: Larger dependency footprint

### ADR-004: Google Cloud Run for Deployment
**Decision**: Use Cloud Run over GKE or Compute Engine
**Rationale**: Managed containers, auto-scaling, pay-per-use
**Trade-off**: Cold start latency, max 60-min request timeout

---

## File Structure Reference

```
src/
├── data/
│   ├── ingestion.py          # API data fetching
│   ├── validation.py         # Pandera schema validation
│   ├── processing.py         # Pandas/Fireducks factory
│   └── sample_data_generator.py
├── features/
│   ├── technical_indicators.py  # RSI, MACD, Bollinger, etc.
│   ├── timeseries.py            # Lag, rolling features
│   └── sentiment.py             # News sentiment
├── models/
│   ├── supervised/
│   │   ├── regression.py        # Base class
│   │   └── xgboost_model.py     # XGBoost impl
│   ├── unsupervised/            # TODO
│   └── ensemble/                # TODO
└── api/                         # FastAPI endpoints (TODO)

docker/
├── Dockerfile.airflow           # Airflow container
├── Dockerfile.mlflow            # MLflow container
└── Dockerfile.api               # FastAPI container

scripts/
├── deploy-gcp.sh                # GCP deployment script
└── local-dev.sh                 # Local development setup
```

---

## Known Issues

1. **XGBoost on Mac**: Requires `brew install libomp`
2. **MLflow 3.x Security**: Downgraded to 2.9.2 due to localhost binding issues
3. **tslearn**: Removed due to Python 3.11 incompatibility
4. **Docker disk space**: Run `docker system prune -f` periodically

---

## Priority Implementation Order

### High Priority (Do First)
1. Walk-forward validation (prevents overfitting)
2. Feature selection (reduces complexity)
3. Triple barrier labeling (better targets)

### Medium Priority
4. Stacking ensemble (improves accuracy)
5. FinBERT sentiment (replaces keyword matching)
6. Redis feature caching (improves efficiency)

### Lower Priority (Nice to Have)
7. Market regime detection
8. GPU training enablement
9. Data augmentation

---

*Last Updated: 2026-02-05*
*Generated for: Financial Market Intelligence MVP*
