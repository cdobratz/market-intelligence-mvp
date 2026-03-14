# MLOps Workflow

## Experiment Tracking

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

**View experiments**:
- Local: http://localhost:5000
- Cloud Run: https://market-intel-mlflow-1001565765695.us-west1.run.app

## Model Registry

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

## Model Serving

The FastAPI service provides real-time stock predictions:

```bash
# Start FastAPI server (or use Docker)
uvicorn src.api.main:app --reload

# Test prediction endpoint (local)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Test prediction endpoint (Cloud Run)
curl -X POST "https://market-intel-api-1001565765695.us-west1.run.app/predict" \
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
