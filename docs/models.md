# ML Models

## Supervised Learning

| Model | Use Case | Key Metrics |
|-------|----------|-------------|
| **XGBoost** | Price prediction | RMSE, MAE, R² |
| **Random Forest** | Direction classification | Accuracy, F1 |
| **LightGBM** | Fast training | RMSE, Training time |
| **LSTM** | Sequence prediction | RMSE, Directional accuracy |

## Unsupervised Learning

| Model | Use Case | Key Metrics |
|-------|----------|-------------|
| **Isolation Forest** | Anomaly detection | Precision, Recall |
| **K-Means** | Asset clustering | Silhouette score |
| **DBSCAN** | Outlier detection | Cluster quality |
| **Autoencoder** | Pattern recognition | Reconstruction error |

## Ensemble Methods

- **Stacking**: Combines multiple base models
- **Weighted Averaging**: Performance-based model combination
- **Meta-learner**: XGBoost meta-model on base predictions
