"""
Model Training DAG - Train and evaluate ML models

This DAG orchestrates the model training pipeline:
- Load engineered features
- Train multiple ML models (supervised & unsupervised)
- Evaluate model performance
- Track experiments with MLflow
- Select champion model
- Register models for deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.sensors.external_task import ExternalTaskSensor
import os
import sys
import logging
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, '/opt/airflow/src')

logger = logging.getLogger(__name__)

DATA_BASE_PATH = os.getenv('DATA_PATH', '/opt/airflow/data')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODEL_SAVE_PATH = os.getenv('MODEL_PATH', '/opt/airflow/models')

# Target column for regression
TARGET_COL = 'target_return'

# Columns to exclude from features
EXCLUDE_COLS = [
    'date', 'symbol', 'target_return', 'target_direction',
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'direction_1d', 'direction_5d', 'direction_class_5d',
    'triple_barrier', 'risk_adj_return_5d', 'vol_20d',
]

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': True,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def _setup_mlflow(experiment_name='market-intelligence'):
    """Configure MLflow tracking."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}. Metrics will be logged locally.")


def _get_feature_columns(df):
    """Get feature columns, excluding targets and metadata."""
    import numpy as np
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    return feature_cols


def load_features(**context):
    """Load engineered features from feature store."""
    import pandas as pd
    from pathlib import Path
    from data.sample_data_generator import generate_training_data, create_targets

    # Try loading from feature store
    latest_features = Path(DATA_BASE_PATH) / 'features' / 'latest' / 'features.parquet'
    execution_date = context['execution_date']
    date_features = Path(DATA_BASE_PATH) / 'features' / execution_date.strftime('%Y-%m-%d') / 'features.parquet'

    df = None
    source = None
    for f in [date_features, latest_features]:
        if f.exists():
            df = pd.read_parquet(f)
            source = str(f)
            logger.info(f"Loaded features from {f}: {df.shape}")
            break

    if df is None:
        logger.warning("No feature store data found, generating synthetic training data")
        train_df, test_df = generate_training_data(
            output_dir=str(Path(DATA_BASE_PATH) / 'processed'),
            n_samples=1000,
            n_symbols=5,
        )
        df = pd.concat([train_df, test_df], ignore_index=True)
        source = 'synthetic'

    # Create target variables if not present
    if TARGET_COL not in df.columns and 'close' in df.columns:
        from data.sample_data_generator import create_targets
        df = create_targets(df, close_col='close')

    # Save combined features for downstream tasks
    output_path = Path(DATA_BASE_PATH) / 'training'
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'training_data.parquet'
    df.to_parquet(output_file)

    feature_cols = _get_feature_columns(df)

    logger.info(f"Training data: {df.shape}, {len(feature_cols)} features, source: {source}")
    return {
        'data_file': str(output_file),
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'feature_columns': feature_cols[:20],  # XCom size limit
        'source': source,
    }


def prepare_train_test_split(**context):
    """Split data into train/validation/test sets (time-series aware)."""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ti = context['task_instance']
    features_info = ti.xcom_pull(task_ids='load_features')
    df = pd.read_parquet(features_info['data_file'])

    # Clean data
    feature_cols = _get_feature_columns(df)

    # Drop rows with NaN target
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # Replace infinities
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    # Time-series aware split: 70% train, 15% val, 15% test
    if 'date' in df.columns:
        df = df.sort_values('date')
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save splits
    output_path = Path(DATA_BASE_PATH) / 'training'
    train_df.to_parquet(output_path / 'train.parquet')
    val_df.to_parquet(output_path / 'val.parquet')
    test_df.to_parquet(output_path / 'test.parquet')

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'train_file': str(output_path / 'train.parquet'),
        'val_file': str(output_path / 'val.parquet'),
        'test_file': str(output_path / 'test.parquet'),
    }

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return split_info


def train_xgboost_regressor(**context):
    """Train XGBoost regression model with MLflow tracking."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from pathlib import Path

    _setup_mlflow('market-intelligence-supervised')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    val_df = pd.read_parquet(split_info['val_file'])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    try:
        from models.supervised.xgboost_model import XGBoostRegressionModel

        model = XGBoostRegressionModel(
            hyperparameters={
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            },
            early_stopping_rounds=20,
        )

        start_time = t.time()
        model.train(X_train, y_train, X_val, y_val)
        training_time = t.time() - start_time

        metrics = model.evaluate(X_val, y_val, prefix='val')

        # Save model
        model_path = Path(MODEL_SAVE_PATH) / 'xgboost'
        model_path.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, model_path / 'model.joblib')

        # Log to MLflow
        try:
            with mlflow.start_run(run_name='xgboost_regressor'):
                mlflow.log_params(model.hyperparameters)
                mlflow.log_metrics(metrics)
                mlflow.log_metric('training_time', training_time)
                mlflow.sklearn.log_model(model.model, 'model')
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        logger.info(f"XGBoost trained: {metrics}")
        return {
            'model_type': 'xgboost_regressor',
            'metrics': metrics,
            'training_time': training_time,
            'model_path': str(model_path / 'model.joblib'),
        }

    except ImportError as e:
        logger.error(f"XGBoost import error: {e}")
        return {'model_type': 'xgboost_regressor', 'status': 'error', 'error': str(e)}


def train_random_forest(**context):
    """Train Random Forest model with MLflow tracking."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from pathlib import Path

    _setup_mlflow('market-intelligence-supervised')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    val_df = pd.read_parquet(split_info['val_file'])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    hyperparams = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42, 'n_jobs': -1}
    model = RandomForestRegressor(**hyperparams)

    start_time = t.time()
    model.fit(X_train, y_train)
    training_time = t.time() - start_time

    preds = model.predict(X_val)
    metrics = {
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, preds))),
        'val_mae': float(mean_absolute_error(y_val, preds)),
        'val_r2': float(r2_score(y_val, preds)),
    }

    # Save model
    model_path = Path(MODEL_SAVE_PATH) / 'random_forest'
    model_path.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, model_path / 'model.joblib')

    try:
        with mlflow.start_run(run_name='random_forest'):
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            mlflow.log_metric('training_time', training_time)
            mlflow.sklearn.log_model(model, 'model')
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    logger.info(f"Random Forest trained: {metrics}")
    return {
        'model_type': 'random_forest',
        'metrics': metrics,
        'training_time': training_time,
        'model_path': str(model_path / 'model.joblib'),
    }


def train_lightgbm(**context):
    """Train LightGBM model with MLflow tracking."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from pathlib import Path

    _setup_mlflow('market-intelligence-supervised')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    val_df = pd.read_parquet(split_info['val_file'])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    try:
        from lightgbm import LGBMRegressor

        hyperparams = {'n_estimators': 200, 'num_leaves': 31, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
        model = LGBMRegressor(**hyperparams)

        start_time = t.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],  # LightGBM handles early stopping via callbacks
        )
        training_time = t.time() - start_time

        preds = model.predict(X_val)
        metrics = {
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, preds))),
            'val_mae': float(mean_absolute_error(y_val, preds)),
            'val_r2': float(r2_score(y_val, preds)),
        }

        model_path = Path(MODEL_SAVE_PATH) / 'lightgbm'
        model_path.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, model_path / 'model.joblib')

        try:
            with mlflow.start_run(run_name='lightgbm'):
                mlflow.log_params(hyperparams)
                mlflow.log_metrics(metrics)
                mlflow.log_metric('training_time', training_time)
                mlflow.sklearn.log_model(model, 'model')
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        logger.info(f"LightGBM trained: {metrics}")
        return {
            'model_type': 'lightgbm',
            'metrics': metrics,
            'training_time': training_time,
            'model_path': str(model_path / 'model.joblib'),
        }

    except ImportError:
        logger.warning("LightGBM not installed, skipping")
        return {'model_type': 'lightgbm', 'status': 'skipped', 'reason': 'not_installed'}


def train_lstm(**context):
    """Train LSTM neural network (placeholder - requires TensorFlow)."""
    logger.info("LSTM training requires TensorFlow/Keras - skipping on Cloud Run")
    return {
        'model_type': 'lstm',
        'status': 'skipped',
        'reason': 'tensorflow_not_configured',
    }


def train_isolation_forest(**context):
    """Train Isolation Forest for anomaly detection."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from sklearn.ensemble import IsolationForest
    from pathlib import Path

    _setup_mlflow('market-intelligence-unsupervised')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    X_train = train_df[feature_cols].fillna(0)

    hyperparams = {'n_estimators': 100, 'contamination': 0.1, 'random_state': 42, 'n_jobs': -1}
    model = IsolationForest(**hyperparams)

    start_time = t.time()
    predictions = model.fit_predict(X_train)
    training_time = t.time() - start_time

    anomalies = (predictions == -1).sum()
    metrics = {
        'anomalies_detected': int(anomalies),
        'anomaly_ratio': float(anomalies / len(X_train)),
    }

    model_path = Path(MODEL_SAVE_PATH) / 'isolation_forest'
    model_path.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, model_path / 'model.joblib')

    try:
        with mlflow.start_run(run_name='isolation_forest'):
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            mlflow.log_metric('training_time', training_time)
            mlflow.sklearn.log_model(model, 'model')
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    logger.info(f"Isolation Forest: {anomalies} anomalies in {len(X_train)} samples")
    return {
        'model_type': 'isolation_forest',
        'metrics': metrics,
        'training_time': training_time,
        'model_path': str(model_path / 'model.joblib'),
    }


def train_kmeans(**context):
    """Train K-Means clustering."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    _setup_mlflow('market-intelligence-unsupervised')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    X_train = train_df[feature_cols].fillna(0)

    # Standardize for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    hyperparams = {'n_clusters': 5, 'max_iter': 300, 'random_state': 42, 'n_init': 10}
    model = KMeans(**hyperparams)

    start_time = t.time()
    labels = model.fit_predict(X_scaled)
    training_time = t.time() - start_time

    sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    metrics = {
        'silhouette_score': float(sil_score),
        'inertia': float(model.inertia_),
    }

    model_path = Path(MODEL_SAVE_PATH) / 'kmeans'
    model_path.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({'model': model, 'scaler': scaler}, model_path / 'model.joblib')

    try:
        with mlflow.start_run(run_name='kmeans_clustering'):
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            mlflow.log_metric('training_time', training_time)
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    logger.info(f"K-Means: silhouette={sil_score:.3f}, inertia={model.inertia_:.1f}")
    return {
        'model_type': 'kmeans',
        'metrics': metrics,
        'training_time': training_time,
        'model_path': str(model_path / 'model.joblib'),
    }


def train_autoencoder(**context):
    """Train Autoencoder for anomaly detection (placeholder)."""
    logger.info("Autoencoder training requires TensorFlow - skipping on Cloud Run")
    return {
        'model_type': 'autoencoder',
        'status': 'skipped',
        'reason': 'tensorflow_not_configured',
    }


def create_ensemble(**context):
    """Create stacking ensemble from trained supervised models."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time as t
    from pathlib import Path

    _setup_mlflow('market-intelligence-ensemble')

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    train_df = pd.read_parquet(split_info['train_file'])
    val_df = pd.read_parquet(split_info['val_file'])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    try:
        from models.ensemble.stacking import StackingEnsemble
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        ensemble = StackingEnsemble(
            use_xgboost=True,
            use_lightgbm=True,
            use_random_forest=True,
            use_ridge=True,
            cv=5,
        )

        start_time = t.time()
        ensemble.fit(X_train, y_train)
        training_time = t.time() - start_time

        preds = ensemble.predict(X_val)
        metrics = {
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, preds))),
            'val_mae': float(mean_absolute_error(y_val, preds)),
            'val_r2': float(r2_score(y_val, preds)),
        }

        model_path = Path(MODEL_SAVE_PATH) / 'ensemble'
        model_path.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(ensemble, model_path / 'model.joblib')

        try:
            with mlflow.start_run(run_name='stacking_ensemble'):
                mlflow.log_metrics(metrics)
                mlflow.log_metric('training_time', training_time)
                mlflow.log_param('n_base_models', len(ensemble.estimators_ if hasattr(ensemble, 'estimators_') else []))
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        logger.info(f"Ensemble trained: {metrics}")
        return {
            'model_type': 'stacking_ensemble',
            'metrics': metrics,
            'training_time': training_time,
            'model_path': str(model_path / 'model.joblib'),
        }

    except Exception as e:
        logger.error(f"Ensemble training error: {e}")
        return {'model_type': 'stacking_ensemble', 'status': 'error', 'error': str(e)}


def evaluate_all_models(**context):
    """Evaluate all trained models on test set."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from pathlib import Path

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']

    test_df = pd.read_parquet(split_info['test_file'])
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]

    # Collect all model results
    model_tasks = {
        'xgboost': 'supervised_models.train_xgboost',
        'random_forest': 'supervised_models.train_random_forest',
        'lightgbm': 'supervised_models.train_lightgbm',
        'lstm': 'supervised_models.train_lstm',
        'isolation_forest': 'unsupervised_models.train_isolation_forest',
        'kmeans': 'unsupervised_models.train_kmeans',
        'autoencoder': 'unsupervised_models.train_autoencoder',
        'ensemble': 'create_ensemble',
    }

    all_results = {}
    best_model = None
    best_rmse = float('inf')

    for name, task_id in model_tasks.items():
        result = ti.xcom_pull(task_ids=task_id)
        if not result or result.get('status') in ['skipped', 'error']:
            continue

        model_path = result.get('model_path')
        if model_path and Path(model_path).exists() and result.get('metrics'):
            all_results[name] = {
                'val_metrics': result['metrics'],
                'training_time': result.get('training_time', 0),
                'model_path': model_path,
            }

            # Evaluate on test set for supervised models
            if 'val_rmse' in result['metrics']:
                try:
                    import joblib
                    loaded = joblib.load(model_path)
                    model = loaded.model if hasattr(loaded, 'model') else loaded
                    if hasattr(model, 'predict'):
                        test_preds = model.predict(X_test)
                        test_metrics = {
                            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_preds))),
                            'test_mae': float(mean_absolute_error(y_test, test_preds)),
                            'test_r2': float(r2_score(y_test, test_preds)),
                        }
                        all_results[name]['test_metrics'] = test_metrics

                        if test_metrics['test_rmse'] < best_rmse:
                            best_rmse = test_metrics['test_rmse']
                            best_model = name
                except Exception as e:
                    logger.warning(f"Error evaluating {name} on test set: {e}")

    logger.info(f"Evaluated {len(all_results)} models. Best: {best_model} (RMSE: {best_rmse:.6f})")
    return {
        'total_models': len(all_results),
        'best_model': best_model,
        'best_rmse': best_rmse,
        'all_results': all_results,
    }


def select_champion_model(**context):
    """Select champion model based on evaluation metrics."""
    ti = context['task_instance']
    evaluation = ti.xcom_pull(task_ids='evaluate_models')

    if not evaluation or not evaluation.get('best_model'):
        logger.warning("No models to select from")
        return {'model_name': 'none', 'reason': 'no_models_evaluated'}

    best_name = evaluation['best_model']
    best_info = evaluation['all_results'].get(best_name, {})

    champion = {
        'model_name': best_name,
        'selected_at': str(datetime.now()),
        'val_metrics': best_info.get('val_metrics', {}),
        'test_metrics': best_info.get('test_metrics', {}),
        'model_path': best_info.get('model_path', ''),
        'reason': 'Lowest test RMSE',
    }

    logger.info(f"Champion model: {best_name}")
    return champion


def calculate_feature_importance(**context):
    """Calculate feature importance from tree-based models."""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ti = context['task_instance']
    split_info = ti.xcom_pull(task_ids='prepare_split')
    feature_cols = split_info['feature_columns']
    champion = ti.xcom_pull(task_ids='select_champion')

    if not champion or not champion.get('model_path'):
        return {'status': 'skipped'}

    try:
        import joblib
        loaded = joblib.load(champion['model_path'])
        model = loaded.model if hasattr(loaded, 'model') else loaded

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols[:len(importances)],
                'importance': importances,
            }).sort_values('importance', ascending=False)

            top_features = importance_df.head(20).to_dict('records')

            # Save importance report
            output_path = Path(MODEL_SAVE_PATH) / 'reports'
            output_path.mkdir(parents=True, exist_ok=True)
            importance_df.to_csv(output_path / 'feature_importance.csv', index=False)

            logger.info(f"Top 5 features: {importance_df.head().to_dict('records')}")
            return {'top_features': top_features, 'status': 'success'}

    except Exception as e:
        logger.warning(f"Feature importance error: {e}")

    return {'status': 'skipped', 'reason': 'no_feature_importances'}


def register_models_mlflow(**context):
    """Register champion model in MLflow Model Registry."""
    import mlflow

    ti = context['task_instance']
    champion = ti.xcom_pull(task_ids='select_champion')

    if not champion or champion.get('model_name') == 'none':
        return {'status': 'skipped', 'reason': 'no_champion'}

    _setup_mlflow('market-intelligence')

    try:
        # Log final champion run
        with mlflow.start_run(run_name=f"champion_{champion['model_name']}"):
            mlflow.log_param('model_type', champion['model_name'])
            mlflow.log_param('selected_at', champion['selected_at'])

            if champion.get('val_metrics'):
                for k, v in champion['val_metrics'].items():
                    mlflow.log_metric(f'champion_{k}', v)

            if champion.get('test_metrics'):
                for k, v in champion['test_metrics'].items():
                    mlflow.log_metric(f'champion_{k}', v)

            # Try to register the model
            if champion.get('model_path'):
                import joblib
                loaded = joblib.load(champion['model_path'])
                model = loaded.model if hasattr(loaded, 'model') else loaded
                result = mlflow.sklearn.log_model(
                    model, 'model',
                    registered_model_name='market-predictor',
                )
                logger.info(f"Model registered as 'market-predictor'")
                return {
                    'status': 'registered',
                    'model_name': 'market-predictor',
                    'model_uri': result.model_uri,
                }

    except Exception as e:
        logger.warning(f"MLflow registration failed: {e}")

    return {
        'status': 'logged',
        'model_name': champion['model_name'],
        'mlflow_uri': MLFLOW_TRACKING_URI,
    }


def generate_model_card(**context):
    """Generate model card documentation."""
    from pathlib import Path

    ti = context['task_instance']
    champion = ti.xcom_pull(task_ids='select_champion')
    evaluation = ti.xcom_pull(task_ids='evaluate_models')
    features = ti.xcom_pull(task_ids='feature_importance')

    model_card = {
        'model_name': champion.get('model_name', 'unknown'),
        'version': '1.0',
        'created_at': str(datetime.now()),
        'val_metrics': champion.get('val_metrics', {}),
        'test_metrics': champion.get('test_metrics', {}),
        'top_features': features.get('top_features', [])[:10] if features else [],
        'total_models_evaluated': evaluation.get('total_models', 0),
        'intended_use': 'Financial market prediction - research/educational use only',
        'limitations': [
            'Past performance does not guarantee future results',
            'Trained on synthetic/limited historical data',
            'Not suitable for production trading without additional validation',
        ],
    }

    # Save model card
    output_path = Path(MODEL_SAVE_PATH) / 'reports'
    output_path.mkdir(parents=True, exist_ok=True)
    card_file = output_path / 'model_card.json'
    with open(card_file, 'w') as f:
        json.dump(model_card, f, indent=2, default=str)

    logger.info(f"Model card saved to {card_file}")
    return model_card


# Define the DAG
with DAG(
    dag_id='model_training_pipeline',
    default_args=default_args,
    description='Train and evaluate machine learning models',
    schedule_interval='0 6 * * 0',  # Run weekly on Sunday at 6 AM UTC
    catchup=False,
    max_active_runs=1,
    tags=['model-training', 'ml-pipeline', 'mlops'],
) as dag:

    wait_for_features = ExternalTaskSensor(
        task_id='wait_for_features',
        external_dag_id='feature_engineering_pipeline',
        external_task_id='store_features',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='poke',
        poke_interval=60,
        timeout=3600,
    )

    load_features_task = PythonOperator(
        task_id='load_features',
        python_callable=load_features,
        provide_context=True,
    )

    prepare_split = PythonOperator(
        task_id='prepare_split',
        python_callable=prepare_train_test_split,
        provide_context=True,
    )

    with TaskGroup('supervised_models', tooltip='Train supervised learning models') as supervised_models:
        xgboost = PythonOperator(
            task_id='train_xgboost',
            python_callable=train_xgboost_regressor,
            provide_context=True,
        )
        rf = PythonOperator(
            task_id='train_random_forest',
            python_callable=train_random_forest,
            provide_context=True,
        )
        lgbm = PythonOperator(
            task_id='train_lightgbm',
            python_callable=train_lightgbm,
            provide_context=True,
        )
        lstm = PythonOperator(
            task_id='train_lstm',
            python_callable=train_lstm,
            provide_context=True,
        )
        [xgboost, rf, lgbm, lstm]

    with TaskGroup('unsupervised_models', tooltip='Train unsupervised learning models') as unsupervised_models:
        isolation = PythonOperator(
            task_id='train_isolation_forest',
            python_callable=train_isolation_forest,
            provide_context=True,
        )
        kmeans = PythonOperator(
            task_id='train_kmeans',
            python_callable=train_kmeans,
            provide_context=True,
        )
        autoencoder = PythonOperator(
            task_id='train_autoencoder',
            python_callable=train_autoencoder,
            provide_context=True,
        )
        [isolation, kmeans, autoencoder]

    ensemble = PythonOperator(
        task_id='create_ensemble',
        python_callable=create_ensemble,
        provide_context=True,
    )

    evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_all_models,
        provide_context=True,
        trigger_rule='none_failed',
    )

    champion = PythonOperator(
        task_id='select_champion',
        python_callable=select_champion_model,
        provide_context=True,
    )

    importance = PythonOperator(
        task_id='feature_importance',
        python_callable=calculate_feature_importance,
        provide_context=True,
    )

    register = PythonOperator(
        task_id='register_mlflow',
        python_callable=register_models_mlflow,
        provide_context=True,
    )

    model_card = PythonOperator(
        task_id='generate_model_card',
        python_callable=generate_model_card,
        provide_context=True,
    )

    end = EmptyOperator(task_id='end')

    wait_for_features >> load_features_task >> prepare_split
    prepare_split >> [supervised_models, unsupervised_models]
    supervised_models >> ensemble
    [ensemble, unsupervised_models] >> evaluate
    evaluate >> champion >> [importance, register, model_card]
    [importance, register, model_card] >> end
