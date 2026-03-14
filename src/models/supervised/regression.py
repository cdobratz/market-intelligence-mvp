"""
Base regression model classes for financial prediction.

This module provides abstract base classes and utilities for regression models
that predict continuous values (e.g., price changes, returns).
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRegressionModel(ABC):
    """Abstract base class for regression models."""

    def __init__(
        self,
        model_name: str,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        """
        Initialize the base regression model.

        Args:
            model_name: Name of the model (e.g., "XGBoost", "RandomForest")
            hyperparameters: Dictionary of model hyperparameters
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.hyperparameters = hyperparameters or {}
        self.random_state = random_state
        self.model = None
        self.feature_names: list[str] | None = None
        self.training_metadata: dict[str, Any] = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the underlying model instance.

        Returns:
            Model instance (e.g., XGBRegressor, RandomForestRegressor)
        """
        pass

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseRegressionModel":
        """
        Train the regression model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()

        # Store training start time
        train_start = datetime.now()

        # Train the model
        if X_val is not None and y_val is not None:
            # Some models support validation sets
            self._train_with_validation(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)

        # Calculate training time
        training_time = (datetime.now() - train_start).total_seconds()

        # Store metadata
        self.training_metadata = {
            "train_samples": len(X_train),
            "n_features": X_train.shape[1],
            "training_time_seconds": training_time,
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"Training completed in {training_time:.2f} seconds")

        return self

    def _train_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Train with validation set (override if model supports it).

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        # Default: just train without using validation
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "test"
    ) -> dict[str, float]:
        """
        Evaluate the model and return metrics.

        Args:
            X: Features
            y: True target values
            dataset_name: Name of the dataset (e.g., "train", "test")

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y - predictions) / y)) * 100

        # Calculate directional accuracy
        if len(y) > 1:
            actual_direction = np.sign(np.diff(y.values))
            pred_direction = np.sign(np.diff(predictions))
            directional_accuracy = (
                (actual_direction == pred_direction).sum() / len(actual_direction) * 100
            )
        else:
            directional_accuracy = 0.0

        metrics = {
            f"{dataset_name}_rmse": rmse,
            f"{dataset_name}_mae": mae,
            f"{dataset_name}_r2": r2,
            f"{dataset_name}_mape": mape,
            f"{dataset_name}_directional_accuracy": directional_accuracy,
        }

        logger.info(f"{dataset_name.upper()} Metrics:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")

        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> dict[str, Any]:
        """
        Perform time-series cross-validation.

        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {n_splits}-fold time-series cross-validation")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {"rmse": [], "mae": [], "r2": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create a fresh model for each fold
            fold_model = self._create_model()
            fold_model.fit(X_train, y_train)

            # Evaluate
            y_pred = fold_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            cv_scores["rmse"].append(rmse)
            cv_scores["mae"].append(mae)
            cv_scores["r2"].append(r2)

            logger.info(f"  Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")

        # Calculate mean and std
        results = {
            "cv_rmse_mean": np.mean(cv_scores["rmse"]),
            "cv_rmse_std": np.std(cv_scores["rmse"]),
            "cv_mae_mean": np.mean(cv_scores["mae"]),
            "cv_mae_std": np.std(cv_scores["mae"]),
            "cv_r2_mean": np.mean(cv_scores["r2"]),
            "cv_r2_std": np.std(cv_scores["r2"]),
        }

        logger.info(f"CV Results: RMSE={results['cv_rmse_mean']:.6f} ± {results['cv_rmse_std']:.6f}")

        return results

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Get feature importance if available.

        Returns:
            DataFrame with feature names and importance scores, or None
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if not hasattr(self.model, "feature_importances_"):
            logger.warning(f"{self.model_name} does not support feature importance")
            return None

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, output_dir: str) -> str:
        """
        Save the model to disk.

        Args:
            output_dir: Directory to save the model

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_filename = f"{self.model_name.lower().replace(' ', '_')}_model.pkl"
        model_path = output_path / model_filename
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
        }

        metadata_path = output_path / f"{self.model_name.lower().replace(' ', '_')}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

        return str(model_path)

    def load_model(self, model_dir: str) -> "BaseRegressionModel":
        """
        Load a saved model.

        Args:
            model_dir: Directory containing the saved model

        Returns:
            Self for method chaining
        """
        model_path = Path(model_dir)

        # Load model
        model_filename = f"{self.model_name.lower().replace(' ', '_')}_model.pkl"
        self.model = joblib.load(model_path / model_filename)

        # Load metadata
        metadata_path = model_path / f"{self.model_name.lower().replace(' ', '_')}_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.training_metadata = metadata["training_metadata"]

        logger.info(f"Model loaded from {model_path}")

        return self

    def log_to_mlflow(
        self,
        experiment_name: str,
        run_name: str | None = None,
        metrics: dict[str, float] | None = None,
        artifacts_dir: str | None = None,
    ) -> str:
        """
        Log model and metrics to MLflow.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of the run (optional)
            metrics: Dictionary of metrics to log
            artifacts_dir: Directory containing artifacts to log (optional)

        Returns:
            MLflow run ID
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(self.hyperparameters)
            mlflow.log_params(
                {
                    "model_type": self.model_name,
                    "n_features": len(self.feature_names) if self.feature_names else 0,
                    "random_state": self.random_state,
                }
            )

            # Log training metadata
            mlflow.log_params(self.training_metadata)

            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            # Log feature importance if available
            feature_importance = self.get_feature_importance()
            if feature_importance is not None:
                importance_path = Path("temp_feature_importance.csv")
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                importance_path.unlink()  # Clean up

            # Log artifacts
            if artifacts_dir:
                mlflow.log_artifacts(artifacts_dir)

            logger.info(f"Logged to MLflow run: {run.info.run_id}")

            return run.info.run_id


def walk_forward_validation(
    model: BaseRegressionModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 30,
    gap: int = 0,
) -> dict[str, Any]:
    """
    Walk-forward validation that simulates real trading conditions.

    This function implements expanding window walk-forward validation,
    which is the gold standard for evaluating time-series models in
    quantitative finance. It prevents look-ahead bias by always training
    on past data and testing on future data.

    Args:
        model: BaseRegressionModel instance to evaluate
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of train/test splits
        test_size: Number of samples in each test fold
        gap: Number of samples to skip between train and test (for data leakage prevention)

    Returns:
        Dictionary with validation results including:
        - mean_rmse, std_rmse: RMSE statistics across folds
        - mean_mae, std_mae: MAE statistics across folds
        - mean_r2, std_r2: R² statistics across folds
        - mean_directional_accuracy: Average directional accuracy
        - fold_metrics: Detailed metrics for each fold
        - all_predictions: All out-of-sample predictions
        - all_actuals: Corresponding actual values
    """
    logger.info(f"Starting walk-forward validation with {n_splits} splits, test_size={test_size}")

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    metrics_per_fold = []
    all_predictions = []
    all_actuals = []
    fold_indices = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"Fold {fold}/{n_splits}: Train size={len(train_idx)}, Test size={len(test_idx)}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create a fresh model instance for each fold
        fold_model = model._create_model()

        # Train on historical data
        fold_model.fit(X_train, y_train)

        # Predict on future data
        y_pred = fold_model.predict(X_test)

        # Calculate fold metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test.values))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = (
                (actual_direction == pred_direction).sum() / len(actual_direction) * 100
            )
        else:
            directional_accuracy = 0.0

        fold_metrics = {
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
        }
        metrics_per_fold.append(fold_metrics)

        # Store predictions
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.values.tolist())
        fold_indices.extend([fold] * len(test_idx))

        logger.info(
            f"  Fold {fold} Results: RMSE={rmse:.6f}, MAE={mae:.6f}, "
            f"R²={r2:.6f}, Dir. Acc={directional_accuracy:.2f}%"
        )

    # Aggregate metrics across folds
    results = {
        "n_splits": n_splits,
        "test_size": test_size,
        "gap": gap,
        "mean_rmse": np.mean([m["rmse"] for m in metrics_per_fold]),
        "std_rmse": np.std([m["rmse"] for m in metrics_per_fold]),
        "mean_mae": np.mean([m["mae"] for m in metrics_per_fold]),
        "std_mae": np.std([m["mae"] for m in metrics_per_fold]),
        "mean_r2": np.mean([m["r2"] for m in metrics_per_fold]),
        "std_r2": np.std([m["r2"] for m in metrics_per_fold]),
        "mean_directional_accuracy": np.mean(
            [m["directional_accuracy"] for m in metrics_per_fold]
        ),
        "std_directional_accuracy": np.std(
            [m["directional_accuracy"] for m in metrics_per_fold]
        ),
        "fold_metrics": metrics_per_fold,
        "all_predictions": all_predictions,
        "all_actuals": all_actuals,
        "fold_indices": fold_indices,
    }

    logger.info(
        f"\nWalk-Forward Validation Complete:"
        f"\n  RMSE: {results['mean_rmse']:.6f} ± {results['std_rmse']:.6f}"
        f"\n  MAE:  {results['mean_mae']:.6f} ± {results['std_mae']:.6f}"
        f"\n  R²:   {results['mean_r2']:.6f} ± {results['std_r2']:.6f}"
        f"\n  Directional Accuracy: {results['mean_directional_accuracy']:.2f}% "
        f"± {results['std_directional_accuracy']:.2f}%"
    )

    return results


def expanding_window_backtest(
    model: BaseRegressionModel,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = 100,
    step_size: int = 1,
    retrain_frequency: int = 20,
) -> dict[str, Any]:
    """
    Expanding window backtest with configurable retraining frequency.

    This implements a more realistic trading simulation where the model
    is retrained periodically as new data becomes available.

    Args:
        model: BaseRegressionModel instance
        X: Feature DataFrame
        y: Target Series
        initial_train_size: Minimum training samples before first prediction
        step_size: Number of steps to predict before potential retrain
        retrain_frequency: Retrain model every N predictions

    Returns:
        Dictionary with backtest results
    """
    logger.info(
        f"Starting expanding window backtest: initial_train={initial_train_size}, "
        f"retrain_freq={retrain_frequency}"
    )

    predictions = []
    actuals = []
    timestamps = []
    trained_model = None
    last_train_idx = 0

    for i in range(initial_train_size, len(X)):
        # Check if retraining is needed
        should_retrain = (
            trained_model is None or
            (i - last_train_idx) >= retrain_frequency
        )

        if should_retrain:
            # Train on all available historical data
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            trained_model = model._create_model()
            trained_model.fit(X_train, y_train)
            last_train_idx = i

            logger.debug(f"Retrained at index {i} with {len(X_train)} samples")

        # Make prediction
        X_test = X.iloc[[i]]
        y_pred = trained_model.predict(X_test)[0]

        predictions.append(y_pred)
        actuals.append(y.iloc[i])

        if hasattr(X, 'index'):
            timestamps.append(X.index[i])

    # Calculate overall metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    directional_accuracy = (actual_direction == pred_direction).sum() / len(actual_direction) * 100

    results = {
        "initial_train_size": initial_train_size,
        "retrain_frequency": retrain_frequency,
        "n_predictions": len(predictions),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "timestamps": timestamps if timestamps else None,
    }

    logger.info(
        f"Expanding Window Backtest Complete: "
        f"RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, "
        f"Dir. Acc={directional_accuracy:.2f}%"
    )

    return results
