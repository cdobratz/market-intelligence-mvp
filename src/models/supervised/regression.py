"""
Base regression model classes for financial prediction.

This module provides abstract base classes and utilities for regression models
that predict continuous values (e.g., price changes, returns).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
import joblib
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRegressionModel(ABC):
    """Abstract base class for regression models."""

    def __init__(
        self,
        model_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
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
        self.feature_names: Optional[list[str]] = None
        self.training_metadata: Dict[str, Any] = {}

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
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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
    ) -> Dict[str, float]:
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
            actual_direction = np.sign(y.diff())
            pred_direction = np.sign(pd.Series(predictions).diff())
            directional_accuracy = (
                (actual_direction == pred_direction).sum() / (len(y) - 1) * 100
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
    ) -> Dict[str, Any]:
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

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
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
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.training_metadata = metadata["training_metadata"]

        logger.info(f"Model loaded from {model_path}")

        return self

    def log_to_mlflow(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts_dir: Optional[str] = None,
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
