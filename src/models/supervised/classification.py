"""
Classification models for market direction prediction.

Predicts whether price will go Up, Down, or Neutral over a given horizon.
Implements XGBoost and Random Forest classifiers with MLflow tracking.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class DirectionClassifier:
    """
    Market direction classifier.

    Predicts binary direction: 1 = price goes up, 0 = price goes down
    over a specified horizon (default 5 days).
    """

    def __init__(
        self,
        classifier_type: str = "xgboost",
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.feature_names: list[str] | None = None
        self.training_metadata: dict[str, Any] = {}

        self.model = self._create_model()

    def _create_model(self):
        if self.classifier_type == "xgboost":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            default_params.update(self.hyperparameters)
            return XGBClassifier(**default_params, random_state=self.random_state)

        elif self.classifier_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
            }
            default_params.update(self.hyperparameters)
            return RandomForestClassifier(**default_params, random_state=self.random_state)

        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "DirectionClassifier":
        self.feature_names = list(X_train.columns)

        if X_val is not None and y_val is not None and self.classifier_type == "xgboost":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        self.training_metadata = {
            "train_samples": len(X_train),
            "n_features": X_train.shape[1],
            "classifier_type": self.classifier_type,
        }

        logger.info(f"Trained {self.classifier_type} classifier on {len(X_train)} samples")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "test"
    ) -> dict[str, float]:
        y_pred = self.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        metrics = {
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_precision": precision,
            f"{dataset_name}_recall": recall,
            f"{dataset_name}_f1": f1,
        }

        logger.info(f"{dataset_name.upper()} Classification Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")

        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        logger.info(f"\n{classification_report(y, y_pred, zero_division=0)}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame | None:
        if not hasattr(self.model, "feature_importances_"):
            return None

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def log_to_mlflow(
        self,
        experiment_name: str,
        run_name: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> str:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("classifier_type", self.classifier_type)
            mlflow.log_params(self.training_metadata)

            if metrics:
                mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(self.model, "model")

            importance = self.get_feature_importance()
            if importance is not None:
                path = "temp_clf_importance.csv"
                importance.to_csv(path, index=False)
                mlflow.log_artifact(path)
                import os
                os.unlink(path)

            return run.info.run_id

    def save_model(self, output_dir: str) -> str:
        import joblib
        from pathlib import Path
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / "classifier_model.pkl"
        joblib.dump(self.model, model_path)

        metadata = {
            "classifier_type": self.classifier_type,
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
        }
        with open(output_path / "classifier_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Classifier saved to {model_path}")
        return str(model_path)

    def load_model(self, model_dir: str) -> "DirectionClassifier":
        import joblib
        from pathlib import Path
        import json

        model_path = Path(model_dir)
        self.model = joblib.load(model_path / "classifier_model.pkl")

        with open(model_path / "classifier_metadata.json") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.training_metadata = metadata["training_metadata"]
        return self


if __name__ == "__main__":
    print("Loading training data...")
    train_df = pd.read_parquet("data/processed/train_data.parquet")
    test_df = pd.read_parquet("data/processed/test_data.parquet")

    feature_cols = [
        col for col in train_df.columns
        if col not in ["date", "symbol", "target_return", "target_direction"]
    ]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Create direction target: 1 if return > 0, else 0
    if "target_direction" in train_df.columns:
        y_train = train_df["target_direction"]
        y_test = test_df["target_direction"]
    else:
        y_train = (train_df["target_return"] > 0).astype(int)
        y_test = (test_df["target_return"] > 0).astype(int)

    print(f"\nTraining on {len(X_train)} samples with {len(feature_cols)} features")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    # Train XGBoost Classifier
    print("\n--- XGBoost Classifier ---")
    xgb_clf = DirectionClassifier(classifier_type="xgboost")
    xgb_clf.train(X_train, y_train)
    xgb_metrics = xgb_clf.evaluate(X_test, y_test)

    # Train Random Forest Classifier
    print("\n--- Random Forest Classifier ---")
    rf_clf = DirectionClassifier(classifier_type="random_forest")
    rf_clf.train(X_train, y_train)
    rf_metrics = rf_clf.evaluate(X_test, y_test)

    print("\nDone.")
