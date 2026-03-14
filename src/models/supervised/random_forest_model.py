"""
Random Forest regression model for financial prediction.

Implements RandomForest-based regression using the BaseRegressionModel
interface with MLflow tracking and feature importance analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.supervised.regression import BaseRegressionModel

logger = logging.getLogger(__name__)


class RandomForestRegressionModel(BaseRegressionModel):
    """
    Random Forest regression model for financial prediction.

    Advantages over single decision trees:
    - Reduces overfitting through bagging
    - Provides feature importance rankings
    - Robust to outliers and noisy features
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        default_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        super().__init__(
            model_name="RandomForest Regressor",
            hyperparameters=default_params,
            random_state=random_state,
        )

    def _create_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            **self.hyperparameters,
            random_state=self.random_state,
        )


def create_random_forest_regressor(
    config: dict[str, Any] | None = None,
) -> RandomForestRegressionModel:
    """Factory function to create Random Forest regression model."""
    hyperparameters = config.get("hyperparameters") if config else None
    random_state = config.get("random_state", 42) if config else 42
    return RandomForestRegressionModel(
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


if __name__ == "__main__":
    print("Loading training data...")
    train_df = pd.read_parquet("data/processed/train_data.parquet")
    test_df = pd.read_parquet("data/processed/test_data.parquet")

    feature_cols = [
        col for col in train_df.columns
        if col not in ["date", "symbol", "target_return", "target_direction"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_return"]

    print(f"\nTraining Random Forest on {len(X_train)} samples with {len(feature_cols)} features")

    model = RandomForestRegressionModel()
    model.train(X_train, y_train)

    train_metrics = model.evaluate(X_train, y_train, "train")
    test_metrics = model.evaluate(X_test, y_test, "test")

    print("\nTop 10 Most Important Features:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10))

    model.save_model("models/random_forest_regressor")
    print("\nModel saved.")
