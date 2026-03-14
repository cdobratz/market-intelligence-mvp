"""
LightGBM regression model for financial prediction.

Implements LightGBM-based regression using the BaseRegressionModel
interface. Faster training than XGBoost with comparable accuracy.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.supervised.regression import BaseRegressionModel

logger = logging.getLogger(__name__)


class LightGBMRegressionModel(BaseRegressionModel):
    """
    LightGBM regression model for financial prediction.

    Advantages:
    - Faster training than XGBoost (histogram-based)
    - Lower memory usage with leaf-wise tree growth
    - Native categorical feature support
    - Early stopping with validation set
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
        early_stopping_rounds: int = 20,
    ):
        self.early_stopping_rounds = early_stopping_rounds

        default_params = {
            "n_estimators": 200,
            "num_leaves": 31,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbose": -1,
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        super().__init__(
            model_name="LightGBM Regressor",
            hyperparameters=default_params,
            random_state=random_state,
        )

    def _create_model(self) -> LGBMRegressor:
        return LGBMRegressor(
            **self.hyperparameters,
            random_state=self.random_state,
        )

    def _train_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train with early stopping using validation set."""
        from lightgbm import early_stopping, log_evaluation

        callbacks = [log_evaluation(period=0)]
        if self.early_stopping_rounds > 0:
            callbacks.append(early_stopping(self.early_stopping_rounds))

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        if hasattr(self.model, "best_iteration_"):
            logger.info(f"Best iteration: {self.model.best_iteration_}")


def create_lightgbm_regressor(
    config: dict[str, Any] | None = None,
) -> LightGBMRegressionModel:
    """Factory function to create LightGBM regression model."""
    hyperparameters = config.get("hyperparameters") if config else None
    random_state = config.get("random_state", 42) if config else 42
    return LightGBMRegressionModel(
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

    print(f"\nTraining LightGBM on {len(X_train)} samples with {len(feature_cols)} features")

    model = LightGBMRegressionModel()
    model.train(X_train, y_train)

    train_metrics = model.evaluate(X_train, y_train, "train")
    test_metrics = model.evaluate(X_test, y_test, "test")

    print("\nTop 10 Most Important Features:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10))

    model.save_model("models/lightgbm_regressor")
    print("\nModel saved.")
