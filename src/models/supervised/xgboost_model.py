"""
XGBoost regression model for financial prediction.

This module implements XGBoost-based regression for predicting continuous
financial metrics like price changes and returns.
"""

from typing import Dict, Any, Optional
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from src.models.supervised.regression import BaseRegressionModel


class XGBoostRegressionModel(BaseRegressionModel):
    """XGBoost regression model for financial prediction."""

    def __init__(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost regression model.

        Args:
            hyperparameters: XGBoost hyperparameters
            random_state: Random seed for reproducibility
        """
        # Default hyperparameters optimized for financial data
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,  # L1 regularization
            "reg_lambda": 1,  # L2 regularization
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }

        # Merge with user-provided params
        if hyperparameters:
            default_params.update(hyperparameters)

        super().__init__(
            model_name="XGBoost Regressor",
            hyperparameters=default_params,
            random_state=random_state,
        )

    def _create_model(self) -> XGBRegressor:
        """
        Create XGBoost regressor instance.

        Returns:
            Configured XGBRegressor
        """
        return XGBRegressor(
            **self.hyperparameters,
            random_state=self.random_state,
            enable_categorical=False,
        )

    def _train_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Train with early stopping using validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        # XGBoost supports early stopping with validation
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )


def create_xgboost_regressor(
    config: Optional[Dict[str, Any]] = None,
) -> XGBoostRegressionModel:
    """
    Factory function to create XGBoost regression model.

    Args:
        config: Configuration dictionary with hyperparameters

    Returns:
        Configured XGBoostRegressionModel instance
    """
    hyperparameters = config.get("hyperparameters") if config else None
    random_state = config.get("random_state", 42) if config else 42

    return XGBoostRegressionModel(
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from pathlib import Path

    print("Loading training data...")
    train_df = pd.read_parquet("data/processed/train_data.parquet")
    test_df = pd.read_parquet("data/processed/test_data.parquet")

    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col not in
                    ["date", "symbol", "target_return", "target_direction"]]

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_return"]

    print(f"\nTraining XGBoost on {len(X_train)} samples with {len(feature_cols)} features")
    print(f"Test set: {len(X_test)} samples")

    # Create and train model
    model = XGBoostRegressionModel(
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
        }
    )

    # Train
    model.train(X_train, y_train)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train, "train")
    test_metrics = model.evaluate(X_test, y_test, "test")

    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10))

    # Save model
    output_dir = Path("models/xgboost_regressor")
    model.save_model(str(output_dir))
    print(f"\nModel saved to {output_dir}")
