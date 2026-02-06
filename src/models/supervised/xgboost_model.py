"""
XGBoost regression model for financial prediction.

This module implements XGBoost-based regression for predicting continuous
financial metrics like price changes and returns.

Features:
- GPU acceleration support (set use_gpu=True)
- Proper early stopping with validation set
- Data augmentation for small datasets
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.supervised.regression import BaseRegressionModel

logger = logging.getLogger(__name__)


def augment_financial_data(
    df: pd.DataFrame,
    noise_level: float = 0.01,
    n_augmentations: int = 1,
) -> pd.DataFrame:
    """
    Add Gaussian noise to create augmented training samples.

    Data augmentation can help improve model generalization when
    training data is limited. Use with caution for financial data.

    Args:
        df: DataFrame to augment
        noise_level: Standard deviation of noise as fraction of column std
        n_augmentations: Number of augmented copies to create

    Returns:
        DataFrame with original and augmented samples
    """
    augmented_dfs = [df]

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for _i in range(n_augmentations):
        augmented = df.copy()

        # Add Gaussian noise scaled by column std
        for col in numeric_cols:
            col_std = df[col].std()
            if col_std > 0:
                noise = np.random.normal(0, noise_level * col_std, len(df))
                augmented[col] = augmented[col] + noise

        augmented_dfs.append(augmented)

    result = pd.concat(augmented_dfs, ignore_index=True)
    logger.info(f"Augmented data: {len(df)} -> {len(result)} samples")
    return result


def check_gpu_available() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        # Try to create a small GPU-enabled model
        test_model = XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=1)
        test_X = np.random.randn(10, 5)
        test_y = np.random.randn(10)
        test_model.fit(test_X, test_y)
        return True
    except Exception:
        return False


class XGBoostRegressionModel(BaseRegressionModel):
    """
    XGBoost regression model for financial prediction.

    Features:
    - GPU acceleration (10-50x speedup when available)
    - Proper early stopping to prevent overfitting
    - Data augmentation support
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int = 42,
        use_gpu: bool = False,
        early_stopping_rounds: int = 20,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.01,
    ):
        """
        Initialize XGBoost regression model.

        Args:
            hyperparameters: XGBoost hyperparameters
            random_state: Random seed for reproducibility
            use_gpu: Enable GPU acceleration if available
            early_stopping_rounds: Rounds for early stopping (0 to disable)
            use_augmentation: Enable data augmentation
            augmentation_noise: Noise level for augmentation
        """
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise

        # Check GPU availability
        if use_gpu:
            self._gpu_available = check_gpu_available()
            if not self._gpu_available:
                logger.warning("GPU requested but not available. Falling back to CPU.")
        else:
            self._gpu_available = False

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

        # GPU-specific settings
        if self.use_gpu and self._gpu_available:
            default_params.update({
                "tree_method": "gpu_hist",
                "device": "cuda",
                "n_jobs": 1,  # GPU doesn't use multiple jobs
            })
            logger.info("XGBoost GPU acceleration enabled")
        else:
            default_params["tree_method"] = "hist"  # Fast CPU method

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

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostRegressionModel":
        """
        Train the XGBoost model with optional augmentation.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, enables early stopping)
            y_val: Validation target (optional)

        Returns:
            Self for method chaining
        """
        # Apply data augmentation if enabled
        if self.use_augmentation:
            train_df = X_train.copy()
            train_df['_target'] = y_train.values
            augmented = augment_financial_data(
                train_df,
                noise_level=self.augmentation_noise
            )
            X_train = augmented.drop('_target', axis=1)
            y_train = augmented['_target']

        # Call parent train method
        return super().train(X_train, y_train, X_val, y_val)

    def _train_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Train with proper early stopping using validation set.

        Early stopping prevents overfitting by monitoring validation
        performance and stopping when it stops improving.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        fit_params = {
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "verbose": False,
        }

        # Enable early stopping if rounds > 0
        if self.early_stopping_rounds > 0:
            # Note: XGBoost 2.0+ uses early_stopping_rounds in fit()
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            logger.info(f"Early stopping enabled: {self.early_stopping_rounds} rounds")

        self.model.fit(X_train, y_train, **fit_params)

        # Log best iteration if early stopping was used
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")


def create_xgboost_regressor(
    config: dict[str, Any] | None = None,
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
    from pathlib import Path

    import numpy as np

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
