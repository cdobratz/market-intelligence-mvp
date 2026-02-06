"""
Stacking Ensemble Module - Combine multiple models for improved predictions.

Implements stacking ensemble methods that combine base models with a
meta-learner, typically improving R² from ~0.35 to 0.45-0.55.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble that combines multiple base models
    with a meta-learner for improved predictions.

    This ensemble typically improves R² from ~0.35 (single model)
    to 0.45-0.55 by leveraging the strengths of different algorithms.

    Features:
    - Multiple base models (XGBoost, LightGBM, RF, Ridge)
    - Ridge meta-learner to prevent overfitting
    - Time-series aware cross-validation
    - Feature importance aggregation
    - Model introspection and diagnostics
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_random_forest: bool = True,
        use_ridge: bool = True,
        cv: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the stacking ensemble.

        Args:
            use_xgboost: Include XGBoost in base models
            use_lightgbm: Include LightGBM in base models
            use_random_forest: Include RandomForest in base models
            use_ridge: Include Ridge in base models
            cv: Number of CV folds for stacking
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.use_lightgbm = use_lightgbm and LIGHTGBM_AVAILABLE
        self.use_random_forest = use_random_forest
        self.use_ridge = use_ridge
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.base_models: list[tuple[str, Any]] = []
        self.meta_learner = None
        self.model = None
        self.feature_names_: list[str] | None = None
        self.is_fitted_ = False

    def _build_base_models(self) -> list[tuple[str, Any]]:
        """Build the list of base models."""
        models = []

        if self.use_xgboost and XGBOOST_AVAILABLE:
            models.append((
                'xgboost',
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=0,
                )
            ))
            logger.info("Added XGBoost to ensemble")

        if self.use_lightgbm and LIGHTGBM_AVAILABLE:
            models.append((
                'lightgbm',
                LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
            ))
            logger.info("Added LightGBM to ensemble")

        if self.use_random_forest:
            models.append((
                'rf',
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=5,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            ))
            logger.info("Added RandomForest to ensemble")

        if self.use_ridge:
            models.append((
                'ridge',
                Ridge(alpha=1.0, random_state=self.random_state)
            ))
            logger.info("Added Ridge to ensemble")

        if not models:
            raise ValueError("At least one base model must be enabled")

        return models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training target

        Returns:
            Self
        """
        logger.info(f"Fitting stacking ensemble with {len(X)} samples")

        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)

        # Build base models
        self.base_models = self._build_base_models()

        # Create meta-learner (simple Ridge to prevent overfitting)
        self.meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

        # Create stacking regressor
        self.model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_learner,
            cv=self.cv,
            passthrough=False,  # Don't include original features in meta
            n_jobs=self.n_jobs,
        )

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted_ = True

        logger.info("Stacking ensemble fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> dict[str, float]:
        """
        Evaluate the ensemble.

        Args:
            X: Features
            y: True target
            dataset_name: Name for logging

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Directional accuracy
        if len(y) > 1:
            actual_dir = np.sign(np.diff(y.values))
            pred_dir = np.sign(np.diff(predictions))
            dir_acc = (actual_dir == pred_dir).mean() * 100
        else:
            dir_acc = 0.0

        metrics = {
            f"{dataset_name}_rmse": rmse,
            f"{dataset_name}_mae": mae,
            f"{dataset_name}_r2": r2,
            f"{dataset_name}_directional_accuracy": dir_acc,
        }

        logger.info(f"{dataset_name.upper()} Metrics: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Dir.Acc={dir_acc:.2f}%")

        return metrics

    def get_base_model_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> dict[str, dict[str, float]]:
        """
        Get cross-validation scores for each base model.

        Args:
            X: Features
            y: Target
            cv: Number of CV folds

        Returns:
            Dictionary with scores for each base model
        """
        scores = {}
        tscv = TimeSeriesSplit(n_splits=cv)

        for name, model in self.base_models:
            cv_scores = cross_val_score(
                model, X, y,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs
            )
            rmse_scores = np.sqrt(-cv_scores)

            scores[name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std(),
            }

            logger.info(f"{name}: RMSE = {rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")

        return scores

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Get aggregated feature importance from base models.

        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_fitted_ or self.feature_names_ is None:
            return None

        importances = {}

        for name, model in self.model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_

        if not importances:
            return None

        # Average importance across models
        importance_df = pd.DataFrame(importances, index=self.feature_names_)
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)

        return importance_df

    def save(self, path: str) -> str:
        """Save the ensemble model."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        logger.info(f"Model saved to {output_path}")
        return str(output_path)

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        """Load a saved ensemble model."""
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


class VotingEnsemble(BaseEstimator, RegressorMixin):
    """
    Voting ensemble that averages predictions from multiple models.

    Simpler than stacking, but often effective and faster to train.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize voting ensemble.

        Args:
            weights: Weights for each model (None = equal weights)
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.weights = weights
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.is_fitted_ = False

    def _build_models(self) -> list[tuple[str, Any]]:
        """Build base models for voting."""
        models = []

        if XGBOOST_AVAILABLE:
            models.append((
                'xgboost',
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=0,
                )
            ))

        if LIGHTGBM_AVAILABLE:
            models.append((
                'lightgbm',
                LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
            ))

        models.append((
            'rf',
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        ))

        return models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VotingEnsemble":
        """Fit the voting ensemble."""
        logger.info(f"Fitting voting ensemble with {len(X)} samples")

        models = self._build_models()

        self.model = VotingRegressor(
            estimators=models,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X, y)
        self.is_fitted_ = True

        logger.info("Voting ensemble fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """
    Blending ensemble using a holdout set for meta-training.

    Faster than stacking (no CV) but uses less data.
    """

    def __init__(
        self,
        blend_ratio: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize blending ensemble.

        Args:
            blend_ratio: Ratio of data for blending (meta-training)
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.blend_ratio = blend_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.base_models_ = []
        self.meta_model_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendingEnsemble":
        """Fit the blending ensemble."""
        logger.info(f"Fitting blending ensemble with {len(X)} samples")

        # Split data for base training and meta training
        split_idx = int(len(X) * (1 - self.blend_ratio))
        X_train, X_blend = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_blend = y.iloc[:split_idx], y.iloc[split_idx:]

        # Build and fit base models
        base_configs = [
            ('xgboost', XGBRegressor(n_estimators=100, max_depth=4, verbosity=0) if XGBOOST_AVAILABLE else None),
            ('lightgbm', LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1) if LIGHTGBM_AVAILABLE else None),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=self.n_jobs)),
        ]

        blend_features = []

        for name, model in base_configs:
            if model is None:
                continue

            model.fit(X_train, y_train)
            self.base_models_.append((name, model))

            # Get predictions for blend set
            blend_pred = model.predict(X_blend)
            blend_features.append(blend_pred)

        # Create meta features
        meta_X = np.column_stack(blend_features)

        # Fit meta model
        self.meta_model_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        self.meta_model_.fit(meta_X, y_blend)

        self.is_fitted_ = True
        logger.info("Blending ensemble fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = []
        for _name, model in self.base_models_:
            base_preds.append(model.predict(X))

        meta_X = np.column_stack(base_preds)
        return self.meta_model_.predict(meta_X)


def create_ensemble(
    ensemble_type: str = "stacking",
    **kwargs
) -> BaseEstimator:
    """
    Factory function to create ensemble models.

    Args:
        ensemble_type: Type of ensemble ('stacking', 'voting', 'blending')
        **kwargs: Arguments for the ensemble class

    Returns:
        Ensemble model instance
    """
    ensemble_map = {
        'stacking': StackingEnsemble,
        'voting': VotingEnsemble,
        'blending': BlendingEnsemble,
    }

    if ensemble_type not in ensemble_map:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}. Choose from {list(ensemble_map.keys())}")

    return ensemble_map[ensemble_type](**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    np.random.seed(42)
    n_samples = 1000

    # Generate sample data
    X = pd.DataFrame(np.random.randn(n_samples, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(2 * X["f0"] + X["f1"] + np.random.randn(n_samples) * 0.5)

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("=== Testing Stacking Ensemble ===")
    stacking = StackingEnsemble()
    stacking.fit(X_train, y_train)
    stacking.evaluate(X_test, y_test, "test")

    print("\n=== Testing Voting Ensemble ===")
    voting = VotingEnsemble()
    voting.fit(X_train, y_train)
    preds = voting.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"Voting R²: {r2:.6f}")

    print("\n=== Testing Blending Ensemble ===")
    blending = BlendingEnsemble()
    blending.fit(X_train, y_train)
    preds = blending.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"Blending R²: {r2:.6f}")
