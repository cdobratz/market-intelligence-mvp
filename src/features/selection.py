"""
Feature Selection Module - Reduce overfitting by selecting important features.

Provides multiple feature selection methods:
- Mutual information
- Correlation-based filtering
- Recursive Feature Elimination with CV (RFECV)
- Variance threshold
- L1-based selection (Lasso)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFECV,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection pipeline to reduce overfitting.

    Supports multiple selection methods:
    - mutual_info: Mutual information between features and target
    - correlation: Remove highly correlated features, keep target-correlated
    - rfecv: Recursive Feature Elimination with Cross-Validation
    - variance: Remove low-variance features
    - lasso: L1-regularization based selection

    Example:
        selector = FeatureSelector(method="mutual_info")
        selected = selector.select_features(X, y, n_features=20)
        X_selected = X[selected]
    """

    SUPPORTED_METHODS = ["mutual_info", "correlation", "rfecv", "variance", "lasso", "combined"]

    def __init__(
        self,
        method: str = "mutual_info",
        task: str = "regression",
        random_state: int = 42,
    ):
        """
        Initialize the feature selector.

        Args:
            method: Selection method (mutual_info, correlation, rfecv, variance, lasso, combined)
            task: Task type ('regression' or 'classification')
            random_state: Random seed for reproducibility
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")

        self.method = method
        self.task = task
        self.random_state = random_state
        self.selected_features: list[str] | None = None
        self.importance_scores: dict[str, float] | None = None
        self.dropped_features: list[str] | None = None

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
    ) -> list[str]:
        """
        Select top N features using the specified method.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            correlation_threshold: Max correlation between features (for correlation method)
            variance_threshold: Min variance threshold (for variance method)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using {self.method} method (target: {n_features} features)")

        # Remove any non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        logger.info(f"Starting with {len(X_numeric.columns)} numeric features")

        if self.method == "mutual_info":
            return self._select_mutual_info(X_numeric, y, n_features)
        elif self.method == "correlation":
            return self._select_correlation(X_numeric, y, n_features, correlation_threshold)
        elif self.method == "rfecv":
            return self._select_rfecv(X_numeric, y)
        elif self.method == "variance":
            return self._select_variance(X_numeric, y, n_features, variance_threshold)
        elif self.method == "lasso":
            return self._select_lasso(X_numeric, y, n_features)
        elif self.method == "combined":
            return self._select_combined(X_numeric, y, n_features, correlation_threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _select_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, n_features: int
    ) -> list[str]:
        """Select features based on mutual information with target."""
        logger.info("Computing mutual information scores...")

        # Choose MI function based on task
        if self.task == "regression":
            mi_func = mutual_info_regression
        else:
            mi_func = mutual_info_classif

        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())

        scores = mi_func(X_clean, y_clean, random_state=self.random_state)
        self.importance_scores = dict(zip(X.columns, scores, strict=True))

        # Sort and select top N
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.selected_features = [f[0] for f in sorted_features[:n_features]]
        self.dropped_features = [f[0] for f in sorted_features[n_features:]]

        logger.info(f"Selected {len(self.selected_features)} features by mutual information")
        return self.selected_features

    def _select_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        threshold: float
    ) -> list[str]:
        """
        Select features by removing highly correlated ones and keeping target-correlated.

        Two-step process:
        1. Remove features with inter-correlation > threshold
        2. Rank remaining by target correlation and select top N
        """
        logger.info(f"Removing features with inter-correlation > {threshold}")

        # Step 1: Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > threshold].tolist()
            if correlated:
                # Keep the one with higher correlation to target
                target_corrs = X[correlated + [col]].corrwith(y).abs()
                keeper = target_corrs.idxmax()
                to_drop.update(set(correlated + [col]) - {keeper})

        X_filtered = X.drop(columns=list(to_drop))
        logger.info(f"Removed {len(to_drop)} highly correlated features, {len(X_filtered.columns)} remaining")

        # Step 2: Rank by target correlation
        target_corr = X_filtered.corrwith(y).abs()
        self.importance_scores = target_corr.to_dict()

        sorted_features = target_corr.sort_values(ascending=False)
        self.selected_features = sorted_features.head(n_features).index.tolist()
        self.dropped_features = list(to_drop) + sorted_features.tail(len(sorted_features) - n_features).index.tolist()

        logger.info(f"Selected {len(self.selected_features)} features by correlation")
        return self.selected_features

    def _select_rfecv(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Select features using Recursive Feature Elimination with CV."""
        logger.info("Running Recursive Feature Elimination with CV...")

        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())

        # Choose estimator based on task
        if self.task == "regression":
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'neg_mean_squared_error'
        else:
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'accuracy'

        selector = RFECV(
            estimator,
            step=1,
            cv=5,
            scoring=scoring,
            min_features_to_select=5,
            n_jobs=-1
        )

        selector.fit(X_clean, y_clean)

        self.selected_features = X.columns[selector.support_].tolist()
        self.dropped_features = X.columns[~selector.support_].tolist()
        self.importance_scores = dict(zip(X.columns, selector.ranking_, strict=True))

        logger.info(f"RFECV selected {len(self.selected_features)} features")
        return self.selected_features

    def _select_variance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        threshold: float
    ) -> list[str]:
        """Select features by removing low-variance ones first, then rank by MI."""
        logger.info(f"Removing features with variance < {threshold}")

        # Handle NaN values
        X_clean = X.fillna(X.median())

        # Normalize to make variance comparable
        X_normalized = (X_clean - X_clean.mean()) / X_clean.std()

        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_normalized)

        # Get features that passed variance filter
        variance_mask = selector.get_support()
        X_filtered = X.loc[:, variance_mask]

        logger.info(f"Removed {(~variance_mask).sum()} low-variance features, {len(X_filtered.columns)} remaining")

        # Now apply MI selection on remaining features
        self.method = "mutual_info"
        selected = self._select_mutual_info(X_filtered, y, n_features)
        self.method = "variance"  # Reset

        return selected

    def _select_lasso(
        self, X: pd.DataFrame, y: pd.Series, n_features: int
    ) -> list[str]:
        """Select features using L1 regularization (Lasso)."""
        logger.info("Selecting features using Lasso regularization...")

        # Handle NaN values
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())

        # Normalize features for Lasso
        X_normalized = (X_clean - X_clean.mean()) / X_clean.std()

        # Fit LassoCV to find optimal alpha
        lasso = LassoCV(cv=5, random_state=self.random_state, n_jobs=-1)
        lasso.fit(X_normalized, y_clean)

        # Get feature importances (absolute coefficients)
        importance = np.abs(lasso.coef_)
        self.importance_scores = dict(zip(X.columns, importance, strict=True))

        # Select features with non-zero coefficients, up to n_features
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Filter to non-zero and limit to n_features
        non_zero = [(f, s) for f, s in sorted_features if s > 0]
        self.selected_features = [f[0] for f in non_zero[:n_features]]
        self.dropped_features = [f[0] for f in sorted_features if f[0] not in self.selected_features]

        logger.info(f"Lasso selected {len(self.selected_features)} features (alpha={lasso.alpha_:.6f})")
        return self.selected_features

    def _select_combined(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        correlation_threshold: float
    ) -> list[str]:
        """
        Combined selection: variance filter -> correlation filter -> MI ranking.

        This is the recommended approach for comprehensive feature selection.
        """
        logger.info("Running combined feature selection pipeline...")

        # Step 1: Remove low-variance features
        X_clean = X.fillna(X.median())
        X_normalized = (X_clean - X_clean.mean()) / X_clean.std()

        var_selector = VarianceThreshold(threshold=0.01)
        var_selector.fit(X_normalized)
        variance_mask = var_selector.get_support()
        X_step1 = X.loc[:, variance_mask]
        logger.info(f"Step 1 (Variance): {len(X.columns)} -> {len(X_step1.columns)}")

        # Step 2: Remove highly correlated features
        corr_matrix = X_step1.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > correlation_threshold):
                correlated = upper.index[upper[col] > correlation_threshold].tolist()
                target_corrs = X_step1[correlated + [col]].corrwith(y).abs()
                keeper = target_corrs.idxmax()
                to_drop.update(set(correlated + [col]) - {keeper})

        X_step2 = X_step1.drop(columns=list(to_drop))
        logger.info(f"Step 2 (Correlation): {len(X_step1.columns)} -> {len(X_step2.columns)}")

        # Step 3: Rank by mutual information and select top N
        if self.task == "regression":
            scores = mutual_info_regression(
                X_step2.fillna(X_step2.median()),
                y.fillna(y.median()),
                random_state=self.random_state
            )
        else:
            scores = mutual_info_classif(
                X_step2.fillna(X_step2.median()),
                y.fillna(y.median()),
                random_state=self.random_state
            )

        self.importance_scores = dict(zip(X_step2.columns, scores, strict=True))
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.selected_features = [f[0] for f in sorted_features[:n_features]]
        self.dropped_features = list(set(X.columns) - set(self.selected_features))

        logger.info(f"Step 3 (MI Ranking): Selected {len(self.selected_features)} features")
        return self.selected_features

    def get_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance scores as a DataFrame.

        Returns:
            DataFrame with feature names, scores, and selection status
        """
        if self.importance_scores is None:
            raise ValueError("No feature selection has been performed yet")

        df = pd.DataFrame([
            {
                "feature": feature,
                "importance": score,
                "selected": feature in (self.selected_features or [])
            }
            for feature, score in self.importance_scores.items()
        ])

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame to include only selected features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with only selected features
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Call select_features() first.")

        # Only include features that exist in X
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]


def quick_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 20,
    method: str = "combined",
    task: str = "regression"
) -> tuple[list[str], pd.DataFrame]:
    """
    Convenience function for quick feature selection.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_features: Number of features to select
        method: Selection method
        task: 'regression' or 'classification'

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    selector = FeatureSelector(method=method, task=task)
    selected = selector.select_features(X, y, n_features=n_features)
    importance_df = selector.get_importance_df()

    return selected, importance_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # Generate sample data with some informative and some noise features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Make some features informative
    y = (
        2 * X["feature_0"] +
        1.5 * X["feature_1"] +
        0.5 * X["feature_2"] +
        np.random.randn(n_samples) * 0.5
    )

    # Add highly correlated features
    X["feature_0_corr"] = X["feature_0"] + np.random.randn(n_samples) * 0.1

    print("=== Testing Feature Selection Methods ===\n")

    for method in ["mutual_info", "correlation", "lasso", "combined"]:
        print(f"\n--- {method.upper()} Method ---")
        selector = FeatureSelector(method=method)
        selected = selector.select_features(X, y, n_features=10)
        print(f"Selected features: {selected[:5]}...")  # Show first 5

        # Check if informative features were selected
        informative = ["feature_0", "feature_1", "feature_2"]
        found = [f for f in informative if f in selected]
        print(f"Found informative features: {found}")
