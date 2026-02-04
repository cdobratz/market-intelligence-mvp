"""
Sample data generator for model training.

This module generates synthetic financial market data for training ML models.
It creates realistic OHLCV (Open, High, Low, Close, Volume) data with trends,
seasonality, and noise, then applies the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.features.timeseries import engineer_features
from src.features.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticMarketDataGenerator:
    """Generate synthetic financial market data for model training."""

    def __init__(
        self,
        start_price: float = 100.0,
        n_days: int = 1000,
        trend: float = 0.0001,
        volatility: float = 0.02,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize the data generator.

        Args:
            start_price: Starting price for the synthetic asset
            n_days: Number of trading days to generate
            trend: Daily trend component (drift)
            volatility: Daily volatility (standard deviation of returns)
            random_seed: Random seed for reproducibility
        """
        self.start_price = start_price
        self.n_days = n_days
        self.trend = trend
        self.volatility = volatility
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_price_series(self) -> pd.Series:
        """
        Generate a synthetic price series using geometric Brownian motion.

        Returns:
            Series of synthetic prices
        """
        # Generate random returns
        returns = np.random.normal(self.trend, self.volatility, self.n_days)

        # Calculate cumulative returns
        cumulative_returns = np.exp(np.cumsum(returns))

        # Calculate prices
        prices = self.start_price * cumulative_returns

        return pd.Series(prices)

    def generate_ohlcv_data(
        self, symbol: str = "SYN", start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.

        Args:
            symbol: Symbol for the synthetic asset
            start_date: Starting date for the data

        Returns:
            DataFrame with OHLCV columns
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.n_days)

        # Generate close prices
        close_prices = self.generate_price_series()

        # Generate date range (business days only)
        dates = pd.date_range(start=start_date, periods=self.n_days, freq="B")

        # Generate OHLC from close prices
        # High: close + random positive amount
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, self.n_days)))

        # Low: close - random positive amount
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, self.n_days)))

        # Open: previous close with small random variation
        open_prices = close_prices.shift(1).fillna(self.start_price) * (
            1 + np.random.normal(0, 0.003, self.n_days)
        )

        # Ensure price relationships are valid (High >= Close >= Low, etc.)
        high_prices = np.maximum(high_prices, close_prices)
        low_prices = np.minimum(low_prices, close_prices)
        high_prices = np.maximum(high_prices, open_prices)
        low_prices = np.minimum(low_prices, open_prices)

        # Generate volume (correlated with price volatility)
        base_volume = 1000000
        volume = base_volume * (1 + np.abs(np.random.normal(0, 0.3, self.n_days)))

        # Create DataFrame
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume.astype(int),
            }
        )

        return df

    def generate_multi_asset_data(
        self, symbols: list[str], correlations: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data for multiple assets with optional correlations.

        Args:
            symbols: List of asset symbols
            correlations: Correlation matrix for asset returns (optional)

        Returns:
            DataFrame with data for all assets
        """
        n_assets = len(symbols)

        if correlations is None:
            # Generate random correlation matrix
            correlations = self._generate_correlation_matrix(n_assets)

        # Generate correlated returns
        returns = self._generate_correlated_returns(n_assets, correlations)

        # Generate OHLCV for each asset
        all_data = []
        for i, symbol in enumerate(symbols):
            # Use the generated returns for this asset
            asset_returns = returns[:, i]

            # Calculate prices from returns
            prices = self.start_price * np.exp(np.cumsum(asset_returns))

            # Create OHLCV data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=self.n_days),
                periods=self.n_days,
                freq="B",
            )

            close_prices = pd.Series(prices)
            high_prices = close_prices * (
                1 + np.abs(np.random.normal(0, 0.005, self.n_days))
            )
            low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, self.n_days)))
            open_prices = close_prices.shift(1).fillna(self.start_price) * (
                1 + np.random.normal(0, 0.003, self.n_days)
            )

            # Ensure valid price relationships
            high_prices = np.maximum(high_prices, close_prices)
            low_prices = np.minimum(low_prices, close_prices)
            high_prices = np.maximum(high_prices, open_prices)
            low_prices = np.minimum(low_prices, open_prices)

            # Generate volume
            volume = 1000000 * (1 + np.abs(np.random.normal(0, 0.3, self.n_days)))

            df = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volume.astype(int),
                }
            )

            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

    def _generate_correlation_matrix(self, n: int) -> np.ndarray:
        """Generate a random valid correlation matrix."""
        # Generate random matrix
        A = np.random.randn(n, n)

        # Make it symmetric positive semidefinite
        corr = np.dot(A, A.T)

        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / d[:, None] / d[None, :]

        return corr

    def _generate_correlated_returns(self, n_assets: int, corr_matrix: np.ndarray) -> np.ndarray:
        """Generate correlated returns using Cholesky decomposition."""
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)

        # Generate uncorrelated returns
        uncorrelated = np.random.normal(self.trend, self.volatility, (self.n_days, n_assets))

        # Apply correlation
        correlated = np.dot(uncorrelated, L.T)

        return correlated


def generate_training_data(
    output_dir: str = "data/processed",
    n_samples: int = 1000,
    n_symbols: int = 5,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate complete training and test datasets with engineered features.

    Args:
        output_dir: Directory to save the generated data
        n_samples: Number of time steps to generate
        n_symbols: Number of assets to generate
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_df, test_df) with engineered features
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_symbols} symbols")

    # Create generator
    generator = SyntheticMarketDataGenerator(
        start_price=100.0,
        n_days=n_samples,
        trend=0.0002,  # Slight upward trend
        volatility=0.02,  # 2% daily volatility
        random_seed=42,
    )

    # Generate symbols
    symbols = [f"ASSET{i+1}" for i in range(n_symbols)]

    # Generate multi-asset data
    raw_data = generator.generate_multi_asset_data(symbols)

    logger.info(f"Generated raw data shape: {raw_data.shape}")

    # Engineer features for each symbol
    all_features = []
    for symbol in symbols:
        symbol_data = raw_data[raw_data["symbol"] == symbol].copy()
        symbol_data = symbol_data.sort_values("date").reset_index(drop=True)

        # Apply feature engineering
        try:
            featured_data = engineer_features(symbol_data)
            logger.info(f"Engineered features for {symbol}: {featured_data.shape}")
            all_features.append(featured_data)
        except Exception as e:
            logger.error(f"Error engineering features for {symbol}: {e}")
            continue

    # Combine all features
    complete_data = pd.concat(all_features, ignore_index=True)

    # Remove rows with NaN (from lagged features)
    complete_data = complete_data.dropna()

    logger.info(f"Complete data shape after feature engineering: {complete_data.shape}")

    # Create target variable: next day's return
    complete_data = complete_data.sort_values(["symbol", "date"])
    complete_data["target_return"] = complete_data.groupby("symbol")["close"].pct_change().shift(-1)

    # Create target labels for classification (Up/Down/Neutral)
    def classify_return(ret: float) -> str:
        if pd.isna(ret):
            return "Neutral"
        elif ret > 0.005:  # > 0.5%
            return "Up"
        elif ret < -0.005:  # < -0.5%
            return "Down"
        else:
            return "Neutral"

    complete_data["target_direction"] = complete_data["target_return"].apply(classify_return)

    # Remove last row for each symbol (no target)
    complete_data = complete_data.groupby("symbol").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Split by date (time-series aware split)
    complete_data = complete_data.sort_values("date")
    split_idx = int(len(complete_data) * train_ratio)

    train_df = complete_data.iloc[:split_idx].copy()
    test_df = complete_data.iloc[split_idx:].copy()

    logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train_data.parquet"
    test_path = output_path / "test_data.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved test data to {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    # Generate sample data
    train_df, test_df = generate_training_data(
        output_dir="data/processed",
        n_samples=1000,
        n_symbols=5,
        train_ratio=0.8,
    )

    print("\n=== Training Data Summary ===")
    print(train_df.info())
    print(f"\nTarget distribution:\n{train_df['target_direction'].value_counts()}")

    print("\n=== Test Data Summary ===")
    print(test_df.info())
    print(f"\nTarget distribution:\n{test_df['target_direction'].value_counts()}")
