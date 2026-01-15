"""
Time-Series Features Module - Generate time-series based features

Provides functions to create time-series features:
- Lag features (previous n periods)
- Rolling statistics (mean, std, min, max)
- Momentum and rate of change
- Autocorrelation features
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_lag_features(
    data: pd.DataFrame,
    columns: List[str],
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features for specified columns
    
    Args:
        data: Input DataFrame
        columns: Column names to create lags for
        lags: List of lag periods (default [1, 2, 3, 5, 7])
        
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 2, 3, 5, 7]
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column {col} not found in data")
            continue
        
        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            result[feature_name] = result[col].shift(lag)
    
    return result


def create_rolling_features(
    data: pd.DataFrame,
    columns: List[str],
    windows: List[int] = None,
    functions: List[str] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Args:
        data: Input DataFrame
        columns: Column names
        windows: Window sizes (default [5, 10, 20])
        functions: Statistics to calculate (default ['mean', 'std', 'min', 'max'])
        
    Returns:
        DataFrame with rolling features
    """
    if windows is None:
        windows = [5, 10, 20]
    
    if functions is None:
        functions = ["mean", "std", "min", "max"]
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column {col} not found in data")
            continue
        
        for window in windows:
            for func in functions:
                feature_name = f"{col}_rolling_{window}_{func}"
                
                if func == "mean":
                    result[feature_name] = result[col].rolling(window=window).mean()
                elif func == "std":
                    result[feature_name] = result[col].rolling(window=window).std()
                elif func == "min":
                    result[feature_name] = result[col].rolling(window=window).min()
                elif func == "max":
                    result[feature_name] = result[col].rolling(window=window).max()
    
    return result


def create_momentum_features(
    data: pd.DataFrame,
    price_col: str = "close",
    periods: List[int] = None
) -> pd.DataFrame:
    """
    Create momentum-based features
    
    Args:
        data: Input DataFrame
        price_col: Name of price column
        periods: List of periods for calculations (default [5, 10, 20])
        
    Returns:
        DataFrame with momentum features
    """
    if periods is None:
        periods = [5, 10, 20]
    
    result = data.copy()
    
    if price_col not in result.columns:
        logger.warning(f"Column {price_col} not found")
        return result
    
    for period in periods:
        # Momentum (absolute change)
        result[f"momentum_{period}"] = result[price_col].diff(period)
        
        # Rate of change (percentage)
        result[f"roc_{period}"] = (
            (result[price_col] - result[price_col].shift(period))
            / result[price_col].shift(period) * 100
        )
        
        # Return
        result[f"return_{period}"] = result[price_col].pct_change(period)
    
    return result


def create_volatility_features(
    data: pd.DataFrame,
    price_col: str = "close",
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create volatility-based features
    
    Args:
        data: Input DataFrame
        price_col: Name of price column
        windows: Window sizes (default [5, 10, 20])
        
    Returns:
        DataFrame with volatility features
    """
    if windows is None:
        windows = [5, 10, 20]
    
    result = data.copy()
    
    if price_col not in result.columns:
        logger.warning(f"Column {price_col} not found")
        return result
    
    returns = result[price_col].pct_change()
    
    for window in windows:
        # Rolling standard deviation of returns
        result[f"volatility_{window}"] = returns.rolling(window=window).std()
        
        # Rolling coefficient of variation
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        result[f"cv_{window}"] = rolling_std / rolling_mean.abs()
        
        # High-Low range
        if "high" in result.columns and "low" in result.columns:
            hl_range = (result["high"] - result["low"]) / result[price_col]
            result[f"hl_range_{window}"] = hl_range.rolling(window=window).mean()
    
    return result


def create_price_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create price-relationship features
    
    Requires: open, high, low, close columns
    
    Args:
        data: Input DataFrame with OHLC data
        
    Returns:
        DataFrame with price features
    """
    result = data.copy()
    
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in result.columns]
    
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return result
    
    # Price position
    result["price_position"] = (
        (result["close"] - result["low"]) / (result["high"] - result["low"])
    )
    
    # Close-Open ratio
    result["co_ratio"] = result["close"] / result["open"]
    
    # High-Low ratio
    result["hl_ratio"] = result["high"] / result["low"]
    
    # Open-Close spread
    result["oc_spread"] = (result["close"] - result["open"]) / result["open"]
    
    # True range ratio
    tr = np.maximum(
        np.maximum(
            result["high"] - result["low"],
            np.abs(result["high"] - result["close"].shift())
        ),
        np.abs(result["low"] - result["close"].shift())
    )
    result["tr_ratio"] = tr / result["close"]
    
    return result


def create_volume_features(
    data: pd.DataFrame,
    volume_col: str = "volume",
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Create volume-based features
    
    Args:
        data: Input DataFrame
        volume_col: Name of volume column
        price_col: Name of price column
        
    Returns:
        DataFrame with volume features
    """
    result = data.copy()
    
    if volume_col not in result.columns or price_col not in result.columns:
        logger.warning("Missing volume or price column")
        return result
    
    # Volume moving average ratio
    vol_ma_5 = result[volume_col].rolling(window=5).mean()
    vol_ma_20 = result[volume_col].rolling(window=20).mean()
    result["volume_ma_ratio_5_20"] = vol_ma_5 / vol_ma_20
    
    # Price-Volume trend
    result["pvt"] = (result[price_col].pct_change() * result[volume_col]).cumsum()
    
    # On-Balance Volume
    obv = np.where(
        result[price_col].diff() > 0,
        result[volume_col],
        np.where(result[price_col].diff() < 0, -result[volume_col], 0)
    )
    result["obv"] = np.cumsum(obv)
    
    # Money Flow Index (simple version)
    typical_price = (result["high"] + result["low"] + result["close"]) / 3 if "high" in result.columns else result[price_col]
    money_flow = typical_price * result[volume_col]
    positive_flow = np.where(typical_price.diff() > 0, money_flow, 0)
    negative_flow = np.where(typical_price.diff() < 0, money_flow, 0)
    
    pos_mf = pd.Series(positive_flow).rolling(window=14).sum()
    neg_mf = pd.Series(negative_flow).rolling(window=14).sum()
    result["money_flow_index"] = 100 * pos_mf / (pos_mf + neg_mf)
    
    return result


def create_trend_features(
    data: pd.DataFrame,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Create trend-based features
    
    Args:
        data: Input DataFrame
        price_col: Name of price column
        
    Returns:
        DataFrame with trend features
    """
    result = data.copy()
    
    if price_col not in result.columns:
        logger.warning(f"Column {price_col} not found")
        return result
    
    # Trend direction (1 = up, -1 = down)
    result["trend_1"] = np.where(result[price_col] > result[price_col].shift(1), 1, -1)
    result["trend_5"] = np.where(result[price_col] > result[price_col].shift(5), 1, -1)
    result["trend_20"] = np.where(result[price_col] > result[price_col].shift(20), 1, -1)
    
    # Consecutive up/down days
    result["up_down"] = np.where(result[price_col].diff() > 0, 1, 0)
    result["consecutive_up"] = result["up_down"].astype(int).groupby(
        (result["up_down"] != result["up_down"].shift()).cumsum()
    ).cumsum()
    
    return result


def create_relative_features(
    data: pd.DataFrame,
    base_col: str = "close"
) -> pd.DataFrame:
    """
    Create features relative to base values
    
    Args:
        data: Input DataFrame
        base_col: Column to use as reference (default 'close')
        
    Returns:
        DataFrame with relative features
    """
    result = data.copy()
    
    if base_col not in result.columns:
        logger.warning(f"Column {base_col} not found")
        return result
    
    # Deviation from 20-day average
    ma_20 = result[base_col].rolling(window=20).mean()
    result[f"{base_col}_pct_from_ma20"] = (
        (result[base_col] - ma_20) / ma_20 * 100
    )
    
    # Deviation from 50-day average
    ma_50 = result[base_col].rolling(window=50).mean()
    result[f"{base_col}_pct_from_ma50"] = (
        (result[base_col] - ma_50) / ma_50 * 100
    )
    
    # Distance from 52-week high/low
    high_252 = result[base_col].rolling(window=252).max()
    low_252 = result[base_col].rolling(window=252).min()
    result[f"{base_col}_52w_high_pct"] = (
        (result[base_col] - high_252) / high_252 * 100
    )
    result[f"{base_col}_52w_low_pct"] = (
        (result[base_col] - low_252) / low_252 * 100
    )
    
    return result


def engineer_features(
    data: pd.DataFrame,
    feature_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Master feature engineering function
    
    Applies all feature engineering steps based on configuration
    
    Args:
        data: Input DataFrame
        feature_config: Configuration dictionary for features
        
    Returns:
        DataFrame with all engineered features
    """
    if feature_config is None:
        feature_config = {
            "lag_columns": ["close"],
            "lag_periods": [1, 2, 3, 5, 7],
            "rolling_columns": ["close"],
            "rolling_windows": [5, 10, 20],
            "rolling_functions": ["mean", "std"],
            "momentum_periods": [5, 10, 20],
            "include_volatility": True,
            "include_price_features": True,
            "include_volume_features": "volume" in data.columns,
            "include_trend_features": True,
            "include_relative_features": True,
        }
    
    result = data.copy()
    
    logger.info("Creating lag features...")
    result = create_lag_features(
        result,
        feature_config.get("lag_columns", ["close"]),
        feature_config.get("lag_periods", [1, 2, 3, 5, 7])
    )
    
    logger.info("Creating rolling features...")
    result = create_rolling_features(
        result,
        feature_config.get("rolling_columns", ["close"]),
        feature_config.get("rolling_windows", [5, 10, 20]),
        feature_config.get("rolling_functions", ["mean", "std"])
    )
    
    logger.info("Creating momentum features...")
    result = create_momentum_features(
        result,
        periods=feature_config.get("momentum_periods", [5, 10, 20])
    )
    
    if feature_config.get("include_volatility", True):
        logger.info("Creating volatility features...")
        result = create_volatility_features(
            result,
            windows=feature_config.get("rolling_windows", [5, 10, 20])
        )
    
    if feature_config.get("include_price_features", True):
        logger.info("Creating price features...")
        result = create_price_features(result)
    
    if feature_config.get("include_volume_features", False):
        logger.info("Creating volume features...")
        result = create_volume_features(result)
    
    if feature_config.get("include_trend_features", True):
        logger.info("Creating trend features...")
        result = create_trend_features(result)
    
    if feature_config.get("include_relative_features", True):
        logger.info("Creating relative features...")
        result = create_relative_features(result)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100)
    sample_data = pd.DataFrame({
        "open": np.random.uniform(100, 150, 100),
        "high": np.random.uniform(150, 160, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 150, 100),
        "volume": np.random.uniform(1000000, 10000000, 100),
    }, index=dates)
    
    # Engineer features
    features = engineer_features(sample_data)
    print(f"Original shape: {sample_data.shape}")
    print(f"With features shape: {features.shape}")
    print(f"\nFeature columns added:")
    print([col for col in features.columns if col not in sample_data.columns])
