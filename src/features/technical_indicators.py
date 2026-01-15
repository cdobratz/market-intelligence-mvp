"""
Technical Indicators Module - Calculate financial technical indicators

Provides functions to calculate various technical indicators:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Average True Range (ATR)
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(
    data: pd.Series,
    period: int = 14,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    Args:
        data: Price series (typically close prices)
        period: Number of periods for calculation (default 14)
        fillna: Whether to fill NaN values (default True)
        
    Returns:
        Series with RSI values
    """
    if len(data) < period:
        logger.warning(f"Data length {len(data)} < period {period}")
        return pd.Series([np.nan] * len(data), index=data.index)
    
    # Calculate price changes
    deltas = data.diff()
    
    # Separate gains and losses
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    
    # Calculate average gain and loss
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    if fillna:
        rsi = rsi.fillna(50)  # Fill with neutral value
    
    return rsi


def calculate_macd(
    data: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    fillna: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages.
    
    Args:
        data: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        fillna: Whether to fill NaN values
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    if fillna:
        macd_line = macd_line.fillna(0)
        signal_line = signal_line.fillna(0)
        histogram = histogram.fillna(0)
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
    fillna: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Bollinger Bands are volatility bands placed above and below a moving average.
    
    Args:
        data: Price series
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2.0)
        fillna: Whether to fill NaN values
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    # Middle band (SMA)
    middle_band = data.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = data.rolling(window=period).std()
    
    # Upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    if fillna:
        upper_band = upper_band.fillna(data)
        middle_band = middle_band.fillna(data)
        lower_band = lower_band.fillna(data)
    
    return upper_band, middle_band, lower_band


def calculate_sma(
    data: pd.Series,
    period: int = 20,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    SMA is the unweighted mean of the previous n data points.
    
    Args:
        data: Price series
        period: Number of periods (default 20)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with SMA values
    """
    sma = data.rolling(window=period).mean()
    
    if fillna:
        sma = sma.fillna(data)
    
    return sma


def calculate_ema(
    data: pd.Series,
    period: int = 20,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    EMA gives more weight to recent prices.
    
    Args:
        data: Price series
        period: Number of periods (default 20)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with EMA values
    """
    ema = data.ewm(span=period, adjust=False).mean()
    
    if fillna:
        ema = ema.fillna(data)
    
    return ema


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    ATR measures volatility, not price direction.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Number of periods (default 14)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with ATR values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    if fillna:
        atr = atr.fillna(tr.mean())
    
    return atr


def calculate_momentum(
    data: pd.Series,
    period: int = 10,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Momentum indicator
    
    Momentum measures the rate of change in price.
    
    Args:
        data: Price series
        period: Number of periods (default 10)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with momentum values
    """
    momentum = data.diff(period)
    
    if fillna:
        momentum = momentum.fillna(0)
    
    return momentum


def calculate_roc(
    data: pd.Series,
    period: int = 12,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Rate of Change (ROC)
    
    ROC measures the percentage change in price over a period.
    
    Args:
        data: Price series
        period: Number of periods (default 12)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with ROC values (percentage)
    """
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    
    if fillna:
        roc = roc.fillna(0)
    
    return roc


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    fillna: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Stochastic compares a closing price to price range over time.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
        smooth_k: Smoothing period for %K (default 3)
        smooth_d: Smoothing period for %D (default 3)
        fillna: Whether to fill NaN values
        
    Returns:
        Tuple of (%K, %D) series
    """
    # Calculate lowest low and highest high
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    # Calculate %K
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Smooth %K
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    
    # Calculate %D (SMA of %K)
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    
    if fillna:
        k_smooth = k_smooth.fillna(50)
        d_smooth = d_smooth.fillna(50)
    
    return k_smooth, d_smooth


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Williams %R
    
    Williams %R is a momentum indicator similar to Stochastic.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with Williams %R values
    """
    # Calculate highest high and lowest low
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate %R
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    if fillna:
        williams_r = williams_r.fillna(-50)
    
    return williams_r


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    fillna: bool = True
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX)
    
    ADX measures trend strength regardless of direction.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Number of periods (default 14)
        fillna: Whether to fill NaN values
        
    Returns:
        Series with ADX values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Plus and Minus DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    if fillna:
        adx = adx.fillna(20)  # Neutral value
    
    return adx


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample price data
    dates = pd.date_range("2024-01-01", periods=100)
    prices = pd.Series(
        np.random.uniform(100, 150, 100),
        index=dates,
        name="close"
    )
    highs = prices + np.random.uniform(0, 5, 100)
    lows = prices - np.random.uniform(0, 5, 100)
    
    # Calculate indicators
    rsi = calculate_rsi(prices, period=14)
    macd, signal, hist = calculate_macd(prices)
    upper, middle, lower = calculate_bollinger_bands(prices)
    sma = calculate_sma(prices, period=20)
    ema = calculate_ema(prices, period=20)
    
    print("RSI Sample:")
    print(rsi.tail())
    print("\nMACD Sample:")
    print(macd.tail())
    print("\nBollinger Bands Sample:")
    print(pd.concat([upper, middle, lower], axis=1).tail())
