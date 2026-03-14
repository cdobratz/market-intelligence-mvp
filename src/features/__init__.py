"""
Features module - Feature engineering for financial data.

This module provides:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Time-series features (lags, rolling stats, momentum)
- Sentiment analysis (keyword-based and FinBERT transformer)
- Feature caching with Redis for efficiency
- Feature selection utilities
"""

import hashlib
import logging
import os
import pickle
from collections.abc import Callable
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import redis for caching support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.debug("redis not installed. Feature caching unavailable.")


class FeatureCache:
    """
    Redis-based feature caching to eliminate redundant computation.

    Caches computed features with a TTL (time-to-live) to avoid
    recalculating features for unchanged data on every DAG run.

    Features:
    - Content-based cache keys (hash of data + function name)
    - Configurable TTL for cache expiration
    - Graceful fallback when Redis unavailable
    - Support for any feature engineering function
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl: int = 86400,  # 24 hours
        enabled: bool = True,
    ):
        """
        Initialize the feature cache.

        Args:
            redis_url: Redis connection URL (default: redis://localhost:6379)
            ttl: Time-to-live for cached features in seconds
            enabled: Whether caching is enabled
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self.enabled = enabled and REDIS_AVAILABLE
        self._client = None
        self._connected = False

    @property
    def client(self) -> Optional["redis.Redis"]:
        """Lazy connect to Redis."""
        if not self.enabled:
            return None

        if self._client is None and REDIS_AVAILABLE:
            try:
                self._client = redis.from_url(self.redis_url)
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self._connected = False
                self._client = None

        return self._client

    def _generate_cache_key(
        self,
        df: pd.DataFrame,
        func_name: str,
        **kwargs
    ) -> str:
        """
        Generate a unique cache key based on data content and function.

        Args:
            df: Input DataFrame
            func_name: Name of the feature function
            **kwargs: Additional parameters that affect the output

        Returns:
            Unique cache key string
        """
        # Hash the DataFrame content
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

        # Include kwargs in the key
        kwargs_str = str(sorted(kwargs.items()))
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]

        return f"features:{func_name}:{data_hash}:{kwargs_hash}"

    def get_or_compute(
        self,
        df: pd.DataFrame,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Get cached features or compute and cache them.

        Args:
            df: Input DataFrame
            feature_func: Feature engineering function
            **kwargs: Additional arguments for the feature function

        Returns:
            DataFrame with computed features
        """
        func_name = feature_func.__name__

        # If caching disabled or not connected, compute directly
        if not self.enabled or self.client is None:
            logger.debug(f"Cache disabled, computing {func_name} directly")
            return feature_func(df, **kwargs)

        # Generate cache key
        cache_key = self._generate_cache_key(df, func_name, **kwargs)

        # Try to get from cache
        try:
            cached = self.client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {func_name}")
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        # Compute features
        logger.info(f"Cache miss for {func_name}, computing...")
        result = feature_func(df, **kwargs)

        # Store in cache
        try:
            self.client.setex(
                cache_key,
                self.ttl,
                pickle.dumps(result)
            )
            logger.debug(f"Cached {func_name} with TTL {self.ttl}s")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

        return result

    def invalidate(self, pattern: str = "features:*") -> int:
        """
        Invalidate cached features matching a pattern.

        Args:
            pattern: Redis key pattern to match

        Returns:
            Number of keys deleted
        """
        if not self.enabled or self.client is None:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or self.client is None:
            return {"enabled": False, "connected": False}

        try:
            info = self.client.info("memory")
            keys = self.client.keys("features:*")
            return {
                "enabled": True,
                "connected": self._connected,
                "feature_keys": len(keys),
                "memory_used": info.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}


# Global cache instance (lazy initialization)
_feature_cache: FeatureCache | None = None


def get_feature_cache() -> FeatureCache:
    """Get or create the global feature cache instance."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache


def cached_feature_engineering(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], pd.DataFrame],
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for cached feature engineering.

    Args:
        df: Input DataFrame
        feature_func: Feature engineering function
        **kwargs: Additional arguments for the feature function

    Returns:
        DataFrame with computed features
    """
    cache = get_feature_cache()
    return cache.get_or_compute(df, feature_func, **kwargs)


# Lazy imports to avoid loading heavy dependencies (transformers/FinBERT) at import time.
# Import these explicitly when needed:
#   from src.features.sentiment import SentimentAnalyzer, extract_sentiment_features
#   from src.features.technical_indicators import calculate_rsi, calculate_macd, ...
#   from src.features.timeseries import engineer_features


def __getattr__(name):
    """Lazy import for commonly used functions."""
    _sentiment_names = {
        "NewsProcessor", "SentimentAnalyzer", "extract_sentiment_features",
        "TransformerSentimentAnalyzer",
    }
    _indicator_names = {
        "calculate_all_indicators_vectorized", "calculate_atr",
        "calculate_bollinger_bands", "calculate_ema", "calculate_macd",
        "calculate_rsi", "calculate_sma",
    }

    if name in _sentiment_names:
        from src.features import sentiment
        return getattr(sentiment, name, None)
    elif name in _indicator_names:
        from src.features import technical_indicators
        return getattr(technical_indicators, name)
    elif name == "engineer_features":
        from src.features.timeseries import engineer_features
        return engineer_features
    raise AttributeError(f"module 'src.features' has no attribute {name!r}")

__all__ = [
    # Caching
    "FeatureCache",
    "get_feature_cache",
    "cached_feature_engineering",
    # Technical indicators
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_sma",
    "calculate_ema",
    "calculate_atr",
    "calculate_all_indicators_vectorized",
    # Time series
    "engineer_features",
    # Sentiment
    "SentimentAnalyzer",
    "extract_sentiment_features",
    "NewsProcessor",
]
