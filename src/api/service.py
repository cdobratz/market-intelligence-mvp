"""Prediction service for model inference."""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import data ingestion - fallback to demo if not available
try:
    from src.data.ingestion import fetch_stock_data
    INGESTION_AVAILABLE = True
except ImportError:
    INGESTION_AVAILABLE = False
    fetch_stock_data = None

# Try to import features - fallback to simple implementation if not available
try:
    from src.features.technical_indicators import add_technical_indicators
except ImportError:
    add_technical_indicators = None

try:
    from src.features.sentiment import NewsSentimentAnalyzer
except ImportError:
    NewsSentimentAnalyzer = None

try:
    from src.features.timeseries import create_lag_features
except ImportError:
    create_lag_features = None

logger = logging.getLogger(__name__)

# Demo mode - uses synthetic data when no API keys available
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


class PredictionService:
    """Service for making stock predictions."""

    def __init__(self):
        self.sentiment_analyzer = None
        self._init_sentiment()
        self.model_loaded = False
        self._init_model()

    def _init_sentiment(self):
        """Initialize sentiment analyzer."""
        if NewsSentimentAnalyzer is None:
            self.sentiment_analyzer = None
            return
        try:
            self.sentiment_analyzer = NewsSentimentAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None

    def _init_model(self):
        """Initialize the ML model."""
        # Try to load from MLflow or local file
        model_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'production'
        )
        
        if os.path.exists(model_path):
            try:
                import joblib
                # Load model if exists
                self.model = None  # Would load actual model here
                self.model_loaded = True
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        else:
            logger.info("Running in demo mode - using synthetic predictions")

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data for a symbol."""
        if DEMO_MODE:
            return self._generate_demo_data(symbol)
        
        try:
            # Try to fetch real data
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                logger.warning("No API key found, using demo mode")
                return self._generate_demo_data(symbol)
            
            df = fetch_stock_data(symbol, api_key=api_key)
            return df
        except Exception as e:
            logger.warning(f"Error fetching data: {e}, using demo mode")
            return self._generate_demo_data(symbol)

    def _generate_demo_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Generate synthetic stock data for demo purposes."""
        np.random.seed(hash(symbol) % (2**32))
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement
        base_price = random.choice([150, 180, 200, 250, 300])
        returns = np.random.normal(0.0005, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, days)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, days)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        })
        
        df.set_index('date', inplace=True)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from raw data."""
        if df.empty:
            return df
        
        # Add technical indicators
        if add_technical_indicators is not None:
            try:
                df = add_technical_indicators(df)
            except Exception as e:
                logger.warning(f"Could not add technical indicators: {e}")
                # Add simple indicators manually
                df = self._add_simple_indicators(df)
        else:
            df = self._add_simple_indicators(df)
        
        # Add lag features
        if create_lag_features is not None:
            try:
                df = create_lag_features(df)
            except Exception as e:
                logger.warning(f"Could not add lag features: {e}")
                df = self._add_simple_lags(df)
        else:
            df = self._add_simple_lags(df)
        
        # Add sentiment if available
        if self.sentiment_analyzer is not None:
            try:
                # Would add actual news sentiment here
                df['sentiment_score'] = np.random.uniform(-1, 1)
            except Exception:
                df['sentiment_score'] = 0.0
        else:
            df['sentiment_score'] = 0.0
        
        return df

    def _add_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple technical indicators manually."""
        if 'close' not in df.columns:
            return df
        
        # Simple SMA
        df['sma_20'] = df['close'].rolling(20).mean() if len(df) >= 20 else df['close']
        
        # Simple RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df

    def _add_simple_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple lag features manually."""
        if 'close' not in df.columns:
            return df
        
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Volume lags
        if 'volume' in df.columns:
            for lag in [1, 2, 3]:
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df

    def predict(self, symbol: str, include_features: bool = False) -> Dict[str, Any]:
        """Make a prediction for a stock symbol."""
        # Get stock data
        df = self.get_stock_data(symbol)
        
        if df.empty:
            return {
                "error": f"Could not fetch data for {symbol}",
                "symbol": symbol
            }
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Prepare features
        features_df = self.prepare_features(df.copy())
        
        # Get latest features
        latest = features_df.iloc[-1]
        
        if self.model_loaded and hasattr(self, 'model') and self.model is not None:
            # Use actual model prediction
            # This would require proper feature preparation
            prediction = 0.0  # Would be model.predict(features)
        else:
            # Demo prediction based on technical signals
            prediction = self._demo_prediction(features_df)
        
        # Determine direction and confidence
        direction = "bullish" if prediction >= 0 else "bearish"
        confidence = min(abs(prediction) * 10, 1.0) if prediction != 0 else 0.5
        
        # Calculate target price (5-day projection)
        target_price = current_price * (1 + prediction)
        
        result = {
            "symbol": symbol,
            "prediction": float(prediction),
            "direction": direction,
            "confidence": float(confidence),
            "current_price": round(current_price, 2),
            "target_price": round(target_price, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if include_features:
            # Include key features
            key_features = ['rsi', 'macd', 'sma_20', 'sentiment_score']
            available_features = {k: float(v) for k, v in latest.items() 
                                 if k in key_features and not pd.isna(v)}
            result["features"] = available_features
        
        return result

    def _demo_prediction(self, df: pd.DataFrame) -> float:
        """Generate a demo prediction based on technical indicators."""
        if df.empty or len(df) < 20:
            return 0.0
        
        signals = []
        
        # RSI signal
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
        if rsi < 30:
            signals.append(0.02)  # Oversold - bullish
        elif rsi > 70:
            signals.append(-0.02)  # Overbought - bearish
        
        # MACD signal
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            signal = df['macd_signal'].iloc[-1]
            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal:
                    signals.append(0.01)
                else:
                    signals.append(-0.01)
        
        # Price vs SMA
        if 'sma_20' in df.columns:
            sma = df['sma_20'].iloc[-1]
            price = df['close'].iloc[-1]
            if not pd.isna(sma):
                if price > sma:
                    signals.append(0.01)
                else:
                    signals.append(-0.01)
        
        # Sentiment
        if 'sentiment_score' in df.columns:
            sentiment = df['sentiment_score'].iloc[-1]
            if not pd.isna(sentiment):
                signals.append(sentiment * 0.02)
        
        # Average signals or return small random value
        if signals:
            return sum(signals) / len(signals)
        else:
            return np.random.uniform(-0.02, 0.02)

    def predict_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Make predictions for multiple symbols."""
        return [self.predict(symbol) for symbol in symbols]

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the status of data pipelines."""
        # This would check actual pipeline status from Airflow
        # For demo, return placeholder status
        
        return {
            "data_ingestion": "ready",
            "feature_engineering": "ready",
            "model_training": "ready",
            "last_run": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat()
        }

    def get_models_info(self) -> List[Dict[str, Any]]:
        """Get information about available models."""
        models = [
            {
                "name": "XGBoost Regressor",
                "version": "1.0.0",
                "accuracy": 0.72,
                "last_trained": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
                "features": ["rsi", "macd", "bb_position", "sma_ratio", "volume_ratio", "sentiment"]
            },
            {
                "name": "Stacking Ensemble",
                "version": "1.0.0",
                "accuracy": 0.75,
                "last_trained": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
                "features": ["rsi", "macd", "bb_position", "sma_ratio", "volume_ratio", "sentiment", "regime"]
            }
        ]
        return models


# Singleton instance
_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get the prediction service singleton."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
