"""
Integration Tests - Feature Engineering Pipeline

Tests the complete flow from raw data to engineered features,
including validation, feature engineering, and benchmarking.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.validation import DataValidator
from data.processing import get_processor, PandasProcessor
from features import (
    calculate_rsi,
    calculate_macd,
    engineer_features,
    extract_sentiment_features,
    NewsProcessor,
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock OHLCV data"""
    dates = pd.date_range("2024-01-01", periods=100)
    return pd.DataFrame({
        "open": np.random.uniform(100, 150, 100),
        "high": np.random.uniform(150, 160, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 150, 100),
        "volume": np.random.uniform(1000000, 10000000, 100),
        "symbol": "AAPL",
    }, index=dates)


@pytest.fixture
def sample_news_data():
    """Create sample news data"""
    return pd.DataFrame({
        "title": [
            "Stock market surges as bullish sentiment takes hold",
            "Market crashes after negative earnings report",
            "Tech stocks rally on positive growth",
        ],
        "description": ["desc1", "desc2", "desc3"],
        "url": ["http://example.com/1", "http://example.com/2", "http://example.com/3"],
        "source": ["Reuters", "Reuters", "Bloomberg"],
        "publishedAt": pd.date_range("2024-01-01", periods=3),
    })


class TestDataValidation:
    """Test data validation module"""
    
    def test_validate_stock_data_pass(self, sample_stock_data):
        """Test valid stock data passes validation"""
        validator = DataValidator()
        is_valid, report = validator.validate_stock_data(sample_stock_data, "AAPL")
        
        assert is_valid
        assert report["is_valid"]
        assert report["checks"]["schema"] == "passed"
        assert report["checks"]["nulls"] == "passed"
    
    def test_validate_stock_data_with_nulls(self, sample_stock_data):
        """Test validation detects null values"""
        sample_stock_data.loc[0, "close"] = np.nan
        
        validator = DataValidator()
        is_valid, report = validator.validate_stock_data(sample_stock_data, "AAPL")
        
        assert is_valid  # Still valid (nulls are warning, not error)
        assert report["checks"]["nulls"] == "detected"
        assert len(report["warnings"]) > 0
    
    def test_data_profiling(self, sample_stock_data):
        """Test data profiling generates statistics"""
        validator = DataValidator()
        profile = validator.get_data_profile(sample_stock_data, "test")
        
        assert profile is not None
        assert "column" in profile.columns
        assert "mean" in profile.columns
        assert "std" in profile.columns
        assert len(profile) == len(sample_stock_data.columns)
    
    def test_outlier_detection(self, sample_stock_data):
        """Test outlier detection works"""
        validator = DataValidator()
        
        # Add outlier
        sample_stock_data.loc[0, "close"] = 1000.0
        
        outliers = validator._detect_outliers(sample_stock_data[["close"]])
        assert len(outliers) > 0


class TestFeatureEngineering:
    """Test feature engineering module"""
    
    def test_rsi_calculation(self, sample_stock_data):
        """Test RSI indicator calculation"""
        rsi = calculate_rsi(sample_stock_data["close"], period=14)
        
        assert len(rsi) == len(sample_stock_data)
        assert rsi.isnull().sum() == 0  # No nulls due to fillna
        assert (rsi >= 0).all() and (rsi <= 100).all()  # RSI range
    
    def test_macd_calculation(self, sample_stock_data):
        """Test MACD indicator calculation"""
        macd, signal, hist = calculate_macd(sample_stock_data["close"])
        
        assert len(macd) == len(sample_stock_data)
        assert len(signal) == len(sample_stock_data)
        assert len(hist) == len(sample_stock_data)
    
    def test_engineer_features(self, sample_stock_data):
        """Test complete feature engineering pipeline"""
        original_cols = len(sample_stock_data.columns)
        
        features = engineer_features(sample_stock_data)
        
        # Should have more columns after engineering
        assert len(features.columns) > original_cols
        
        # Check for specific feature types
        feature_names = features.columns.tolist()
        assert any("lag" in col for col in feature_names)
        assert any("rolling" in col for col in feature_names)
        assert any("momentum" in col for col in feature_names)


class TestSentimentAnalysis:
    """Test sentiment analysis module"""
    
    def test_sentiment_extraction(self, sample_news_data):
        """Test sentiment extraction from news"""
        processed = extract_sentiment_features(sample_news_data)
        
        # Check sentiment columns added
        assert "sentiment_positive" in processed.columns
        assert "sentiment_negative" in processed.columns
        assert "sentiment_sentiment_score" in processed.columns
        
        # Check sentiment scores are in valid range
        assert (processed["sentiment_sentiment_score"] >= -1).all()
        assert (processed["sentiment_sentiment_score"] <= 1).all()
    
    def test_news_processor(self, sample_news_data):
        """Test complete news processing"""
        processor = NewsProcessor()
        processed, aggregated = processor.process_news_data(sample_news_data)
        
        assert len(processed) == len(sample_news_data)
        assert "sentiment" in aggregated.columns[0] or len(aggregated) > 0


class TestDataProcessing:
    """Test data processing layer"""
    
    def test_pandas_processor(self, sample_stock_data):
        """Test Pandas processor"""
        processor = PandasProcessor()
        
        assert processor.backend == "pandas"
        assert str(processor) == "DataProcessor(backend='pandas')"
    
    def test_get_processor_auto(self):
        """Test auto processor selection"""
        processor = get_processor("auto")
        assert processor is not None
        assert processor.backend in ["pandas", "fireducks"]
    
    def test_processor_data_validation(self, sample_stock_data):
        """Test processor validation"""
        processor = PandasProcessor()
        report = processor.validate_data(sample_stock_data, "stocks")
        
        assert "checks" in report or "error" in report
    
    def test_processor_feature_engineering(self, sample_stock_data):
        """Test processor feature engineering"""
        processor = PandasProcessor()
        features = processor.engineer_features(sample_stock_data)
        
        assert len(features.columns) > len(sample_stock_data.columns)


class TestPipeline:
    """Integration tests for complete pipeline"""
    
    def test_stock_data_pipeline(self, sample_stock_data):
        """Test complete stock data pipeline"""
        processor = PandasProcessor()
        
        # Validate
        validation_report = processor.validate_data(sample_stock_data, "stocks")
        assert validation_report.get("is_valid") or "checks" in validation_report
        
        # Engineer features
        features = processor.engineer_features(sample_stock_data)
        assert features is not None
        assert len(features) == len(sample_stock_data)
    
    def test_pipeline_with_file_io(self, sample_stock_data):
        """Test pipeline with file I/O"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save data
            data_path = tmpdir / "sample.parquet"
            sample_stock_data.to_parquet(data_path)
            
            # Load and process
            processor = PandasProcessor()
            loaded_data = processor.load_data(str(data_path))
            
            assert len(loaded_data) == len(sample_stock_data)
            
            # Engineer features
            features = processor.engineer_features(loaded_data)
            
            # Save features
            features_path = tmpdir / "features.parquet"
            features.to_parquet(features_path)
            
            assert features_path.exists()
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with synthetic data"""
        # Create data
        dates = pd.date_range("2024-01-01", periods=250)
        df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": np.cumsum(np.random.randn(250)) + 100,
            "high": np.cumsum(np.random.randn(250)) + 105,
            "low": np.cumsum(np.random.randn(250)) + 95,
            "close": np.cumsum(np.random.randn(250)) + 100,
            "volume": np.random.uniform(1000000, 10000000, 250),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        
        # Validate
        validator = DataValidator()
        is_valid, _ = validator.validate_stock_data(df, "TEST")
        assert is_valid
        
        # Engineer features
        features = engineer_features(df)
        assert len(features) > len(df.columns)
        
        # Check no NaNs in key columns
        assert features[["open", "high", "low", "close", "volume"]].isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
