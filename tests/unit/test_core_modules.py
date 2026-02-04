"""
Unit Tests - Core Modules

Comprehensive unit tests for:
1. DataValidator.validate_stock_data() - Stock data validation with error/warning reporting
2. TechnicalIndicators.calculate_rsi() - RSI computation with value range and short data handling
3. FeatureEngineering.engineer_features() - Feature column addition without data alteration
4. SentimentAnalyzer.analyze_sentiment() - Sentiment classification into positive/negative/neutral
5. DataProcessor.get_processor() - Correct processor instance selection including 'auto'
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.validation import DataValidator
from data.processing import get_processor, PandasProcessor, FireducksProcessor, DataProcessor
from features.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
from features.timeseries import engineer_features
from features.sentiment import SentimentAnalyzer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_stock_data():
    """Create sample stock OHLCV data with valid characteristics"""
    dates = pd.date_range("2024-01-01", periods=100, name="date")
    data = pd.DataFrame({
        "open": np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
        "high": np.linspace(105, 155, 100) + np.random.normal(0, 1, 100),
        "low": np.linspace(95, 145, 100) + np.random.normal(0, 1, 100),
        "close": np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
        "volume": np.random.uniform(1000000, 10000000, 100),
        "symbol": "AAPL",
    }, index=dates)
    
    # Ensure proper OHLC relationships
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    
    return data


@pytest.fixture
def short_price_data():
    """Create short price series (less than indicator period)"""
    dates = pd.date_range("2024-01-01", periods=5)
    return pd.Series(
        [100.0, 101.5, 99.8, 102.3, 101.0],
        index=dates,
        name="close"
    )


@pytest.fixture
def long_price_data():
    """Create long price series for reliable indicator calculation"""
    dates = pd.date_range("2024-01-01", periods=100)
    return pd.Series(
        np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
        index=dates,
        name="close"
    )


@pytest.fixture
def sample_news_data():
    """Create sample news data for sentiment analysis"""
    return pd.DataFrame({
        "title": [
            "Stock market surges as bullish sentiment takes hold",
            "Market crashes after negative earnings report",
            "Trading session with mixed signals",
            "Tech stocks rally on positive growth outlook",
            "Economic data shows weak performance"
        ],
        "description": ["desc"] * 5,
        "url": [f"http://example.com/{i}" for i in range(5)],
        "source": ["Reuters", "Reuters", "AP", "Bloomberg", "Reuters"],
        "publishedAt": pd.date_range("2024-01-01", periods=5),
    })


# ============================================================================
# Test Case 1: DataValidator.validate_stock_data()
# ============================================================================

class TestDataValidatorStockData:
    """Test DataValidator.validate_stock_data() for validation and error/warning reporting"""
    
    def test_validate_valid_stock_data(self, sample_stock_data):
        """Test that valid stock data passes validation"""
        validator = DataValidator()
        is_valid, report = validator.validate_stock_data(sample_stock_data, "AAPL")
        
        assert is_valid is True
        assert report["is_valid"] is True
        assert report["symbol"] == "AAPL"
        assert report["checks"]["schema"] == "passed"
        assert report["checks"]["nulls"] == "passed"
        assert len(report["errors"]) == 0
    
    def test_validate_detects_null_values(self, sample_stock_data):
        """Test that validation detects null values and reports warnings"""
        validator = DataValidator()
        sample_stock_data.loc[sample_stock_data.index[0], "close"] = np.nan
        sample_stock_data.loc[sample_stock_data.index[1], "volume"] = np.nan
        
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        # Nulls are warnings, not errors
        assert is_valid is True
        assert report["checks"]["nulls"] == "detected"
        assert len(report["warnings"]) > 0
        assert any("Null" in w for w in report["warnings"])
    
    def test_validate_detects_high_low_violation(self, sample_stock_data):
        """Test that validation detects when high < low"""
        validator = DataValidator()
        sample_stock_data.loc[sample_stock_data.index[10], "high"] = 50.0
        sample_stock_data.loc[sample_stock_data.index[10], "low"] = 100.0
        
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        assert is_valid is True  # Still valid (not an error)
        assert len(report["warnings"]) > 0
        assert any("high < low" in w for w in report["warnings"])
    
    def test_validate_detects_zero_volume(self, sample_stock_data):
        """Test that validation detects excessive zero volume"""
        validator = DataValidator()
        # Set >10% of rows to zero volume
        sample_stock_data.iloc[0:16, sample_stock_data.columns.get_loc("volume")] = 0.0
        
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        assert is_valid is True
        assert len(report["warnings"]) > 0
        assert any("zero volume" in w for w in report["warnings"])
    
    def test_validate_detects_outliers(self, sample_stock_data):
        """Test that validation detects outliers in price/volume"""
        validator = DataValidator()
        sample_stock_data.loc[sample_stock_data.index[5], "close"] = 1000.0  # Extreme outlier
        
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        assert is_valid is True
        assert "outliers" in report["checks"]
        assert "detected" in report["checks"]["outliers"]
        assert len(report["warnings"]) > 0
    
    def test_validate_schema_validation_failure(self, sample_stock_data):
        """Test that invalid schema types are caught or coerced"""
        validator = DataValidator()
        # Remove required column to cause schema failure
        sample_stock_data_invalid = sample_stock_data.drop(columns=["symbol"])
        
        is_valid, report = validator.validate_stock_data(sample_stock_data_invalid, "TEST")
        
        # Schema error is an error (hard failure) when required column missing
        assert is_valid is False
        assert report["checks"]["schema"] == "failed"
        assert len(report["errors"]) > 0
    
    def test_validate_missing_date_index(self, sample_stock_data):
        """Test validation with non-datetime index"""
        validator = DataValidator()
        sample_stock_data.reset_index(inplace=True)
        sample_stock_data.index = range(len(sample_stock_data))
        
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        # Should still validate data content
        assert "is_valid" in report
    
    def test_validate_report_contains_timestamp(self, sample_stock_data):
        """Test that validation report includes timestamp"""
        validator = DataValidator()
        before = datetime.now()
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        after = datetime.now()
        
        assert "timestamp" in report
        report_time = datetime.fromisoformat(report["timestamp"])
        assert before <= report_time <= after
    
    def test_validate_returns_tuple(self, sample_stock_data):
        """Test that validate_stock_data returns (bool, dict) tuple"""
        validator = DataValidator()
        result = validator.validate_stock_data(sample_stock_data, "TEST")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)


# ============================================================================
# Test Case 2: TechnicalIndicators.calculate_rsi()
# ============================================================================

class TestTechnicalIndicatorsRSI:
    """Test calculate_rsi() for value range and short data handling"""
    
    def test_rsi_value_range(self, long_price_data):
        """Test that RSI values are within valid range [0, 100]"""
        rsi = calculate_rsi(long_price_data, period=14)
        
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()
    
    def test_rsi_no_nan_values(self, long_price_data):
        """Test that RSI has no NaN values (due to fillna)"""
        rsi = calculate_rsi(long_price_data, period=14, fillna=True)
        
        assert rsi.isnull().sum() == 0
    
    def test_rsi_length_matches_input(self, long_price_data):
        """Test that RSI output length matches input length"""
        rsi = calculate_rsi(long_price_data, period=14)
        
        assert len(rsi) == len(long_price_data)
    
    def test_rsi_short_data_handling(self, short_price_data):
        """Test that RSI handles data shorter than period"""
        rsi = calculate_rsi(short_price_data, period=14, fillna=True)
        
        # Should return all NaN or neutral values
        assert len(rsi) == len(short_price_data)
        assert isinstance(rsi, pd.Series)
    
    def test_rsi_uptrend_vs_downtrend(self):
        """Test that RSI distinguishes between uptrend and downtrend"""
        # Strong uptrend
        uptrend = pd.Series(range(100, 200))
        rsi_up = calculate_rsi(uptrend, period=14)
        
        # Strong downtrend
        downtrend = pd.Series(range(200, 100, -1))
        rsi_down = calculate_rsi(downtrend, period=14)
        
        # Later values should show opposite trends
        assert rsi_up.iloc[-1] > rsi_down.iloc[-1]
    
    def test_rsi_neutral_market(self):
        """Test RSI in neutral/sideways market"""
        # Oscillating prices
        neutral = pd.Series([100, 101, 100, 101, 100, 101] * 10)
        rsi = calculate_rsi(neutral, period=14, fillna=True)
        
        # Should be near neutral (50)
        assert rsi.mean() > 40
        assert rsi.mean() < 60
    
    def test_rsi_custom_period(self, long_price_data):
        """Test RSI with custom period parameter"""
        rsi_7 = calculate_rsi(long_price_data, period=7)
        rsi_21 = calculate_rsi(long_price_data, period=21)
        
        assert len(rsi_7) == len(long_price_data)
        assert len(rsi_21) == len(long_price_data)
        # Different periods should produce different values
        assert not rsi_7.equals(rsi_21)
    
    def test_rsi_fillna_parameter(self, long_price_data):
        """Test RSI with fillna enabled and disabled"""
        rsi_filled = calculate_rsi(long_price_data, period=14, fillna=True)
        rsi_unfilled = calculate_rsi(long_price_data, period=14, fillna=False)
        
        assert rsi_filled.isnull().sum() == 0
        assert rsi_unfilled.isnull().sum() > 0
    
    def test_rsi_preserves_index(self, long_price_data):
        """Test that RSI preserves the input series index"""
        rsi = calculate_rsi(long_price_data, period=14)
        
        assert rsi.index.equals(long_price_data.index)


# ============================================================================
# Test Case 3: FeatureEngineering.engineer_features()
# ============================================================================

class TestFeatureEngineeringEngineerFeatures:
    """Test engineer_features() for adding features without altering original data"""
    
    def test_engineer_features_adds_columns(self, sample_stock_data):
        """Test that engineer_features adds new feature columns"""
        original_cols = set(sample_stock_data.columns)
        features = engineer_features(sample_stock_data)
        new_cols = set(features.columns)
        
        # Should have added features
        assert len(new_cols) > len(original_cols)
        assert original_cols.issubset(new_cols)
    
    def test_engineer_features_preserves_original_data(self, sample_stock_data):
        """Test that original data columns are unchanged"""
        original_data = sample_stock_data.copy()
        features = engineer_features(sample_stock_data)
        
        # Original columns should have identical values
        for col in original_data.columns:
            pd.testing.assert_series_equal(
                features[col].astype(original_data[col].dtype),
                original_data[col],
                check_exact=False,
                rtol=1e-10
            )
    
    def test_engineer_features_row_count_unchanged(self, sample_stock_data):
        """Test that feature engineering doesn't change number of rows"""
        features = engineer_features(sample_stock_data)
        
        assert len(features) == len(sample_stock_data)
    
    def test_engineer_features_index_preserved(self, sample_stock_data):
        """Test that index is preserved after feature engineering"""
        features = engineer_features(sample_stock_data)
        
        assert features.index.equals(sample_stock_data.index)
    
    def test_engineer_features_creates_lag_features(self, sample_stock_data):
        """Test that lag features are created"""
        features = engineer_features(sample_stock_data)
        
        lag_features = [col for col in features.columns if "lag" in col]
        assert len(lag_features) > 0
    
    def test_engineer_features_creates_rolling_features(self, sample_stock_data):
        """Test that rolling window features are created"""
        features = engineer_features(sample_stock_data)
        
        rolling_features = [col for col in features.columns if "rolling" in col]
        assert len(rolling_features) > 0
    
    def test_engineer_features_creates_momentum_features(self, sample_stock_data):
        """Test that momentum features are created"""
        features = engineer_features(sample_stock_data)
        
        momentum_features = [col for col in features.columns if "momentum" in col or "roc" in col]
        assert len(momentum_features) > 0
    
    def test_engineer_features_creates_trend_features(self, sample_stock_data):
        """Test that trend features are created"""
        features = engineer_features(sample_stock_data)
        
        trend_features = [col for col in features.columns if "trend" in col]
        assert len(trend_features) > 0
    
    def test_engineer_features_volume_features(self, sample_stock_data):
        """Test that volume features are created when volume column exists"""
        features = engineer_features(sample_stock_data)
        
        volume_features = [col for col in features.columns if "volume" in col.lower() or "obv" in col]
        # Should have volume features since volume column exists
        assert len(volume_features) > 0
    
    def test_engineer_features_price_features(self, sample_stock_data):
        """Test that price relationship features are created"""
        features = engineer_features(sample_stock_data)
        
        price_features = [col for col in features.columns if "price" in col or "ratio" in col]
        assert len(price_features) > 0
    
    def test_engineer_features_custom_config(self, sample_stock_data):
        """Test engineer_features with custom configuration"""
        config = {
            "lag_columns": ["close"],
            "lag_periods": [1, 2],  # Only 2 lags
            "rolling_columns": ["close"],
            "rolling_windows": [5],  # Only 5-day window
            "include_volatility": False,
            "include_price_features": True,
            "include_volume_features": False,
            "include_trend_features": False,
        }
        
        features = engineer_features(sample_stock_data, config)
        
        # Should have lag and rolling features
        assert any("lag" in col for col in features.columns)
        # Should not have some features
        assert not any("volatility" in col for col in features.columns)
    
    def test_engineer_features_handles_missing_optional_columns(self):
        """Test engineer_features with missing optional columns"""
        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104] * 20,
        }, index=pd.date_range("2024-01-01", periods=100))
        
        # Should not fail even without OHLC
        features = engineer_features(data)
        
        assert len(features) == len(data)
        assert "close" in features.columns
    
    def test_engineer_features_returns_dataframe(self, sample_stock_data):
        """Test that engineer_features returns a DataFrame"""
        features = engineer_features(sample_stock_data)
        
        assert isinstance(features, pd.DataFrame)


# ============================================================================
# Test Case 4: SentimentAnalyzer.analyze_sentiment()
# ============================================================================

class TestSentimentAnalyzerSentiment:
    """Test SentimentAnalyzer.analyze_sentiment() for sentiment classification"""
    
    def test_sentiment_positive_text(self):
        """Test that clearly positive text is classified as positive"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("bullish surge gain profit positive")
        
        assert result["sentiment_label"] == "positive"
        assert result["positive"] > 0
        assert result["sentiment_score"] > 0.1
    
    def test_sentiment_negative_text(self):
        """Test that clearly negative text is classified as negative"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("bearish crash loss failure negative")
        
        assert result["sentiment_label"] == "negative"
        assert result["negative"] > 0
        assert result["sentiment_score"] < -0.1
    
    def test_sentiment_neutral_text(self):
        """Test that neutral text is classified as neutral"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("trading session with mixed signals")
        
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] >= -0.1
        assert result["sentiment_score"] <= 0.1
    
    def test_sentiment_empty_text(self):
        """Test sentiment analysis on empty text"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("")
        
        assert result["sentiment_label"] == "neutral"
        assert result["neutral"] == 1.0
        assert result["sentiment_score"] == 0.0
    
    def test_sentiment_null_text(self):
        """Test sentiment analysis on NaN/None text"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment(None)
        
        assert result["sentiment_label"] == "neutral"
    
    def test_sentiment_returns_dict(self):
        """Test that analyze_sentiment returns a dictionary"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("bullish")
        
        assert isinstance(result, dict)
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "sentiment_score" in result
        assert "sentiment_label" in result
    
    def test_sentiment_score_range(self):
        """Test that sentiment_score is in range [-1, 1]"""
        analyzer = SentimentAnalyzer()
        
        positive_result = analyzer.analyze_sentiment("bullish gain rally")
        negative_result = analyzer.analyze_sentiment("crash loss bearish")
        neutral_result = analyzer.analyze_sentiment("normal trading")
        
        for result in [positive_result, negative_result, neutral_result]:
            assert -1 <= result["sentiment_score"] <= 1
    
    def test_sentiment_ratio_sum(self):
        """Test that positive + negative ratios sum reasonably"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("bullish gain surge")
        
        total_ratio = result["positive"] + result["negative"]
        if total_ratio > 0:
            assert 0 <= result["positive"] <= 1
            assert 0 <= result["negative"] <= 1
    
    def test_sentiment_case_insensitive(self):
        """Test that sentiment analysis is case-insensitive"""
        analyzer = SentimentAnalyzer()
        
        result_lower = analyzer.analyze_sentiment("bullish gain positive")
        result_upper = analyzer.analyze_sentiment("BULLISH GAIN POSITIVE")
        
        assert result_lower["sentiment_label"] == result_upper["sentiment_label"]
    
    def test_sentiment_mixed_text(self):
        """Test sentiment on mixed positive/negative text"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("bullish gain but also crash loss")
        
        # Should be neutral-ish since both sentiments are present
        assert isinstance(result["sentiment_label"], str)
        assert result["sentiment_label"] in ["positive", "negative", "neutral"]
    
    def test_sentiment_analyze_texts_method(self, sample_news_data):
        """Test analyze_texts method with multiple texts"""
        analyzer = SentimentAnalyzer()
        results_df = analyzer.analyze_texts(sample_news_data["title"].tolist())
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(sample_news_data)
        assert "sentiment_label" in results_df.columns


# ============================================================================
# Test Case 5: DataProcessor.get_processor()
# ============================================================================

class TestDataProcessorGetProcessor:
    """Test get_processor() for correct instance selection"""
    
    def test_get_processor_pandas(self):
        """Test get_processor with 'pandas' backend"""
        processor = get_processor("pandas")
        
        assert isinstance(processor, PandasProcessor)
        assert processor.backend == "pandas"
    
    def test_get_processor_fireducks(self):
        """Test get_processor with 'fireducks' backend"""
        processor = get_processor("fireducks")
        
        assert isinstance(processor, DataProcessor)
        # Will fallback to pandas if fireducks not available
        assert processor.backend in ["fireducks", "pandas"]
    
    def test_get_processor_auto(self):
        """Test get_processor with 'auto' backend"""
        processor = get_processor("auto")
        
        assert isinstance(processor, DataProcessor)
        assert processor.backend in ["pandas", "fireducks"]
    
    def test_get_processor_auto_selects_fireducks_if_available(self):
        """Test that 'auto' selects fireducks when available"""
        processor = get_processor("auto")
        
        # Should be consistent with availability
        from data.processing import FIREDUCKS_AVAILABLE
        if FIREDUCKS_AVAILABLE:
            assert processor.backend == "fireducks"
        else:
            assert processor.backend == "pandas"
    
    def test_get_processor_invalid_backend(self):
        """Test get_processor with invalid backend defaults to pandas"""
        processor = get_processor("invalid_backend")
        
        assert isinstance(processor, DataProcessor)
        assert processor.backend == "pandas"
    
    def test_get_processor_returns_dataprocessor(self):
        """Test that get_processor returns DataProcessor instance"""
        processors = [
            get_processor("pandas"),
            get_processor("fireducks"),
            get_processor("auto"),
        ]
        
        for processor in processors:
            assert isinstance(processor, DataProcessor)
    
    def test_processor_has_backend_attribute(self):
        """Test that returned processor has backend attribute"""
        processor = get_processor("auto")
        
        assert hasattr(processor, "backend")
        assert isinstance(processor.backend, str)
    
    def test_processor_repr(self):
        """Test processor string representation"""
        processor = get_processor("pandas")
        
        repr_str = str(processor)
        assert "DataProcessor" in repr_str
        assert "pandas" in repr_str
    
    def test_get_processor_consistency(self):
        """Test that multiple calls with same backend return equivalent processors"""
        processor1 = get_processor("pandas")
        processor2 = get_processor("pandas")
        
        assert processor1.backend == processor2.backend
    
    def test_processor_instance_types(self):
        """Test that get_processor returns correct instance types"""
        pandas_proc = get_processor("pandas")
        assert isinstance(pandas_proc, PandasProcessor)
        
        # Fireducks returns FireducksProcessor if available, else PandasProcessor
        fireducks_proc = get_processor("fireducks")
        from data.processing import FIREDUCKS_AVAILABLE
        if FIREDUCKS_AVAILABLE:
            assert isinstance(fireducks_proc, FireducksProcessor)
        else:
            # Falls back but backend should be set
            assert fireducks_proc.backend in ["pandas", "fireducks"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules"""
    
    def test_validation_and_feature_engineering(self, sample_stock_data):
        """Test validation followed by feature engineering"""
        validator = DataValidator()
        is_valid, report = validator.validate_stock_data(sample_stock_data, "TEST")
        
        if is_valid:
            features = engineer_features(sample_stock_data)
            assert len(features) == len(sample_stock_data)
    
    def test_processor_with_validation(self, sample_stock_data):
        """Test processor validation method"""
        processor = get_processor("auto")
        report = processor.validate_data(sample_stock_data, "stocks")
        
        assert isinstance(report, dict)
    
    def test_processor_with_feature_engineering(self, sample_stock_data):
        """Test processor feature engineering"""
        processor = get_processor("auto")
        features = processor.engineer_features(sample_stock_data)
        
        assert len(features.columns) > len(sample_stock_data.columns)
    
    def test_rsi_with_engineered_features(self, sample_stock_data):
        """Test RSI calculation on engineered features"""
        features = engineer_features(sample_stock_data)
        
        # Calculate RSI on close price
        rsi = calculate_rsi(features["close"], period=14)
        
        assert len(rsi) == len(features)
        assert (rsi >= 0).all() and (rsi <= 100).all()
    
    def test_sentiment_with_news_processing(self, sample_news_data):
        """Test sentiment analysis with full news data"""
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_texts(sample_news_data["title"].tolist())
        
        assert len(results) == len(sample_news_data)
        assert "sentiment_label" in results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
