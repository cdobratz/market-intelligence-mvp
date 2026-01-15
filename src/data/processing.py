"""
Data Processing Layer - Unified interface for Pandas and Fireducks

Provides factory pattern for switching between processing backends:
- Pandas (baseline)
- Fireducks (optimized)

Both backends provide identical API for seamless switching.
"""

from typing import Literal, Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import Fireducks
try:
    import fireducks.pandas as fd
    FIREDUCKS_AVAILABLE = True
except ImportError:
    FIREDUCKS_AVAILABLE = False


class DataProcessor:
    """
    Base class for data processing
    
    Defines unified interface for feature engineering pipeline
    """
    
    def __init__(self, backend: Literal["pandas", "fireducks"] = "pandas"):
        """
        Initialize processor
        
        Args:
            backend: Processing backend ('pandas' or 'fireducks')
        """
        self.backend = backend
        self._validate_backend()
    
    def _validate_backend(self) -> None:
        """Validate backend availability"""
        if self.backend == "fireducks" and not FIREDUCKS_AVAILABLE:
            logger.warning(
                f"Fireducks not available, falling back to pandas. "
                f"Install with: pip install fireducks"
            )
            self.backend = "pandas"
    
    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet file
        
        Args:
            path: Path to Parquet file
            **kwargs: Additional arguments
            
        Returns:
            DataFrame
        """
        if self.backend == "fireducks":
            import fireducks.pandas as fd
            return fd.read_parquet(path, **kwargs)
        else:
            return pd.read_parquet(path, **kwargs)
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Engineer features on data
        
        Args:
            df: Input DataFrame
            config: Feature configuration
            
        Returns:
            DataFrame with engineered features
        """
        from ..features import engineer_features as eng_features
        
        return eng_features(df, config)
    
    def validate_data(
        self,
        df: pd.DataFrame,
        data_type: str
    ) -> Dict[str, Any]:
        """
        Validate data quality
        
        Args:
            df: Input DataFrame
            data_type: Type of data (stocks, forex, crypto, news)
            
        Returns:
            Validation report
        """
        from .validation import DataValidator
        
        validator = DataValidator()
        
        if data_type == "stocks":
            return validator.validate_stock_data(df, "unknown")[1]
        elif data_type == "forex":
            return validator.validate_forex_data(df, "unknown")[1]
        elif data_type == "crypto":
            return validator.validate_crypto_data(df, "unknown")[1]
        elif data_type == "news":
            return validator.validate_news_data(df)[1]
        else:
            return {"error": f"Unknown data type: {data_type}"}
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DataProcessor(backend='{self.backend}')"


class PandasProcessor(DataProcessor):
    """Pandas-based data processor"""
    
    def __init__(self):
        """Initialize Pandas processor"""
        super().__init__(backend="pandas")


class FireducksProcessor(DataProcessor):
    """Fireducks-based data processor"""
    
    def __init__(self):
        """Initialize Fireducks processor"""
        if not FIREDUCKS_AVAILABLE:
            logger.warning("Fireducks not available, using Pandas instead")
        super().__init__(backend="fireducks")


def get_processor(
    backend: Literal["pandas", "fireducks", "auto"] = "auto"
) -> DataProcessor:
    """
    Get data processor instance
    
    Factory function for creating appropriate processor
    
    Args:
        backend: Backend choice ('pandas', 'fireducks', or 'auto')
                'auto' selects fireducks if available, otherwise pandas
                
    Returns:
        DataProcessor instance
    """
    if backend == "auto":
        if FIREDUCKS_AVAILABLE:
            return FireducksProcessor()
        else:
            return PandasProcessor()
    elif backend == "fireducks":
        return FireducksProcessor()
    elif backend == "pandas":
        return PandasProcessor()
    else:
        logger.warning(f"Unknown backend '{backend}', defaulting to pandas")
        return PandasProcessor()


# Convenience functions

def load_and_validate(
    path: str,
    data_type: str,
    backend: str = "auto"
) -> tuple:
    """
    Load and validate data in one step
    
    Args:
        path: Path to Parquet file
        data_type: Type of data
        backend: Processing backend
        
    Returns:
        Tuple of (data, validation_report)
    """
    processor = get_processor(backend)
    
    logger.info(f"Loading data using {processor.backend}...")
    data = processor.load_data(path)
    
    logger.info("Validating data...")
    report = processor.validate_data(data, data_type)
    
    return data, report


def load_and_engineer(
    path: str,
    config: Optional[Dict[str, Any]] = None,
    backend: str = "auto"
) -> pd.DataFrame:
    """
    Load and engineer features in one step
    
    Args:
        path: Path to Parquet file
        config: Feature configuration
        backend: Processing backend
        
    Returns:
        DataFrame with engineered features
    """
    processor = get_processor(backend)
    
    logger.info(f"Loading data using {processor.backend}...")
    data = processor.load_data(path)
    
    logger.info("Engineering features...")
    features = processor.engineer_features(data, config)
    
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    logger.info("Creating sample data...")
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "symbol": "AAPL",
        "open": range(100, 200),
        "high": range(105, 205),
        "low": range(95, 195),
        "close": range(100, 200),
        "volume": range(1000000, 1000000 + 100),
    })
    
    # Test with Pandas
    logger.info("\nTesting Pandas processor...")
    pandas_proc = PandasProcessor()
    logger.info(f"Processor: {pandas_proc}")
    
    # Test with Fireducks (if available)
    if FIREDUCKS_AVAILABLE:
        logger.info("\nTesting Fireducks processor...")
        fireducks_proc = FireducksProcessor()
        logger.info(f"Processor: {fireducks_proc}")
    
    # Test auto-selection
    logger.info("\nTesting auto processor...")
    auto_proc = get_processor("auto")
    logger.info(f"Auto processor: {auto_proc}")
