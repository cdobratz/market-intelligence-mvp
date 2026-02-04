"""
Data Validation Module - Quality checks and schema validation for financial data

This module provides comprehensive data quality validation including:
- Schema validation for stocks, forex, crypto, and news data
- Outlier detection using IQR method
- Null value checks and statistics
- Date range and temporal consistency validation
- Data profiling and validation reports
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Index
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Define Pandera schemas for different data types

STOCK_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "open": Column(float, checks=pa.Check.ge(0), nullable=True),
        "high": Column(float, checks=pa.Check.ge(0), nullable=True),
        "low": Column(float, checks=pa.Check.ge(0), nullable=True),
        "close": Column(float, checks=pa.Check.ge(0), nullable=True),
        "volume": Column(float, checks=pa.Check.ge(0), nullable=True),
        "symbol": Column(str, checks=pa.Check.str_length(1, 5)),
    },
    index=Index(pa.DateTime, name="date"),
    strict=False,
    coerce=True,
)

FOREX_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "open": Column(float, checks=pa.Check.ge(0)),
        "high": Column(float, checks=pa.Check.ge(0)),
        "low": Column(float, checks=pa.Check.ge(0)),
        "close": Column(float, checks=pa.Check.ge(0)),
        "pair": Column(str, checks=pa.Check.str_length(6, 6)),
    },
    index=Index(pa.DateTime, name="date"),
    strict=False,
    coerce=True,
)

CRYPTO_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "price": Column(float, checks=pa.Check.ge(0)),
        "market_cap": Column(float, checks=pa.Check.ge(0), required=False),
        "volume": Column(float, checks=pa.Check.ge(0), required=False),
        "coin_id": Column(str),
    },
    index=Index(pa.DateTime, name="timestamp"),
    strict=False,
    coerce=True,
)

NEWS_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "title": Column(str),
        "description": Column(str, required=False),
        "url": Column(str),
        "source": Column(str),
        "publishedAt": Column(pa.DateTime),
    },
    strict=False,
    coerce=True,
)


class DataValidator:
    """
    Comprehensive data validation class for financial data
    
    Attributes:
        validation_results: Dictionary storing validation outcomes
        data_profiles: Dictionary storing data statistics
    """

    def __init__(self) -> None:
        """Initialize validator with empty results and profiles"""
        self.validation_results: Dict[str, Any] = {}
        self.data_profiles: Dict[str, pd.DataFrame] = {}
        self.validation_errors: Dict[str, List[str]] = {}

    def validate_stock_data(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate stock data against schema and business rules
        
        Args:
            df: Stock data DataFrame with OHLCV data
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            # Schema validation
            STOCK_DATA_SCHEMA.validate(df)
            report["checks"]["schema"] = "passed"
        except pa.errors.SchemaError as e:
            report["checks"]["schema"] = "failed"
            report["errors"].append(f"Schema validation failed: {str(e)}")

        # OHLC relationship check (High >= Low >= Low, etc.)
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            invalid_ohlc = (df["high"] < df["low"]).sum()
            if invalid_ohlc > 0:
                report["warnings"].append(f"{invalid_ohlc} rows with high < low")

        # Volume checks
        if "volume" in df.columns:
            zero_volume = (df["volume"] == 0).sum()
            if zero_volume > len(df) * 0.1:  # More than 10% zero volume
                report["warnings"].append(f"{zero_volume} rows with zero volume")

        # Null value checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            report["checks"]["nulls"] = "detected"
            report["warnings"].append(f"Null values found: {null_counts.to_dict()}")
        else:
            report["checks"]["nulls"] = "passed"

        # Outlier detection
        outliers = self._detect_outliers(df[["close", "volume"]])
        if len(outliers) > 0:
            report["checks"]["outliers"] = f"detected_{len(outliers)}"
            report["warnings"].append(f"{len(outliers)} outliers detected in price/volume")
        else:
            report["checks"]["outliers"] = "none"

        # Date continuity check
        if len(df) > 1 and hasattr(df.index, 'to_series'):
            try:
                date_diffs = df.index.to_series().diff().dt.days.value_counts()
                expected_diff = 1  # Daily data
                if expected_diff not in date_diffs.index:
                    report["warnings"].append("Missing trading days detected")
            except (AttributeError, TypeError):
                # Skip date continuity check if index is not datetime
                pass

        is_valid = len(report["errors"]) == 0
        report["is_valid"] = is_valid

        return is_valid, report

    def validate_forex_data(self, df: pd.DataFrame, pair: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate forex data
        
        Args:
            df: Forex data DataFrame
            pair: Currency pair (e.g., 'EURUSD')
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            FOREX_DATA_SCHEMA.validate(df)
            report["checks"]["schema"] = "passed"
        except pa.errors.SchemaError as e:
            report["checks"]["schema"] = "failed"
            report["errors"].append(f"Schema validation failed: {str(e)}")

        # High >= Low check
        if "high" in df.columns and "low" in df.columns:
            invalid = (df["high"] < df["low"]).sum()
            if invalid > 0:
                report["warnings"].append(f"{invalid} rows with high < low")

        # Price range reasonableness (forex typically 0.5 to 150)
        if "close" in df.columns:
            unreasonable = ((df["close"] < 0.01) | (df["close"] > 500)).sum()
            if unreasonable > 0:
                report["warnings"].append(f"{unreasonable} prices outside typical range")

        # Null checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            report["checks"]["nulls"] = "detected"
            report["warnings"].append(f"Null values found: {null_counts.to_dict()}")
        else:
            report["checks"]["nulls"] = "passed"

        is_valid = len(report["errors"]) == 0
        report["is_valid"] = is_valid

        return is_valid, report

    def validate_crypto_data(self, df: pd.DataFrame, coin_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate cryptocurrency data
        
        Args:
            df: Crypto data DataFrame
            coin_id: Coin identifier (e.g., 'bitcoin')
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "coin_id": coin_id,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            CRYPTO_DATA_SCHEMA.validate(df)
            report["checks"]["schema"] = "passed"
        except pa.errors.SchemaError as e:
            report["checks"]["schema"] = "failed"
            report["errors"].append(f"Schema validation failed: {str(e)}")

        # Price validation
        if "price" in df.columns:
            if (df["price"] <= 0).any():
                report["errors"].append("Negative or zero prices found")

            outliers = self._detect_outliers(df[["price"]])
            if len(outliers) > 0:
                report["checks"]["outliers"] = f"detected_{len(outliers)}"
            else:
                report["checks"]["outliers"] = "none"

        # Null checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            report["checks"]["nulls"] = "detected"
            report["warnings"].append(f"Null values found: {null_counts.to_dict()}")
        else:
            report["checks"]["nulls"] = "passed"

        is_valid = len(report["errors"]) == 0
        report["is_valid"] = is_valid

        return is_valid, report

    def validate_news_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate news data
        
        Args:
            df: News data DataFrame
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "data_type": "news",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check required columns
        required = ["title", "url", "publishedAt"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            report["errors"].append(f"Missing required columns: {missing}")

        # Schema validation
        try:
            NEWS_DATA_SCHEMA.validate(df)
            report["checks"]["schema"] = "passed"
        except pa.errors.SchemaError as e:
            report["checks"]["schema"] = "failed"
            report["errors"].append(f"Schema validation failed: {str(e)}")

        # Content checks
        if "title" in df.columns:
            empty_titles = (df["title"].str.len() == 0).sum()
            if empty_titles > 0:
                report["warnings"].append(f"{empty_titles} rows with empty titles")

        # Date checks
        if "publishedAt" in df.columns:
            try:
                dates = pd.to_datetime(df["publishedAt"])
                future_dates = (dates > datetime.now()).sum()
                if future_dates > 0:
                    report["warnings"].append(f"{future_dates} articles with future dates")
            except Exception as e:
                report["errors"].append(f"Date parsing error: {str(e)}")

        # Null checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            report["checks"]["nulls"] = "detected"
            report["warnings"].append(f"Null values: {null_counts.to_dict()}")
        else:
            report["checks"]["nulls"] = "passed"

        is_valid = len(report["errors"]) == 0
        report["is_valid"] = is_valid

        return is_valid, report

    @staticmethod
    def _detect_outliers(
        df: pd.DataFrame, 
        method: str = "iqr", 
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers using IQR method
        
        Args:
            df: DataFrame with numeric columns
            method: Detection method ('iqr' or 'zscore')
            threshold: IQR multiplier (default 1.5)
            
        Returns:
            Array of boolean values indicating outliers
        """
        if method == "iqr":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1).values
            return np.where(outliers)[0]
        
        elif method == "zscore":
            z_scores = np.abs((df - df.mean()) / df.std())
            outliers = (z_scores > 3).any(axis=1).values
            return np.where(outliers)[0]
        
        return np.array([])

    def get_data_profile(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Generate statistical profile of data
        
        Args:
            df: Input DataFrame
            name: Profile name
            
        Returns:
            DataFrame with statistics
        """
        profile = {
            "column": [],
            "dtype": [],
            "non_null_count": [],
            "null_count": [],
            "null_percentage": [],
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "unique_values": [],
        }

        for col in df.columns:
            profile["column"].append(col)
            profile["dtype"].append(str(df[col].dtype))
            profile["non_null_count"].append(df[col].notna().sum())
            profile["null_count"].append(df[col].isna().sum())
            profile["null_percentage"].append(
                (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
            )

            if pd.api.types.is_numeric_dtype(df[col]):
                profile["min"].append(df[col].min())
                profile["max"].append(df[col].max())
                profile["mean"].append(df[col].mean())
                profile["std"].append(df[col].std())
            else:
                profile["min"].append(None)
                profile["max"].append(None)
                profile["mean"].append(None)
                profile["std"].append(None)

            profile["unique_values"].append(df[col].nunique())

        profile_df = pd.DataFrame(profile)
        self.data_profiles[name] = profile_df
        return profile_df

    def generate_validation_report(
        self, 
        validation_reports: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            validation_reports: List of validation reports
            output_path: Optional path to save report as JSON
            
        Returns:
            Summary validation report
        """
        total_checks = len(validation_reports)
        passed_checks = sum(1 for r in validation_reports if r.get("is_valid", False))

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_validations": total_checks,
            "passed": passed_checks,
            "failed": total_checks - passed_checks,
            "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            "total_warnings": sum(len(r.get("warnings", [])) for r in validation_reports),
            "total_errors": sum(len(r.get("errors", [])) for r in validation_reports),
            "details": validation_reports,
        }

        if output_path:
            import json
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_path}")

        return summary

    def validate_date_range(
        self,
        df: pd.DataFrame,
        date_column: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Validate data falls within expected date range
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            start_date: Minimum expected date
            end_date: Maximum expected date
            
        Returns:
            Validation report
        """
        report = {
            "check": "date_range",
            "valid": True,
            "issues": [],
        }

        if date_column not in df.columns:
            report["valid"] = False
            report["issues"].append(f"Date column '{date_column}' not found")
            return report

        try:
            dates = pd.to_datetime(df[date_column])
        except Exception as e:
            report["valid"] = False
            report["issues"].append(f"Cannot parse dates: {str(e)}")
            return report

        if start_date and (dates < pd.Timestamp(start_date)).any():
            report["issues"].append(f"Data before start_date: {start_date}")
            report["valid"] = False

        if end_date and (dates > pd.Timestamp(end_date)).any():
            report["issues"].append(f"Data after end_date: {end_date}")
            report["valid"] = False

        return report

    def check_data_completeness(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        min_rows: int = 0
    ) -> Dict[str, Any]:
        """
        Check if data meets completeness requirements
        
        Args:
            df: Input DataFrame
            required_columns: List of required columns
            min_rows: Minimum number of rows required
            
        Returns:
            Completeness check report
        """
        report = {
            "check": "completeness",
            "valid": True,
            "issues": [],
            "row_count": len(df),
            "column_count": len(df.columns),
        }

        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            report["valid"] = False
            report["issues"].append(f"Missing columns: {missing_cols}")

        # Check minimum rows
        if len(df) < min_rows:
            report["valid"] = False
            report["issues"].append(f"Insufficient rows: {len(df)} < {min_rows}")

        return report


def validate_pipeline_data(
    data_path: Path,
    data_type: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate data from pipeline
    
    Args:
        data_path: Path to Parquet files
        data_type: Type of data (stocks, forex, crypto, news)
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    validator = DataValidator()
    validation_reports = []

    try:
        # Find all parquet files
        parquet_files = list(data_path.glob("**/*.parquet"))
        
        if not parquet_files:
            return False, {"error": f"No parquet files found in {data_path}"}

        for file_path in parquet_files:
            df = pd.read_parquet(file_path)

            if data_type == "stocks":
                is_valid, report = validator.validate_stock_data(df, "unknown")
            elif data_type == "forex":
                is_valid, report = validator.validate_forex_data(df, "unknown")
            elif data_type == "crypto":
                is_valid, report = validator.validate_crypto_data(df, "unknown")
            elif data_type == "news":
                is_valid, report = validator.validate_news_data(df)
            else:
                return False, {"error": f"Unknown data type: {data_type}"}

            validation_reports.append(report)

        # Generate summary
        summary = validator.generate_validation_report(validation_reports)
        overall_valid = all(r.get("is_valid", False) for r in validation_reports)

        return overall_valid, summary

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100)
    sample_stock = pd.DataFrame({
        "open": np.random.uniform(100, 150, 100),
        "high": np.random.uniform(150, 160, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 150, 100),
        "volume": np.random.uniform(1000000, 10000000, 100),
        "symbol": "AAPL",
    }, index=dates)

    # Validate
    validator = DataValidator()
    is_valid, report = validator.validate_stock_data(sample_stock, "AAPL")
    print(f"Validation passed: {is_valid}")
    print(f"Report: {report}")
    
    # Get profile
    profile = validator.get_data_profile(sample_stock, "sample_stock")
    print(f"\nData Profile:\n{profile}")
