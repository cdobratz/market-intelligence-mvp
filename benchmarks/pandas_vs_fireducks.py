"""
Pandas vs Fireducks Benchmark Suite

Comprehensive performance comparison between Pandas and Fireducks
for financial data processing operations.

Tests data loading, transformations, aggregations, and merges
on datasets of varying sizes (100K, 1M, 10M rows).
"""

import time
import psutil
import os
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import Fireducks
try:
    import fireducks.pandas as fd
    FIREDUCKS_AVAILABLE = True
except ImportError:
    FIREDUCKS_AVAILABLE = False
    logger.warning("Fireducks not installed. Install with: pip install fireducks")


class BenchmarkConfig:
    """Configuration for benchmark tests"""
    
    DATASET_SIZES = {
        "small": 100_000,
        "medium": 1_000_000,
        "large": 10_000_000,
    }
    
    OPERATIONS = [
        "load_parquet",
        "groupby_aggregation",
        "rolling_window",
        "merge_operation",
        "feature_engineering",
    ]


class ProcessMonitor:
    """Monitor memory and CPU usage during operations"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0
        self.peak_memory = 0
        self.start_time = 0
    
    def start(self) -> None:
        """Start monitoring"""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.start_time = time.time()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)
        return current
    
    def stop(self) -> Tuple[float, float, float]:
        """
        Stop monitoring and return metrics
        
        Returns:
            Tuple of (elapsed_time, peak_memory_mb, memory_increase_mb)
        """
        elapsed = time.time() - self.start_time
        current_memory = self.get_memory_usage()
        memory_increase = current_memory - self.start_memory
        
        return elapsed, self.peak_memory, memory_increase


class DataGenerator:
    """Generate synthetic financial data for benchmarking"""
    
    @staticmethod
    def generate_stock_data(
        num_rows: int,
        num_symbols: int = 10,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic stock OHLCV data
        
        Args:
            num_rows: Number of rows
            num_symbols: Number of stock symbols
            seed: Random seed
            
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(seed)
        
        # Create date index
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_rows // num_symbols + 10)
        dates = pd.date_range(start=start_date, periods=num_rows // num_symbols, freq='D')
        dates = np.repeat(dates, num_symbols)[:num_rows]
        
        # Create symbols
        symbols = [f"SYM{i:03d}" for i in range(num_symbols)]
        symbols = np.tile(symbols, num_rows // num_symbols + 1)[:num_rows]
        
        # Generate OHLCV data
        base_price = np.random.uniform(100, 200, num_rows)
        
        data = pd.DataFrame({
            "date": dates,
            "symbol": symbols,
            "open": base_price,
            "high": base_price + np.random.uniform(0, 5, num_rows),
            "low": base_price - np.random.uniform(0, 5, num_rows),
            "close": base_price + np.random.uniform(-2, 2, num_rows),
            "volume": np.random.uniform(1_000_000, 10_000_000, num_rows),
        })
        
        return data
    
    @staticmethod
    def generate_time_series_data(
        num_rows: int,
        num_assets: int = 5,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate time-series data with multiple assets
        
        Args:
            num_rows: Number of rows
            num_assets: Number of assets
            seed: Random seed
            
        Returns:
            DataFrame with time-series data
        """
        np.random.seed(seed)
        
        dates = pd.date_range(start="2023-01-01", periods=num_rows, freq='H')
        
        data = pd.DataFrame({
            "timestamp": dates,
            "asset_id": np.random.choice(range(num_assets), num_rows),
            "price": np.random.uniform(100, 200, num_rows),
            "volume": np.random.uniform(1000, 10000, num_rows),
            "returns": np.random.normal(0.0001, 0.01, num_rows),
        })
        
        return data


class BenchmarkSuite:
    """Main benchmark suite for Pandas vs Fireducks comparison"""
    
    def __init__(self, output_dir: Path = Path("benchmarks/results")):
        """
        Initialize benchmark suite
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.data_generator = DataGenerator()
    
    def benchmark_load_parquet(
        self,
        df: pd.DataFrame,
        size_name: str,
        framework: str = "pandas"
    ) -> Dict[str, float]:
        """
        Benchmark Parquet loading
        
        Args:
            df: Data to save and reload
            size_name: Size category name
            framework: 'pandas' or 'fireducks'
            
        Returns:
            Benchmark metrics
        """
        # Save to Parquet
        temp_path = self.output_dir / f"temp_{size_name}.parquet"
        df.to_parquet(temp_path)
        
        monitor = ProcessMonitor()
        monitor.start()
        
        # Load with target framework
        if framework == "fireducks" and FIREDUCKS_AVAILABLE:
            loaded_df = fd.read_parquet(temp_path)
        else:
            loaded_df = pd.read_parquet(temp_path)
        
        elapsed, peak_mem, mem_increase = monitor.stop()
        
        # Cleanup
        temp_path.unlink()
        
        return {
            "operation": "load_parquet",
            "size": size_name,
            "framework": framework,
            "rows": len(df),
            "columns": len(df.columns),
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": mem_increase,
        }
    
    def benchmark_groupby_aggregation(
        self,
        df: pd.DataFrame,
        size_name: str,
        framework: str = "pandas"
    ) -> Dict[str, float]:
        """
        Benchmark groupby and aggregation
        
        Args:
            df: Input data
            size_name: Size category name
            framework: 'pandas' or 'fireducks'
            
        Returns:
            Benchmark metrics
        """
        # Convert to target framework if needed
        if framework == "fireducks" and FIREDUCKS_AVAILABLE:
            df = fd.DataFrame(df)
        
        monitor = ProcessMonitor()
        monitor.start()
        
        # Groupby aggregation
        result = df.groupby("symbol").agg({
            "close": ["mean", "std", "min", "max"],
            "volume": ["mean", "sum"],
        })
        
        elapsed, peak_mem, mem_increase = monitor.stop()
        
        return {
            "operation": "groupby_aggregation",
            "size": size_name,
            "framework": framework,
            "rows": len(df),
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": mem_increase,
        }
    
    def benchmark_rolling_window(
        self,
        df: pd.DataFrame,
        size_name: str,
        framework: str = "pandas"
    ) -> Dict[str, float]:
        """
        Benchmark rolling window calculations
        
        Args:
            df: Input data
            size_name: Size category name
            framework: 'pandas' or 'fireducks'
            
        Returns:
            Benchmark metrics
        """
        # Convert to target framework if needed
        if framework == "fireducks" and FIREDUCKS_AVAILABLE:
            df = fd.DataFrame(df)
        
        monitor = ProcessMonitor()
        monitor.start()
        
        # Rolling window operations
        df["rolling_mean_5"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=5).mean()
        )
        df["rolling_std_20"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=20).std()
        )
        
        elapsed, peak_mem, mem_increase = monitor.stop()
        
        return {
            "operation": "rolling_window",
            "size": size_name,
            "framework": framework,
            "rows": len(df),
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": mem_increase,
        }
    
    def benchmark_merge_operation(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        size_name: str,
        framework: str = "pandas"
    ) -> Dict[str, float]:
        """
        Benchmark merge operations
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            size_name: Size category name
            framework: 'pandas' or 'fireducks'
            
        Returns:
            Benchmark metrics
        """
        # Convert to target framework if needed
        if framework == "fireducks" and FIREDUCKS_AVAILABLE:
            df1 = fd.DataFrame(df1)
            df2 = fd.DataFrame(df2)
        
        monitor = ProcessMonitor()
        monitor.start()
        
        # Merge operation
        result = pd.merge(df1, df2, on="symbol", how="inner")
        
        elapsed, peak_mem, mem_increase = monitor.stop()
        
        return {
            "operation": "merge_operation",
            "size": size_name,
            "framework": framework,
            "rows_df1": len(df1),
            "rows_df2": len(df2),
            "result_rows": len(result),
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": mem_increase,
        }
    
    def benchmark_feature_engineering(
        self,
        df: pd.DataFrame,
        size_name: str,
        framework: str = "pandas"
    ) -> Dict[str, float]:
        """
        Benchmark comprehensive feature engineering
        
        Args:
            df: Input data
            size_name: Size category name
            framework: 'pandas' or 'fireducks'
            
        Returns:
            Benchmark metrics
        """
        # Convert to target framework if needed
        if framework == "fireducks" and FIREDUCKS_AVAILABLE:
            df = fd.DataFrame(df)
        
        monitor = ProcessMonitor()
        monitor.start()
        
        # Feature engineering pipeline
        df_copy = df.copy()
        
        # Lags
        df_copy["close_lag_1"] = df_copy.groupby("symbol")["close"].shift(1)
        df_copy["close_lag_5"] = df_copy.groupby("symbol")["close"].shift(5)
        
        # Rolling statistics
        df_copy["close_ma_20"] = df_copy.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=20).mean()
        )
        df_copy["close_std_20"] = df_copy.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=20).std()
        )
        
        # Momentum
        df_copy["returns"] = df_copy.groupby("symbol")["close"].pct_change()
        df_copy["momentum_5"] = df_copy.groupby("symbol")["close"].transform(
            lambda x: x.diff(5)
        )
        
        elapsed, peak_mem, mem_increase = monitor.stop()
        
        return {
            "operation": "feature_engineering",
            "size": size_name,
            "framework": framework,
            "rows": len(df),
            "features_created": 6,
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": mem_increase,
        }
    
    def run_all_benchmarks(self) -> None:
        """Run all benchmarks for all dataset sizes"""
        logger.info("Starting comprehensive benchmark suite...")
        
        all_results = []
        
        for size_name, num_rows in BenchmarkConfig.DATASET_SIZES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking {size_name} dataset ({num_rows:,} rows)")
            logger.info(f"{'='*60}")
            
            # Generate data
            logger.info("Generating synthetic data...")
            df = self.data_generator.generate_stock_data(num_rows)
            
            # Test each operation with both frameworks
            for framework in ["pandas", "fireducks"]:
                if framework == "fireducks" and not FIREDUCKS_AVAILABLE:
                    logger.warning(f"Skipping {framework} (not installed)")
                    continue
                
                logger.info(f"\nTesting with {framework.upper()}")
                
                # Load Parquet
                logger.info("  - Testing load_parquet...")
                result = self.benchmark_load_parquet(df, size_name, framework)
                all_results.append(result)
                self._log_result(result)
                
                # Groupby aggregation
                logger.info("  - Testing groupby_aggregation...")
                result = self.benchmark_groupby_aggregation(df, size_name, framework)
                all_results.append(result)
                self._log_result(result)
                
                # Rolling window
                logger.info("  - Testing rolling_window...")
                result = self.benchmark_rolling_window(df, size_name, framework)
                all_results.append(result)
                self._log_result(result)
                
                # Merge operation
                logger.info("  - Testing merge_operation...")
                df2 = df[["symbol", "close"]].drop_duplicates(subset=["symbol"])
                result = self.benchmark_merge_operation(df, df2, size_name, framework)
                all_results.append(result)
                self._log_result(result)
                
                # Feature engineering
                logger.info("  - Testing feature_engineering...")
                result = self.benchmark_feature_engineering(df, size_name, framework)
                all_results.append(result)
                self._log_result(result)
        
        # Save results
        self.save_results(all_results)
        self.generate_comparison_report(all_results)
    
    @staticmethod
    def _log_result(result: Dict[str, Any]) -> None:
        """Log benchmark result"""
        framework = result["framework"]
        operation = result["operation"]
        time_sec = result["time_seconds"]
        memory_mb = result["peak_memory_mb"]
        
        logger.info(f"    {operation:20} | {time_sec:8.3f}s | {memory_mb:8.1f}MB")
    
    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Save results to JSON
        
        Args:
            results: List of benchmark results
        """
        results_df = pd.DataFrame(results)
        
        # Save as JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {json_path}")
        
        # Save as CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        self.results = results
    
    def generate_comparison_report(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate comparison report
        
        Args:
            results: List of benchmark results
        """
        df = pd.DataFrame(results)
        
        # Calculate speedups
        report_lines = [
            "\n" + "="*80,
            "PANDAS VS FIREDUCKS BENCHMARK REPORT",
            "="*80,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Fireducks Available: {FIREDUCKS_AVAILABLE}\n",
        ]
        
        for size in BenchmarkConfig.DATASET_SIZES.keys():
            size_data = df[df["size"] == size]
            
            report_lines.append(f"\n{size.upper()} DATASET")
            report_lines.append("-" * 80)
            
            for operation in BenchmarkConfig.OPERATIONS:
                op_data = size_data[size_data["operation"] == operation]
                
                if len(op_data) == 0:
                    continue
                
                if len(op_data) == 2:
                    pandas_time = op_data[op_data["framework"] == "pandas"]["time_seconds"].values[0]
                    fireducks_time = op_data[op_data["framework"] == "fireducks"]["time_seconds"].values[0]
                    speedup = pandas_time / fireducks_time
                    
                    report_lines.append(
                        f"\n{operation}:"
                        f"\n  Pandas:    {pandas_time:8.3f}s"
                        f"\n  Fireducks: {fireducks_time:8.3f}s"
                        f"\n  Speedup:   {speedup:8.2f}x"
                    )
                else:
                    pandas_time = op_data[op_data["framework"] == "pandas"]["time_seconds"].values[0]
                    report_lines.append(
                        f"\n{operation}:"
                        f"\n  Pandas:    {pandas_time:8.3f}s"
                        f"\n  Fireducks: (not available)"
                    )
        
        report_lines.append("\n" + "="*80)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        print(report_text)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmarks
    suite = BenchmarkSuite()
    suite.run_all_benchmarks()
