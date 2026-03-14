"""
Feature Engineering DAG - Transform raw data into ML-ready features

This DAG orchestrates the feature engineering pipeline:
- Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Create time-series features (lags, rolling stats, momentum)
- Calculate cross-asset correlations
- Process sentiment from news data
- Compare Pandas vs Fireducks performance
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.sensors.external_task import ExternalTaskSensor
import os
import sys
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, '/opt/airflow/src')

logger = logging.getLogger(__name__)

DATA_BASE_PATH = os.getenv('DATA_PATH', '/opt/airflow/data')

default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': True,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=60),
}


def load_raw_data(**context):
    """Load raw data from storage and prepare for feature engineering."""
    import pandas as pd
    from pathlib import Path

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    raw_path = Path(DATA_BASE_PATH) / 'raw'
    loaded_data = {}

    # Load stock data
    stock_files = list(raw_path.glob('stocks/**/*.parquet'))
    if stock_files:
        stock_dfs = [pd.read_parquet(f) for f in stock_files]
        stock_data = pd.concat(stock_dfs, ignore_index=True) if stock_dfs else pd.DataFrame()
        if not stock_data.empty:
            loaded_data['stocks'] = len(stock_data)
            # Save combined to processed path for downstream tasks
            processed_path = Path(DATA_BASE_PATH) / 'processed' / date_str
            processed_path.mkdir(parents=True, exist_ok=True)
            stock_data.to_parquet(processed_path / 'stocks_combined.parquet')
            logger.info(f"Loaded {len(stock_data)} stock records from {len(stock_files)} files")

    # Load crypto data
    crypto_files = list(raw_path.glob('crypto/**/*.parquet'))
    if crypto_files:
        crypto_dfs = [pd.read_parquet(f) for f in crypto_files]
        crypto_data = pd.concat(crypto_dfs, ignore_index=True) if crypto_dfs else pd.DataFrame()
        if not crypto_data.empty:
            loaded_data['crypto'] = len(crypto_data)
            processed_path = Path(DATA_BASE_PATH) / 'processed' / date_str
            processed_path.mkdir(parents=True, exist_ok=True)
            crypto_data.to_parquet(processed_path / 'crypto_combined.parquet')
            logger.info(f"Loaded {len(crypto_data)} crypto records")

    # If no real data found, generate synthetic
    if not loaded_data:
        logger.warning("No raw data found, generating synthetic data for feature engineering")
        from data.sample_data_generator import SyntheticMarketDataGenerator

        generator = SyntheticMarketDataGenerator(n_days=500, random_seed=42)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        stock_data = generator.generate_multi_asset_data(symbols)

        processed_path = Path(DATA_BASE_PATH) / 'processed' / date_str
        processed_path.mkdir(parents=True, exist_ok=True)
        stock_data.to_parquet(processed_path / 'stocks_combined.parquet')
        loaded_data['stocks'] = len(stock_data)
        loaded_data['synthetic'] = True
        logger.info(f"Generated {len(stock_data)} synthetic stock records")

    return {
        'date': date_str,
        'data_path': str(Path(DATA_BASE_PATH) / 'processed' / date_str),
        'records': loaded_data,
        'total_records': sum(v for k, v in loaded_data.items() if isinstance(v, int)),
    }


def calculate_technical_indicators(**context):
    """Calculate technical indicators using vectorized implementation."""
    from features.technical_indicators import calculate_all_indicators_vectorized
    import pandas as pd
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    stock_file = data_path / 'stocks_combined.parquet'
    if not stock_file.exists():
        logger.warning("No stock data found for technical indicators")
        return {'status': 'skipped', 'reason': 'no_data'}

    df = pd.read_parquet(stock_file)
    results = {}

    # Process each symbol separately (indicators need contiguous time series)
    symbol_col = 'symbol' if 'symbol' in df.columns else None
    if symbol_col:
        symbols = df[symbol_col].unique()
        all_featured = []
        for symbol in symbols:
            symbol_df = df[df[symbol_col] == symbol].copy().sort_index()
            # Ensure required columns exist
            required = ['close', 'high', 'low']
            if all(col in symbol_df.columns for col in required):
                featured = calculate_all_indicators_vectorized(symbol_df)
                all_featured.append(featured)
                results[symbol] = featured.shape[1] - symbol_df.shape[1]
                logger.info(f"Added {results[symbol]} indicators for {symbol}")

        if all_featured:
            combined = pd.concat(all_featured)
            output_file = data_path / 'stocks_with_indicators.parquet'
            combined.to_parquet(output_file)
            return {
                'status': 'success',
                'indicators_per_symbol': results,
                'output_file': str(output_file),
                'total_columns': combined.shape[1],
            }
    else:
        # Single asset without symbol column
        required = ['close', 'high', 'low']
        if all(col in df.columns for col in required):
            featured = calculate_all_indicators_vectorized(df)
            output_file = data_path / 'stocks_with_indicators.parquet'
            featured.to_parquet(output_file)
            return {
                'status': 'success',
                'indicators_added': featured.shape[1] - df.shape[1],
                'output_file': str(output_file),
            }

    return {'status': 'skipped', 'reason': 'missing_columns'}


def create_timeseries_features(**context):
    """Create time-series features using the timeseries module."""
    from features.timeseries import engineer_features
    import pandas as pd
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    # Use indicators file if available, otherwise raw stocks
    input_file = data_path / 'stocks_with_indicators.parquet'
    if not input_file.exists():
        input_file = data_path / 'stocks_combined.parquet'
    if not input_file.exists():
        return {'status': 'skipped', 'reason': 'no_data'}

    df = pd.read_parquet(input_file)
    symbol_col = 'symbol' if 'symbol' in df.columns else None

    feature_config = {
        'lag_columns': ['close'],
        'lag_periods': [1, 2, 3, 5, 7, 14],
        'rolling_columns': ['close'],
        'rolling_windows': [5, 10, 20],
        'rolling_functions': ['mean', 'std'],
        'momentum_periods': [5, 10, 20],
        'include_volatility': True,
        'include_price_features': True,
        'include_volume_features': 'volume' in df.columns,
        'include_trend_features': True,
        'include_relative_features': True,
    }

    if symbol_col:
        all_featured = []
        for symbol in df[symbol_col].unique():
            symbol_df = df[df[symbol_col] == symbol].copy().sort_index()
            featured = engineer_features(symbol_df, feature_config)
            all_featured.append(featured)

        combined = pd.concat(all_featured)
    else:
        combined = engineer_features(df, feature_config)

    output_file = data_path / 'stocks_with_ts_features.parquet'
    combined.to_parquet(output_file)

    new_features = combined.shape[1] - df.shape[1]
    logger.info(f"Created {new_features} time-series features")
    return {
        'status': 'success',
        'features_added': new_features,
        'total_columns': combined.shape[1],
        'output_file': str(output_file),
    }


def calculate_correlations(**context):
    """Calculate cross-asset correlations."""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    stock_file = data_path / 'stocks_combined.parquet'
    if not stock_file.exists():
        return {'status': 'skipped', 'reason': 'no_data'}

    df = pd.read_parquet(stock_file)
    symbol_col = 'symbol' if 'symbol' in df.columns else None

    if not symbol_col or df[symbol_col].nunique() < 2:
        return {'status': 'skipped', 'reason': 'need_multiple_assets'}

    # Calculate return correlations between assets
    date_col = 'date' if 'date' in df.columns else df.index.name
    pivot_col = 'close' if 'close' in df.columns else df.columns[0]

    try:
        if 'date' in df.columns:
            returns = df.pivot(index='date', columns='symbol', values='close').pct_change()
        else:
            returns = df.pivot_table(values='close', columns='symbol', index=df.index).pct_change()

        corr_matrix = returns.corr()

        # Save correlation matrix
        output_file = data_path / 'correlation_matrix.parquet'
        corr_matrix.to_parquet(output_file)

        # Extract top correlations
        pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                pairs.append({
                    'asset1': corr_matrix.index[i],
                    'asset2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j]),
                })

        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        logger.info(f"Calculated {len(pairs)} correlation pairs")

        return {
            'status': 'success',
            'correlation_pairs': len(pairs),
            'top_correlations': pairs[:5],
            'output_file': str(output_file),
        }
    except Exception as e:
        logger.error(f"Correlation calculation error: {e}")
        return {'status': 'error', 'error': str(e)}


def process_sentiment(**context):
    """Process news sentiment using keyword-based analyzer."""
    import pandas as pd
    import json
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    # Find news data
    news_path = Path(DATA_BASE_PATH) / 'raw' / 'news'
    news_files = list(news_path.glob('**/*.json'))

    if not news_files:
        logger.info("No news data available for sentiment analysis")
        return {'status': 'skipped', 'reason': 'no_news_data'}

    try:
        from features.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
    except ImportError:
        logger.warning("SentimentAnalyzer not available, using basic sentiment")
        return {'status': 'skipped', 'reason': 'no_sentiment_module'}

    all_sentiments = []
    for news_file in news_files:
        with open(news_file) as f:
            articles = json.load(f)

        for article in articles:
            if isinstance(article, dict):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = analyzer.analyze(text)
                sentiment['title'] = article.get('title', '')
                sentiment['publishedAt'] = article.get('publishedAt', '')
                all_sentiments.append(sentiment)

    if all_sentiments:
        sentiment_df = pd.DataFrame(all_sentiments)
        output_file = data_path / 'sentiment_scores.parquet'
        sentiment_df.to_parquet(output_file)
        logger.info(f"Processed sentiment for {len(all_sentiments)} articles")
        return {
            'status': 'success',
            'articles_processed': len(all_sentiments),
            'output_file': str(output_file),
        }

    return {'status': 'success', 'articles_processed': 0}


def decide_processing_engine(**context):
    """Decide which processing engine to use based on data size."""
    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')

    total_records = data_info.get('total_records', 0)

    if total_records < 100000:
        logger.info(f"Using Pandas for {total_records} records")
        return 'process_with_pandas'
    else:
        logger.info(f"Using Fireducks for {total_records} records")
        return 'process_with_fireducks'


def process_with_pandas(**context):
    """Combine all features using Pandas."""
    import pandas as pd
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    start_time = time.time()

    # Load the most complete feature file available
    feature_files = [
        data_path / 'stocks_with_ts_features.parquet',
        data_path / 'stocks_with_indicators.parquet',
        data_path / 'stocks_combined.parquet',
    ]

    df = None
    for f in feature_files:
        if f.exists():
            df = pd.read_parquet(f)
            logger.info(f"Loaded features from {f.name}: {df.shape}")
            break

    if df is None:
        return {'engine': 'pandas', 'status': 'skipped', 'reason': 'no_data'}

    # Drop rows with too many NaN values (from lagged features)
    initial_rows = len(df)
    nan_threshold = 0.5  # Drop rows with >50% NaN
    df = df.dropna(thresh=int(df.shape[1] * (1 - nan_threshold)))
    logger.info(f"Dropped {initial_rows - len(df)} rows with excessive NaNs")

    # Fill remaining NaN with forward fill then 0
    df = df.fillna(method='ffill').fillna(0)

    # Save processed features
    output_file = data_path / 'features_pandas.parquet'
    df.to_parquet(output_file)

    elapsed = time.time() - start_time
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    logger.info(f"Pandas processing: {elapsed:.2f}s, {memory_mb:.1f}MB, {df.shape}")
    return {
        'engine': 'pandas',
        'status': 'success',
        'elapsed_time': elapsed,
        'memory_mb': memory_mb,
        'shape': list(df.shape),
        'output_file': str(output_file),
    }


def process_with_fireducks(**context):
    """Process features using Fireducks (falls back to Pandas)."""
    import time as t

    start_time = t.time()

    try:
        import fireducks.pandas as fpd
        engine_name = 'fireducks'
        logger.info(f"Using Fireducks v{fpd.__version__}")
    except ImportError:
        logger.info("Fireducks not available, falling back to Pandas")
        import pandas as fpd
        engine_name = 'pandas_fallback'

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    from pathlib import Path
    data_path = Path(data_info['data_path'])

    feature_files = [
        data_path / 'stocks_with_ts_features.parquet',
        data_path / 'stocks_with_indicators.parquet',
        data_path / 'stocks_combined.parquet',
    ]

    df = None
    for f in feature_files:
        if f.exists():
            df = fpd.read_parquet(str(f))
            break

    if df is None:
        return {'engine': engine_name, 'status': 'skipped'}

    df = df.dropna(thresh=int(df.shape[1] * 0.5))
    df = df.fillna(method='ffill').fillna(0)

    output_file = data_path / 'features_fireducks.parquet'
    df.to_parquet(str(output_file))

    elapsed = t.time() - start_time
    logger.info(f"{engine_name} processing: {elapsed:.2f}s")
    return {
        'engine': engine_name,
        'status': 'success',
        'elapsed_time': elapsed,
        'shape': list(df.shape),
        'output_file': str(output_file),
    }


def process_with_spark(**context):
    """Process features using Apache Spark (placeholder)."""
    logger.info("Spark processing not configured for Cloud Run. Skipping.")
    return {'engine': 'spark', 'status': 'skipped', 'reason': 'not_configured'}


def benchmark_performance(**context):
    """Compare performance across processing engines."""
    ti = context['task_instance']

    results = {}
    for task_id in ['process_with_pandas', 'process_with_fireducks', 'process_with_spark']:
        result = ti.xcom_pull(task_ids=task_id)
        if result and result.get('status') == 'success':
            results[result['engine']] = {
                'elapsed_time': result.get('elapsed_time', 0),
                'memory_mb': result.get('memory_mb', 'N/A'),
            }

    if len(results) > 1:
        engines = list(results.keys())
        logger.info(f"Benchmark comparison: {results}")
    elif results:
        logger.info(f"Only one engine ran: {results}")

    return {'benchmark_results': results}


def validate_features(**context):
    """Validate engineered features for quality."""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    data_path = Path(data_info['data_path'])

    # Find the processed features file
    feature_files = [
        data_path / 'features_pandas.parquet',
        data_path / 'features_fireducks.parquet',
        data_path / 'stocks_with_ts_features.parquet',
        data_path / 'stocks_with_indicators.parquet',
    ]

    df = None
    source = None
    for f in feature_files:
        if f.exists():
            df = pd.read_parquet(f)
            source = f.name
            break

    if df is None:
        return {'validation_status': 'skipped', 'reason': 'no_features'}

    checks = {}

    # Null check
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    checks['null_percentage'] = null_pct
    checks['null_check'] = 'passed' if null_pct < 5 else 'warning'

    # Infinite value check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    checks['infinite_values'] = int(inf_count)
    checks['inf_check'] = 'passed' if inf_count == 0 else 'warning'

    # Constant columns check (zero variance)
    constant_cols = [col for col in numeric_cols if df[col].std() == 0]
    checks['constant_columns'] = constant_cols
    checks['variance_check'] = 'passed' if not constant_cols else 'warning'

    # Shape check
    checks['rows'] = df.shape[0]
    checks['columns'] = df.shape[1]
    checks['source_file'] = source

    overall = 'passed' if all(
        v == 'passed' for k, v in checks.items() if k.endswith('_check')
    ) else 'warnings'

    logger.info(f"Feature validation: {overall} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return {'validation_status': overall, 'checks': checks}


def store_features(**context):
    """Store final engineered features."""
    import pandas as pd
    from pathlib import Path
    import shutil

    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    data_path = Path(data_info['data_path'])

    # Find best available feature file
    feature_files = [
        data_path / 'features_pandas.parquet',
        data_path / 'features_fireducks.parquet',
        data_path / 'stocks_with_ts_features.parquet',
        data_path / 'stocks_with_indicators.parquet',
    ]

    source_file = None
    for f in feature_files:
        if f.exists():
            source_file = f
            break

    if source_file is None:
        return {'status': 'skipped', 'reason': 'no_features'}

    # Copy to feature store location
    feature_store = Path(DATA_BASE_PATH) / 'features' / date_str
    feature_store.mkdir(parents=True, exist_ok=True)
    dest = feature_store / 'features.parquet'
    shutil.copy2(source_file, dest)

    # Also save a "latest" symlink/copy for easy access
    latest_dir = Path(DATA_BASE_PATH) / 'features' / 'latest'
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, latest_dir / 'features.parquet')

    df = pd.read_parquet(dest)
    logger.info(f"Features stored at {dest}: {df.shape}")

    return {
        'feature_path': str(dest),
        'latest_path': str(latest_dir / 'features.parquet'),
        'shape': list(df.shape),
        'format': 'parquet',
    }


# Define the DAG
with DAG(
    dag_id='feature_engineering_pipeline',
    default_args=default_args,
    description='Engineer features from raw market data',
    schedule_interval='0 4 * * *',  # Run daily at 4 AM UTC (after ingestion)
    catchup=False,
    max_active_runs=1,
    tags=['feature-engineering', 'ml-pipeline', 'etl'],
) as dag:

    wait_for_ingestion = ExternalTaskSensor(
        task_id='wait_for_ingestion',
        external_dag_id='data_ingestion_pipeline',
        external_task_id='end',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='poke',
        poke_interval=60,
        timeout=3600,
    )

    load_data = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_raw_data,
        provide_context=True,
    )

    with TaskGroup('feature_creation', tooltip='Create various features') as feature_creation:
        technical_indicators = PythonOperator(
            task_id='technical_indicators',
            python_callable=calculate_technical_indicators,
            provide_context=True,
        )
        timeseries_features = PythonOperator(
            task_id='timeseries_features',
            python_callable=create_timeseries_features,
            provide_context=True,
        )
        correlations = PythonOperator(
            task_id='correlations',
            python_callable=calculate_correlations,
            provide_context=True,
        )
        sentiment = PythonOperator(
            task_id='sentiment',
            python_callable=process_sentiment,
            provide_context=True,
        )
        [technical_indicators, timeseries_features, correlations, sentiment]

    decide_engine = BranchPythonOperator(
        task_id='decide_engine',
        python_callable=decide_processing_engine,
        provide_context=True,
    )

    pandas_processing = PythonOperator(
        task_id='process_with_pandas',
        python_callable=process_with_pandas,
        provide_context=True,
    )

    fireducks_processing = PythonOperator(
        task_id='process_with_fireducks',
        python_callable=process_with_fireducks,
        provide_context=True,
    )

    spark_processing = PythonOperator(
        task_id='process_with_spark',
        python_callable=process_with_spark,
        provide_context=True,
    )

    benchmark = PythonOperator(
        task_id='benchmark',
        python_callable=benchmark_performance,
        provide_context=True,
        trigger_rule='none_failed',
    )

    validate = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True,
    )

    store = PythonOperator(
        task_id='store_features',
        python_callable=store_features,
        provide_context=True,
    )

    wait_for_ingestion >> load_data >> feature_creation >> decide_engine
    decide_engine >> [pandas_processing, fireducks_processing, spark_processing]
    [pandas_processing, fireducks_processing, spark_processing] >> benchmark
    benchmark >> validate >> store
