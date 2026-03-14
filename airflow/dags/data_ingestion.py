"""
Data Ingestion DAG - Fetch financial data from multiple APIs

This DAG orchestrates the ingestion of financial market data from:
- Alpha Vantage (stock prices and technical indicators)
- CoinGecko (cryptocurrency data)
- News API (financial news for sentiment analysis)

When API keys are not available, falls back to synthetic data generation.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
import json
import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, '/opt/airflow/src')

logger = logging.getLogger(__name__)

# Data storage base path (ephemeral on Cloud Run, persistent on local Docker)
DATA_BASE_PATH = os.getenv('DATA_PATH', '/opt/airflow/data')

# Stock symbols and crypto coins to track
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
FOREX_PAIRS = [('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY')]
CRYPTO_COINS = ['bitcoin', 'ethereum', 'cardano', 'polkadot', 'solana']

# Default arguments for the DAG
default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}


def fetch_stock_data(**context):
    """Fetch stock data from Alpha Vantage API."""
    from data.ingestion import AlphaVantageClient, save_to_parquet

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.warning("ALPHA_VANTAGE_API_KEY not set, generating synthetic stock data")
        return _generate_synthetic_stocks(date_str)

    client = AlphaVantageClient(api_key=api_key)
    results = {}

    for symbol in STOCK_SYMBOLS:
        try:
            df = client.get_daily_stock_data(symbol, outputsize='compact')
            file_path = save_to_parquet(df, DATA_BASE_PATH + '/raw', 'stocks', date_str)
            results[symbol] = {
                'status': 'success',
                'rows': len(df),
                'file': file_path,
            }
            logger.info(f"Fetched {len(df)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            results[symbol] = {'status': 'error', 'error': str(e)}

    return {'date': date_str, 'symbols': results}


def fetch_forex_data(**context):
    """Fetch forex data from Alpha Vantage API."""
    from data.ingestion import AlphaVantageClient, save_to_parquet

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.warning("ALPHA_VANTAGE_API_KEY not set, generating synthetic forex data")
        return _generate_synthetic_forex(date_str)

    client = AlphaVantageClient(api_key=api_key)
    results = {}

    for from_curr, to_curr in FOREX_PAIRS:
        pair = f"{from_curr}{to_curr}"
        try:
            df = client.get_fx_daily(from_curr, to_curr, outputsize='compact')
            file_path = save_to_parquet(df, DATA_BASE_PATH + '/raw', 'forex', date_str)
            results[pair] = {
                'status': 'success',
                'rows': len(df),
                'file': file_path,
            }
            logger.info(f"Fetched {len(df)} rows for {pair}")
        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            results[pair] = {'status': 'error', 'error': str(e)}

    return {'date': date_str, 'pairs': results}


def fetch_crypto_data(**context):
    """Fetch cryptocurrency data from CoinGecko API (no API key required)."""
    from data.ingestion import CoinGeckoClient, save_to_parquet
    import pandas as pd

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    try:
        client = CoinGeckoClient()
        df = client.get_coin_market_data(CRYPTO_COINS, vs_currency='usd', days=30)

        if not df.empty:
            # Set timestamp as index for parquet save
            df = df.set_index('timestamp')
            file_path = save_to_parquet(df, DATA_BASE_PATH + '/raw', 'crypto', date_str)
            logger.info(f"Fetched {len(df)} crypto records")
            return {
                'date': date_str,
                'status': 'success',
                'rows': len(df),
                'file': file_path,
                'coins': CRYPTO_COINS,
            }
    except Exception as e:
        logger.error(f"CoinGecko error: {e}")

    # Fallback to synthetic data
    logger.warning("Falling back to synthetic crypto data")
    return _generate_synthetic_crypto(date_str)


def fetch_news_data(**context):
    """Fetch financial news from News API."""
    from data.ingestion import NewsAPIClient, save_to_json

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        logger.warning("NEWS_API_KEY not set, skipping news fetch")
        return {'date': date_str, 'status': 'skipped', 'reason': 'no_api_key'}

    try:
        client = NewsAPIClient(api_key=api_key)

        # Fetch business headlines
        headlines_df = client.get_top_headlines(category='business', country='us')

        # Fetch market-specific news
        keywords = ['stock market', 'cryptocurrency', 'federal reserve']
        all_articles = [headlines_df] if not headlines_df.empty else []

        for keyword in keywords:
            try:
                articles_df = client.get_everything(
                    query=keyword,
                    from_date=(execution_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                    to_date=date_str,
                )
                if not articles_df.empty:
                    all_articles.append(articles_df)
            except Exception as e:
                logger.warning(f"Error fetching news for '{keyword}': {e}")

        import pandas as pd
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True).drop_duplicates(subset=['url'])
            # Save as JSON (news articles have nested structures)
            articles_list = combined.to_dict(orient='records')
            file_path = save_to_json(articles_list, DATA_BASE_PATH + '/raw', 'news', date_str)
            logger.info(f"Fetched {len(combined)} unique articles")
            return {
                'date': date_str,
                'status': 'success',
                'articles': len(combined),
                'file': file_path,
            }

        return {'date': date_str, 'status': 'success', 'articles': 0}

    except Exception as e:
        logger.error(f"News API error: {e}")
        return {'date': date_str, 'status': 'error', 'error': str(e)}


def validate_ingested_data(**context):
    """Validate all ingested data using DataValidator."""
    from data.validation import DataValidator
    import pandas as pd
    from pathlib import Path

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    ti = context['task_instance']

    validator = DataValidator()
    validation_reports = []

    # Validate stock data
    stock_result = ti.xcom_pull(task_ids='fetch_data.fetch_stocks')
    if stock_result and isinstance(stock_result, dict):
        symbols = stock_result.get('symbols', {})
        for symbol, info in symbols.items():
            if isinstance(info, dict) and info.get('status') == 'success' and info.get('file'):
                try:
                    df = pd.read_parquet(info['file'])
                    is_valid, report = validator.validate_stock_data(df, symbol)
                    validation_reports.append(report)
                except Exception as e:
                    logger.warning(f"Could not validate stock data for {symbol}: {e}")

    # Validate crypto data
    crypto_result = ti.xcom_pull(task_ids='fetch_data.fetch_crypto')
    if crypto_result and isinstance(crypto_result, dict) and crypto_result.get('file'):
        try:
            df = pd.read_parquet(crypto_result['file'])
            for coin_id in df.get('coin_id', pd.Series()).unique():
                coin_df = df[df['coin_id'] == coin_id]
                is_valid, report = validator.validate_crypto_data(coin_df, coin_id)
                validation_reports.append(report)
        except Exception as e:
            logger.warning(f"Could not validate crypto data: {e}")

    # Generate summary
    summary = validator.generate_validation_report(
        validation_reports,
        output_path=Path(DATA_BASE_PATH) / 'validation' / f'report_{date_str}.json'
    )

    logger.info(
        f"Validation complete: {summary['passed']}/{summary['total_validations']} passed, "
        f"{summary['total_warnings']} warnings, {summary['total_errors']} errors"
    )

    return {
        'date': date_str,
        'validation_status': 'passed' if summary['failed'] == 0 else 'warnings',
        'passed': summary['passed'],
        'failed': summary['failed'],
        'warnings': summary['total_warnings'],
    }


def store_raw_data(**context):
    """Consolidate and verify raw data storage."""
    from pathlib import Path

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    ti = context['task_instance']

    stored_files = []

    # Collect file paths from all fetch tasks
    for task_id in ['fetch_data.fetch_stocks', 'fetch_data.fetch_forex',
                    'fetch_data.fetch_crypto', 'fetch_data.fetch_news']:
        result = ti.xcom_pull(task_ids=task_id)
        if isinstance(result, dict):
            if 'file' in result:
                stored_files.append(result['file'])
            # Handle nested symbol/pair results
            for key in ['symbols', 'pairs']:
                if key in result:
                    for name, info in result[key].items():
                        if isinstance(info, dict) and 'file' in info:
                            stored_files.append(info['file'])

    logger.info(f"Data stored: {len(stored_files)} files for {date_str}")
    for f in stored_files:
        logger.info(f"  - {f}")

    return {
        'date': date_str,
        'storage_path': f"{DATA_BASE_PATH}/raw",
        'files': stored_files,
        'file_count': len(stored_files),
    }


# --- Synthetic data fallback functions ---

def _generate_synthetic_stocks(date_str):
    """Generate synthetic stock data when API keys are unavailable."""
    from data.sample_data_generator import SyntheticMarketDataGenerator
    from data.ingestion import save_to_parquet

    generator = SyntheticMarketDataGenerator(n_days=100, random_seed=42)
    results = {}
    for symbol in STOCK_SYMBOLS:
        df = generator.generate_ohlcv_data(symbol=symbol)
        # Adapt columns: set date as index, drop symbol for parquet
        df = df.set_index('date')
        file_path = save_to_parquet(df, DATA_BASE_PATH + '/raw', 'stocks', date_str)
        results[symbol] = {'status': 'success', 'rows': len(df), 'file': file_path, 'synthetic': True}
    return {'date': date_str, 'symbols': results, 'synthetic': True}


def _generate_synthetic_forex(date_str):
    """Generate synthetic forex data."""
    from data.sample_data_generator import SyntheticMarketDataGenerator
    from data.ingestion import save_to_parquet
    import pandas as pd

    generator = SyntheticMarketDataGenerator(start_price=1.1, n_days=100, volatility=0.005, random_seed=43)
    results = {}
    for from_curr, to_curr in FOREX_PAIRS:
        pair = f"{from_curr}{to_curr}"
        df = generator.generate_ohlcv_data(symbol=pair)
        df = df.rename(columns={'symbol': 'pair'})
        df = df.set_index('date')
        # Drop volume for forex
        df = df.drop(columns=['volume'], errors='ignore')
        file_path = save_to_parquet(df, DATA_BASE_PATH + '/raw', 'forex', date_str)
        results[pair] = {'status': 'success', 'rows': len(df), 'file': file_path, 'synthetic': True}
    return {'date': date_str, 'pairs': results, 'synthetic': True}


def _generate_synthetic_crypto(date_str):
    """Generate synthetic crypto data."""
    from data.sample_data_generator import SyntheticMarketDataGenerator
    from data.ingestion import save_to_parquet
    import pandas as pd
    import numpy as np

    generator = SyntheticMarketDataGenerator(start_price=50000, n_days=100, volatility=0.04, random_seed=44)
    all_data = []
    for coin in CRYPTO_COINS:
        df = generator.generate_ohlcv_data(symbol=coin)
        coin_df = pd.DataFrame({
            'price': df['close'].values,
            'market_cap': (df['close'] * np.random.uniform(1e9, 1e11)).values,
            'volume': df['volume'].values,
            'coin_id': coin,
        }, index=df['date'])
        coin_df.index.name = 'timestamp'
        all_data.append(coin_df)

    combined = pd.concat(all_data)
    file_path = save_to_parquet(combined, DATA_BASE_PATH + '/raw', 'crypto', date_str)
    return {
        'date': date_str,
        'status': 'success',
        'rows': len(combined),
        'file': file_path,
        'coins': CRYPTO_COINS,
        'synthetic': True,
    }


# Define the DAG
with DAG(
    dag_id='data_ingestion_pipeline',
    default_args=default_args,
    description='Ingest financial market data from multiple APIs',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM UTC
    catchup=False,
    max_active_runs=1,
    tags=['data-ingestion', 'market-data', 'etl'],
) as dag:

    start = EmptyOperator(task_id='start')

    with TaskGroup('fetch_data', tooltip='Fetch data from various sources') as fetch_data:
        fetch_stocks = PythonOperator(
            task_id='fetch_stocks',
            python_callable=fetch_stock_data,
            provide_context=True,
        )
        fetch_forex = PythonOperator(
            task_id='fetch_forex',
            python_callable=fetch_forex_data,
            provide_context=True,
        )
        fetch_crypto = PythonOperator(
            task_id='fetch_crypto',
            python_callable=fetch_crypto_data,
            provide_context=True,
        )
        fetch_news = PythonOperator(
            task_id='fetch_news',
            python_callable=fetch_news_data,
            provide_context=True,
        )
        [fetch_stocks, fetch_forex, fetch_crypto, fetch_news]

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_ingested_data,
        provide_context=True,
    )

    store = PythonOperator(
        task_id='store_raw_data',
        python_callable=store_raw_data,
        provide_context=True,
    )

    end = EmptyOperator(task_id='end')

    start >> fetch_data >> validate >> store >> end
