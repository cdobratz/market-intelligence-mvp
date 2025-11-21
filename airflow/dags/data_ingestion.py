"""
Data Ingestion DAG - Fetch financial data from multiple APIs

This DAG orchestrates the ingestion of financial market data from:
- Alpha Vantage (stock prices and technical indicators)
- CoinGecko (cryptocurrency data)
- News API (financial news for sentiment analysis)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import custom functions (to be implemented)
# from src.data.ingestion import (
#     fetch_alpha_vantage_data,
#     fetch_coingecko_data,
#     fetch_news_data,
#     validate_raw_data,
# )


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


# Placeholder functions (will be implemented in src/data/ingestion.py)
def fetch_stock_data(**context):
    """Fetch stock data from Alpha Vantage"""
    print("Fetching stock data from Alpha Vantage...")
    # TODO: Implement actual API call
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    execution_date = context['execution_date']
    
    print(f"Fetching data for {len(symbols)} symbols on {execution_date}")
    # Actual implementation will save to data/raw/stocks/
    return {'status': 'success', 'symbols': symbols, 'date': str(execution_date)}


def fetch_forex_data(**context):
    """Fetch forex data from Alpha Vantage"""
    print("Fetching forex data from Alpha Vantage...")
    # TODO: Implement actual API call
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    execution_date = context['execution_date']
    
    print(f"Fetching data for {len(pairs)} currency pairs on {execution_date}")
    return {'status': 'success', 'pairs': pairs, 'date': str(execution_date)}


def fetch_crypto_data(**context):
    """Fetch cryptocurrency data from CoinGecko"""
    print("Fetching crypto data from CoinGecko...")
    # TODO: Implement actual API call
    coins = ['bitcoin', 'ethereum', 'cardano', 'polkadot', 'solana']
    execution_date = context['execution_date']
    
    print(f"Fetching data for {len(coins)} cryptocurrencies on {execution_date}")
    return {'status': 'success', 'coins': coins, 'date': str(execution_date)}


def fetch_news_data(**context):
    """Fetch financial news from News API"""
    print("Fetching news data from News API...")
    # TODO: Implement actual API call
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        print("WARNING: NEWS_API_KEY not set, skipping news fetch")
        return {'status': 'skipped', 'reason': 'no_api_key'}
    
    execution_date = context['execution_date']
    keywords = ['stock market', 'cryptocurrency', 'financial news']
    
    print(f"Fetching news for keywords: {keywords} on {execution_date}")
    return {'status': 'success', 'keywords': keywords, 'date': str(execution_date)}


def validate_ingested_data(**context):
    """Validate the ingested data"""
    print("Validating ingested data...")
    # TODO: Implement data quality checks
    # - Check for null values
    # - Validate date ranges
    # - Check data types
    # - Verify record counts
    
    ti = context['task_instance']
    stock_result = ti.xcom_pull(task_ids='fetch_data.fetch_stocks')
    forex_result = ti.xcom_pull(task_ids='fetch_data.fetch_forex')
    crypto_result = ti.xcom_pull(task_ids='fetch_data.fetch_crypto')
    news_result = ti.xcom_pull(task_ids='fetch_data.fetch_news')
    
    print(f"Stock data: {stock_result}")
    print(f"Forex data: {forex_result}")
    print(f"Crypto data: {crypto_result}")
    print(f"News data: {news_result}")
    
    return {'validation_status': 'passed', 'timestamp': str(datetime.now())}


def store_raw_data(**context):
    """Store raw data in appropriate format (Parquet)"""
    print("Storing raw data to data lake...")
    # TODO: Implement storage logic
    # - Convert to Parquet format
    # - Partition by date
    # - Store in data/raw/ directory
    
    execution_date = context['execution_date']
    data_path = f"/opt/airflow/data/raw/{execution_date.strftime('%Y-%m-%d')}"
    
    print(f"Data stored at: {data_path}")
    return {'storage_path': data_path, 'format': 'parquet'}


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
    
    # Start task
    start = EmptyOperator(
        task_id='start',
        dag=dag,
    )
    
    # Task Group for data fetching
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
        
        # All fetch tasks run in parallel
        [fetch_stocks, fetch_forex, fetch_crypto, fetch_news]
    
    # Validate data
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_ingested_data,
        provide_context=True,
    )
    
    # Store data
    store = PythonOperator(
        task_id='store_raw_data',
        python_callable=store_raw_data,
        provide_context=True,
    )
    
    # End task
    end = EmptyOperator(
        task_id='end',
        dag=dag,
    )
    
    # Define task dependencies
    start >> fetch_data >> validate >> store >> end


"""
DAG Structure:

    start
      |
   fetch_data (Task Group)
      |-- fetch_stocks
      |-- fetch_forex
      |-- fetch_crypto
      |-- fetch_news
      |
   validate_data
      |
   store_raw_data
      |
    end

Schedule: Daily at 2 AM UTC
Retries: 3 with 5-minute delay
Timeout: 30 minutes per task
"""
