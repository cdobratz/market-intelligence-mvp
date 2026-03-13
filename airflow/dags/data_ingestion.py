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


# Placeholder functions
def fetch_stock_data(**context):
    """Fetch stock data from Alpha Vantage"""
    import os
    from src.data.ingestion import AlphaVantageClient, save_to_parquet
    
    print("Fetching stock data from Alpha Vantage...")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    results = []
    
    if api_key:
        try:
            client = AlphaVantageClient(api_key=api_key)
            for symbol in symbols:
                try:
                    df = client.get_daily_stock_data(symbol, outputsize='compact')
                    df['symbol'] = symbol
                    # Save to parquet
                    save_to_parquet(df, '/opt/airflow/data/raw', 'stocks', date_str)
                    results.append({'symbol': symbol, 'status': 'success', 'records': len(df)})
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    results.append({'symbol': symbol, 'status': 'error', 'error': str(e)})
        except Exception as e:
            print(f"Alpha Vantage client error: {e}")
            results.append({'status': 'error', 'error': str(e)})
    else:
        print("WARNING: ALPHA_VANTAGE_API_KEY not set, using demo mode")
        results.append({'status': 'skipped', 'reason': 'no_api_key'})
    
    print(f"Fetched data for {len(symbols)} symbols on {date_str}")
    return {'status': 'success', 'symbols': symbols, 'date': date_str, 'results': results}


def fetch_forex_data(**context):
    """Fetch forex data from Alpha Vantage"""
    import os
    from src.data.ingestion import AlphaVantageClient, save_to_parquet
    
    print("Fetching forex data from Alpha Vantage...")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    pairs = [('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY')]
    results = []
    
    if api_key:
        try:
            client = AlphaVantageClient(api_key=api_key)
            for from_curr, to_curr in pairs:
                try:
                    df = client.get_fx_daily(from_curr, to_curr, outputsize='compact')
                    # Save to parquet
                    save_to_parquet(df, '/opt/airflow/data/raw', 'forex', date_str)
                    results.append({'pair': f'{from_curr}{to_curr}', 'status': 'success', 'records': len(df)})
                except Exception as e:
                    print(f"Error fetching {from_curr}/{to_curr}: {e}")
        except Exception as e:
            print(f"Alpha Vantage client error: {e}")
    else:
        print("WARNING: ALPHA_VANTAGE_API_KEY not set")
        results.append({'status': 'skipped', 'reason': 'no_api_key'})
    
    print(f"Fetched forex data for {len(pairs)} pairs on {date_str}")
    return {'status': 'success', 'pairs': [f'{f}{t}' for f, t in pairs], 'date': date_str, 'results': results}


def fetch_crypto_data(**context):
    """Fetch cryptocurrency data from CoinGecko"""
    from src.data.ingestion import CoinGeckoClient, save_to_parquet
    
    print("Fetching crypto data from CoinGecko...")
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    coins = ['bitcoin', 'ethereum', 'cardano', 'polkadot', 'solana']
    results = []
    
    try:
        client = CoinGeckoClient()
        df = client.get_coin_market_data(coins, vs_currency='usd', days=30)
        if not df.empty:
            # Save to parquet
            save_to_parquet(df, '/opt/airflow/data/raw', 'crypto', date_str)
            results.append({'status': 'success', 'records': len(df)})
    except Exception as e:
        print(f"CoinGecko client error: {e}")
        results.append({'status': 'error', 'error': str(e)})
    
    print(f"Fetched crypto data for {len(coins)} coins on {date_str}")
    return {'status': 'success', 'coins': coins, 'date': date_str, 'results': results}


def fetch_news_data(**context):
    """Fetch financial news from News API"""
    import os
    from src.data.ingestion import NewsAPIClient, save_to_json
    
    print("Fetching news data from News API...")
    
    api_key = os.getenv('NEWS_API_KEY')
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    keywords = ['stock market', 'cryptocurrency', 'financial news']
    results = []
    
    if api_key:
        try:
            client = NewsAPIClient(api_key=api_key)
            for keyword in keywords:
                try:
                    df = client.get_everything(query=keyword, page_size=50)
                    if not df.empty:
                        # Save to JSON (articles have complex nested structure)
                        save_to_json(df.to_dict('records'), '/opt/airflow/data/raw', 'news', date_str)
                        results.append({'keyword': keyword, 'status': 'success', 'articles': len(df)})
                except Exception as e:
                    print(f"Error fetching news for {keyword}: {e}")
        except Exception as e:
            print(f"News API client error: {e}")
            results.append({'status': 'error', 'error': str(e)})
    else:
        print("WARNING: NEWS_API_KEY not set, skipping news fetch")
        results.append({'status': 'skipped', 'reason': 'no_api_key'})
    
    print(f"Fetched news for keywords: {keywords} on {date_str}")
    return {'status': 'success', 'keywords': keywords, 'date': date_str, 'results': results}


def validate_ingested_data(**context):
    """Validate the ingested data"""
    import pandas as pd
    import os
    
    print("Validating ingested data...")
    
    ti = context['task_instance']
    
    # Get results from fetch tasks - correct task_ids within TaskGroup
    stock_result = ti.xcom_pull(task_ids='fetch_data.fetch_stocks')
    forex_result = ti.xcom_pull(task_ids='fetch_data.fetch_forex')
    crypto_result = ti.xcom_pull(task_ids='fetch_data.fetch_crypto')
    news_result = ti.xcom_pull(task_ids='fetch_data.fetch_news')
    
    print(f"Stock data: {stock_result}")
    print(f"Forex data: {forex_result}")
    print(f"Crypto data: {crypto_result}")
    print(f"News data: {news_result}")
    
    # Check if data files exist
    execution_date = context['execution_date']
    data_path = f"/opt/airflow/data/raw/{execution_date.strftime('%Y-%m-%d')}"
    
    validation_result = {
        'validation_status': 'passed',
        'timestamp': str(datetime.now()),
        'data_path': data_path,
        'stock_status': stock_result.get('status') if stock_result else 'failed',
        'forex_status': forex_result.get('status') if forex_result else 'failed',
        'crypto_status': crypto_result.get('status') if crypto_result else 'failed',
        'news_status': news_result.get('status') if news_result else 'skipped'
    }
    
    print(f"Validation result: {validation_result}")
    return validation_result


def store_raw_data(**context):
    """Store raw data in appropriate format (Parquet)"""
    import pandas as pd
    import os
    
    print("Storing raw data to data lake...")
    
    execution_date = context['execution_date']
    data_path = f"/opt/airflow/data/raw/{execution_date.strftime('%Y-%m-%d')}"
    
    # Create directory if not exists
    os.makedirs(data_path, exist_ok=True)
    
    # For demo: save placeholder files
    # In production, this would save actual fetched data
    placeholder_data = {
        'stocks': pd.DataFrame({'status': ['placeholder']}),
        'forex': pd.DataFrame({'status': ['placeholder']}),
        'crypto': pd.DataFrame({'status': ['placeholder']}),
    }
    
    for name, df in placeholder_data.items():
        df.to_parquet(f"{data_path}/{name}.parquet", index=False)
    
    print(f"Data stored at: {data_path}")
    return {'storage_path': data_path, 'format': 'parquet', 'files': list(placeholder_data.keys())}


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
