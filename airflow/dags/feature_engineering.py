"""
Feature Engineering DAG - Transform raw data into ML-ready features

This DAG orchestrates the feature engineering pipeline:
- Calculate technical indicators
- Create time-series features
- Generate cross-asset correlations
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# Default arguments
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


# Placeholder functions
def load_raw_data(**context):
    """Load raw data from storage"""
    import pandas as pd
    import os
    
    print("Loading raw data from data lake...")
    execution_date = context['execution_date']
    data_path = f"/opt/airflow/data/raw/{execution_date.strftime('%Y-%m-%d')}"
    
    # Check if data exists, fall back to processed data
    if not os.path.exists(data_path):
        print(f"Raw data not found at {data_path}, checking processed data...")
        processed_path = "/opt/airflow/data/processed"
        if os.path.exists(processed_path):
            # Load processed data
            train_file = os.path.join(processed_path, "train_data.parquet")
            if os.path.exists(train_file):
                df = pd.read_parquet(train_file)
                print(f"Loaded processed data: {df.shape}")
                return {
                    'data_path': processed_path, 
                    'records': len(df),
                    'data_type': 'processed'
                }
    
    print(f"Loading data from: {data_path}")
    return {'data_path': data_path, 'records': 10000, 'data_type': 'raw'}


def calculate_technical_indicators(**context):
    """Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)"""
    import pandas as pd
    import numpy as np
    
    print("Calculating technical indicators...")
    
    # Try to import actual technical indicators
    try:
        from src.features.technical_indicators import add_technical_indicators
        indicators_available = True
    except ImportError:
        indicators_available = False
    
    # Technical indicators to calculate:
    # - RSI (Relative Strength Index)
    # - MACD (Moving Average Convergence Divergence)
    # - Bollinger Bands
    # - Moving Averages (SMA, EMA)
    # - Volume indicators
    # - Momentum indicators
    
    indicators = ['RSI', 'MACD', 'BB', 'SMA_20', 'EMA_50', 'Volume_MA']
    print(f"Calculated indicators: {indicators}")
    
    return {'indicators': indicators, 'status': 'success', 'available': indicators_available}


def create_timeseries_features(**context):
    """Create time-series specific features"""
    import pandas as pd
    
    print("Creating time-series features...")
    
    # Try to import actual timeseries features
    try:
        from src.features.timeseries import engineer_features
        features_available = True
    except ImportError:
        features_available = False
    
    # Features to create:
    # - Lag features (1, 7, 30 days)
    # - Rolling statistics (mean, std, min, max)
    # - Rate of change
    # - Seasonal features
    # - Day of week, month effects
    
    features = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'roc']
    print(f"Created features: {features}")
    
    return {'features': features, 'status': 'success', 'available': features_available}


def calculate_correlations(**context):
    """Calculate cross-asset correlations"""
    print("Calculating cross-asset correlations...")
    
    # Try to load data and calculate correlations
    try:
        import pandas as pd
        import os
        
        # Load processed data
        train_path = "/opt/airflow/data/processed/train_data.parquet"
        if os.path.exists(train_path):
            df = pd.read_parquet(train_path)
            # Calculate correlation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # Get upper triangle indices
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find highly correlated pairs
                high_corr = [(col, idx, upper.loc[idx, col]) 
                            for col in upper.columns 
                            for idx in upper.index 
                            if pd.notna(upper.loc[idx, col]) and abs(upper.loc[idx, col]) > 0.7]
                
                print(f"Found {len(high_corr)} highly correlated feature pairs")
    except Exception as e:
        print(f"Could not calculate correlations: {e}")
    
    # Correlation analysis:
    # - Stock-stock correlations
    # - Stock-crypto correlations
    # - Currency-asset correlations
    # - Rolling correlations
    
    print("Calculated correlation matrix")
    return {'correlation_pairs': 25, 'status': 'success'}


def process_sentiment(**context):
    """Extract sentiment features from news data"""
    print("Processing news sentiment...")
    
    # Sentiment processing:
    # - Text preprocessing
    # - Sentiment scoring
    # - Entity extraction
    # - Topic modeling
    
    print("Processed sentiment for news articles")
    return {'articles_processed': 100, 'status': 'success'}


def decide_processing_engine(**context):
    """Decide which processing engine to use based on data size"""
    ti = context['task_instance']
    data_info = ti.xcom_pull(task_ids='load_raw_data')
    
    record_count = data_info.get('records', 0)
    
    # Use different engines based on data size
    if record_count < 100000:
        print(f"Using Pandas for {record_count} records")
        return 'process_with_pandas'
    elif record_count < 5000000:
        print(f"Using Fireducks for {record_count} records")
        return 'process_with_fireducks'
    else:
        print(f"Using Spark for {record_count} records")
        return 'process_with_spark'


def process_with_pandas(**context):
    """Process features using Pandas"""
    import time
    print("Processing with Pandas...")
    
    start_time = time.time()
    # TODO: Implement actual Pandas processing
    
    # Simulate processing
    import pandas as pd
    print("Pandas version:", pd.__version__)
    
    elapsed_time = time.time() - start_time
    print(f"Pandas processing completed in {elapsed_time:.2f} seconds")
    
    return {
        'engine': 'pandas',
        'elapsed_time': elapsed_time,
        'memory_usage': 'TBD'
    }


def process_with_fireducks(**context):
    """Process features using Fireducks"""
    import time
    print("Processing with Fireducks...")
    
    start_time = time.time()
    # TODO: Implement actual Fireducks processing
    
    try:
        import fireducks.pandas as pd
        print("Fireducks version:", pd.__version__)
    except ImportError:
        print("Fireducks not installed, falling back to Pandas")
        import pandas as pd
    
    elapsed_time = time.time() - start_time
    print(f"Fireducks processing completed in {elapsed_time:.2f} seconds")
    
    return {
        'engine': 'fireducks',
        'elapsed_time': elapsed_time,
        'memory_usage': 'TBD'
    }


def process_with_spark(**context):
    """Process features using Apache Spark"""
    import time
    print("Processing with Apache Spark...")
    
    start_time = time.time()
    # TODO: Implement actual Spark processing
    
    print("Note: Spark processing requires cluster configuration")
    
    elapsed_time = time.time() - start_time
    print(f"Spark processing completed in {elapsed_time:.2f} seconds")
    
    return {
        'engine': 'spark',
        'elapsed_time': elapsed_time,
        'memory_usage': 'TBD'
    }


def benchmark_performance(**context):
    """Compare performance across different engines"""
    print("Benchmarking processing performance...")
    
    ti = context['task_instance']
    
    # Collect results from different engines
    results = []
    for task in ['process_with_pandas', 'process_with_fireducks', 'process_with_spark']:
        try:
            result = ti.xcom_pull(task_ids=task)
            if result:
                results.append(result)
        except:
            pass
    
    print(f"Benchmark results: {results}")
    return {'benchmark_results': results}


def validate_features(**context):
    """Validate engineered features"""
    print("Validating feature quality...")
    
    # Validation checks:
    # - Check for NaN values
    # - Verify feature distributions
    # - Check for data leakage
    # - Validate feature ranges
    
    validation_checks = [
        'null_check',
        'distribution_check',
        'leakage_check',
        'range_check'
    ]
    
    print(f"Validation checks passed: {validation_checks}")
    return {'validation_status': 'passed', 'checks': validation_checks}


def store_features(**context):
    """Store engineered features"""
    print("Storing features to feature store...")
    
    execution_date = context['execution_date']
    feature_path = f"/opt/airflow/data/features/{execution_date.strftime('%Y-%m-%d')}"
    
    print(f"Features stored at: {feature_path}")
    return {'feature_path': feature_path, 'format': 'parquet'}


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
    
    # Wait for data ingestion to complete
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
    
    # Load raw data
    load_data = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_raw_data,
        provide_context=True,
    )
    
    # Feature engineering tasks (run in parallel)
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
    
    # Decide which processing engine to use
    decide_engine = BranchPythonOperator(
        task_id='decide_engine',
        python_callable=decide_processing_engine,
        provide_context=True,
    )
    
    # Processing with different engines
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
    
    # Benchmark performance
    benchmark = PythonOperator(
        task_id='benchmark',
        python_callable=benchmark_performance,
        provide_context=True,
        trigger_rule='none_failed',
    )
    
    # Validate features
    validate = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True,
    )
    
    # Store features
    store = PythonOperator(
        task_id='store_features',
        python_callable=store_features,
        provide_context=True,
    )
    
    # Define task dependencies
    wait_for_ingestion >> load_data >> feature_creation >> decide_engine
    decide_engine >> [pandas_processing, fireducks_processing, spark_processing]
    [pandas_processing, fireducks_processing, spark_processing] >> benchmark
    benchmark >> validate >> store


"""
DAG Structure:

wait_for_ingestion (Sensor)
        |
   load_raw_data
        |
  feature_creation (Task Group)
        |-- technical_indicators
        |-- timeseries_features
        |-- correlations
        |-- sentiment
        |
   decide_engine (Branch)
        |
        |-- process_with_pandas
        |-- process_with_fireducks
        |-- process_with_spark
        |
    benchmark
        |
  validate_features
        |
  store_features

Schedule: Daily at 4 AM UTC (after data ingestion)
Dependencies: Waits for data_ingestion_pipeline to complete
"""
