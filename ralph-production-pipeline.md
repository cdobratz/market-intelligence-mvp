# Market Intelligence Platform: Production ML Pipeline Implementation

## Context
- Project currently deployed to Cloud Run (API, MLflow, Airflow, Scheduler)
- Phase 3 (ML Models) at 15% - only XGBoost baseline implemented
- Need to transition from synthetic data to real market data
- Docker containers showing warnings in Desktop
- DAGs recently wired up but untested end-to-end
- Target: Working predictions on live dashboard like interview PDF describes

## Priority Implementation (Interview-Ready Features)

### Phase 1: Docker Health and Infrastructure (Iterations 1-8)

1. Check Docker container status and fix warnings
   - Run: docker ps -a
   - Identify containers with restart loops or errors
   - Check logs: docker logs [container] --tail 50
   - Fix common issues: disk space, memory limits, port conflicts

2. Verify service connectivity
   - Test Airflow webserver: curl http://localhost:8080/health
   - Test MLflow: curl http://localhost:5000/health
   - Test PostgreSQL: docker exec postgres pg_isready
   - Test Redis: docker exec redis redis-cli ping

3. Validate Cloud Run deployment matches local
   - Compare environment variables
   - Check Cloud SQL connection strings
   - Verify artifact storage paths

4. Document current infrastructure state in CLAUDE.md

### Phase 2: Real Data Integration (Iterations 9-20)

5. Configure Alpha Vantage API (stocks)
   - Update .env with valid ALPHA_VANTAGE_API_KEY
   - Test connection: src/data/ingestion.py fetch for AAPL
   - Implement rate limiting (5 calls per min, 25 per day free tier)
   - Store raw data: data/raw/stocks/AAPL_daily.parquet

6. Configure CoinGecko API (crypto)
   - Test BTC, ETH price fetching
   - No API key needed for free tier
   - Store: data/raw/crypto/BTC_daily.parquet

7. Configure News API (sentiment)
   - Update .env with valid NEWS_API_KEY
   - Fetch financial news for symbols
   - Store: data/raw/news/YYYY-MM-DD.parquet

8. Update data_ingestion_pipeline DAG
   - Replace sample data generator with real API calls
   - Add retry logic with exponential backoff
   - Implement data validation (Pandera schemas)
   - Schedule: Daily 2AM UTC

9. Trigger and verify data ingestion
   - Run: docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline
   - Check success: airflow/logs/
   - Verify Parquet files created in data/raw/

10. Update feature_engineering_pipeline DAG
    - Point to real data sources (not synthetic)
    - Calculate 50+ technical indicators on real prices
    - Store: data/features/SYMBOL_features.parquet

### Phase 3: Model Training on Real Data (Iterations 21-35)

11. Retrain XGBoost on real market data
    - Load features from data/features/
    - Implement walk-forward validation (per interview guide)
    - Target: 5-day forward returns
    - Log to MLflow with experiment name 'real_data_baseline'

12. Implement Random Forest regressor
    - Use same features and validation strategy
    - Compare to XGBoost in MLflow
    - File: src/models/supervised/random_forest_model.py

13. Implement LightGBM regressor
    - Faster training than XGBoost
    - File: src/models/supervised/lightgbm_model.py

14. Implement direction classifier (Up/Down/Neutral)
    - XGBoost Classifier
    - Random Forest Classifier
    - Target: Binary direction (close at t+5 greater than close at t)
    - File: src/models/supervised/classification.py

15. Implement feature selection pipeline
    - Reduce 56 features to top 20 using mutual information
    - File: src/features/selection.py
    - Document which features selected

16. Update model_training_pipeline DAG
    - Train all models (XGBoost, RF, LightGBM, Classifier)
    - Log all experiments to MLflow
    - Register best model to MLflow Model Registry
    - Schedule: Weekly Sunday 6AM UTC

### Phase 4: API Integration and Predictions (Iterations 36-50)

17. Update FastAPI endpoint /predict
    - Load production model from MLflow Model Registry
    - Accept symbol parameter: /predict?symbol=AAPL
    - Fetch latest features from data/features/
    - Return: prediction, direction, confidence, current_price

18. Implement batch prediction endpoint /predict/batch
    - Accept list of symbols: POST /predict/batch with JSON body containing symbols array
    - Return predictions for all symbols

19. Update dashboard UI (src/api/templates/index.html)
    - Show real-time predictions for entered symbol
    - Display confidence scores (per interview PDF)
    - Show technical indicators (RSI, MACD, Bollinger)
    - Pipeline status indicator

20. Deploy to Cloud Run
    - Push code to trigger GitHub Actions
    - Verify deployment at Cloud Run URL
    - Test /predict endpoint with AAPL, GOOGL, MSFT
    - Verify MLflow tracking shows experiments

### Phase 5: End-to-End Validation (Iterations 51-60)

21. Execute full pipeline locally
    - Trigger data_ingestion_pipeline
    - Wait for completion, trigger feature_engineering_pipeline
    - Wait for completion, trigger model_training_pipeline
    - Check MLflow for logged experiments

22. Test API predictions
    - Test predict endpoint with symbol=AAPL
    - Verify response has: prediction, direction, confidence
    - Test on multiple symbols (GOOGL, MSFT, TSLA)

23. Verify interview-ready features
    - 50+ technical indicators ✓
    - Walk-forward validation ✓
    - MLflow experiment tracking ✓
    - Cloud Run deployment ✓
    - Live dashboard ✓
    - Real data ingestion ✓

24. Document metrics in CLAUDE.md
    - Model performance (R², directional accuracy)
    - API latency (target less than 100ms)
    - Pipeline execution times
    - Cost breakdown (should match interview PDF: 95 to 140 dollars per month)

## Success Criteria

When ALL of these are true, output <promise>PRODUCTION_READY</promise>:

✅ All Docker containers healthy (no warnings)
✅ Data ingestion fetches real data from Alpha Vantage, CoinGecko, News API
✅ Feature engineering creates 50+ indicators on real prices
✅ 4+ models trained (XGBoost, RF, LightGBM, Classifier)
✅ Models achieve R² greater than 0.35 (baseline from interview guide)
✅ MLflow tracks all experiments with metrics
✅ API serves predictions at Cloud Run URL
✅ Dashboard shows predictions with confidence scores
✅ All 3 DAGs execute successfully end-to-end
✅ Project matches interview PDF capabilities

## After 60 Iterations If Not Complete

- Document blocking issues (API keys, rate limits, Cloud Run errors)
- List completed features vs remaining
- Provide manual steps for unautomated tasks
- Update roadmap.md with actual progress

## Key Files to Modify

- src/data/ingestion.py (add real API clients)
- airflow/dags/data_ingestion.py (replace synthetic data)
- airflow/dags/feature_engineering.py (point to real data)
- airflow/dags/model_training.py (train on real features)
- src/models/supervised/random_forest_model.py (new)
- src/models/supervised/lightgbm_model.py (new)
- src/models/supervised/classification.py (new)
- src/features/selection.py (new)
- src/api/main.py (update /predict endpoint)
- CLAUDE.md (update deployment status, metrics)
- docs/roadmap.md (update Phase 3 progress)

## Docker Health Checks

- docker system df (check disk usage)
- docker stats --no-stream (check memory and CPU)
- docker-compose logs -f --tail=20 (check for errors)
- Fix warnings before proceeding to data integration

## API Key Validation

- Test Alpha Vantage with curl command
- Test News API with curl command
- Store in .env file, never commit to git

## MLflow Best Practices (from interview PDF)

- Log hyperparameters: n_estimators, max_depth, learning_rate
- Log metrics: RMSE, MAE, R², directional_accuracy
- Log artifacts: feature_importance.png, confusion_matrix.png
- Tag experiments: 'real_data', 'production_candidate'
- Register models: name='market-predictor', stage='Production'

## Implementation Strategy

Each iteration should:
1. Work on one specific task from the phases above
2. Test the implementation immediately
3. Log results to console
4. Update documentation if needed
5. Move to next task only after verification

Focus on getting end-to-end flow working before optimization.
Prioritize features that are demo-able in interview setting.
