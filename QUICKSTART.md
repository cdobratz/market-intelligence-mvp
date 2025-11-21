# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Run Setup Script

```bash
./setup.sh
```

This will:
- Check prerequisites (Docker, Docker Compose)
- Install uv if needed
- Create virtual environment
- Install Python dependencies
- Create necessary directories
- Set up .env file

## Step 2: Add API Keys

Edit the `.env` file and add your API keys:

```bash
nano .env
```

### Get Your Free API Keys:

1. **Alpha Vantage** (Stock Data)
   - Visit: https://www.alphavantage.co/support/#api-key
   - Click "GET YOUR FREE API KEY TODAY"
   - Fill in basic info and get instant key
   - Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key_here`

2. **News API** (Financial News)
   - Visit: https://newsapi.org/register
   - Register with email
   - Get API key immediately
   - Add to `.env`: `NEWS_API_KEY=your_key_here`

3. **CoinGecko** (Crypto Data)
   - No API key required for free tier! ✅

## Step 3: Start Services

```bash
docker-compose up -d
```

Wait 2-3 minutes for services to initialize.

## Step 4: Verify Services

Check that all services are running:

```bash
docker-compose ps
```

You should see all services as "Up":
- postgres
- airflow-webserver
- airflow-scheduler
- mlflow
- redis
- jupyter

## Step 5: Access the UIs

Open in your browser:

| Service | URL | Login |
|---------|-----|-------|
| **Airflow** | http://localhost:8080 | airflow / airflow |
| **MLflow** | http://localhost:5000 | (no login) |
| **Jupyter** | http://localhost:8888 | token: jupyter |

## Step 6: Run First Pipeline

### Option A: Via Airflow UI

1. Go to http://localhost:8080
2. Login with `airflow` / `airflow`
3. Find `data_ingestion_pipeline` DAG
4. Click the "Play" button to trigger it
5. Watch it run in real-time!

### Option B: Via Command Line

```bash
docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline
```

## Step 7: Monitor Progress

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow-scheduler

# Airflow task logs (via UI)
Go to Airflow UI → DAGs → data_ingestion_pipeline → Graph → Click on task → View Log
```

### Check Data
```bash
# See what data was fetched
ls -lh data/raw/

# View in Python
python -c "import pandas as pd; df = pd.read_parquet('data/raw/stocks/date=*/stocks_*.parquet'); print(df.head())"
```

## Step 8: Explore in Jupyter

1. Open http://localhost:8888 (token: `jupyter`)
2. Navigate to `work/` directory
3. Create new notebook
4. Try this:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load stock data
df = pd.read_parquet('../data/raw/stocks/date=2024-01-15/stocks_2024-01-15.parquet')

# Quick analysis
print(df.head())
df.groupby('symbol')['close'].plot(legend=True, figsize=(12,6))
plt.title('Stock Prices')
plt.show()
```

## Common Commands

### Docker Management
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a service
docker-compose restart airflow-scheduler

# View logs
docker-compose logs -f [service-name]

# Remove everything (including data)
docker-compose down -v
```

### Airflow Management
```bash
# List DAGs
docker-compose exec airflow-scheduler airflow dags list

# Trigger a DAG
docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline

# Pause/Unpause a DAG
docker-compose exec airflow-scheduler airflow dags pause data_ingestion_pipeline
docker-compose exec airflow-scheduler airflow dags unpause data_ingestion_pipeline

# Clear task instances
docker-compose exec airflow-scheduler airflow tasks clear data_ingestion_pipeline
```

### Python Environment
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install new package
uv pip install package-name

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .
```

## Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker ps

# Reset everything
docker-compose down -v
docker-compose up -d
```

### Permission errors
```bash
# Fix Airflow permissions
sudo chown -R $(id -u):$(id -g) airflow/logs airflow/plugins data/
```

### Airflow DAG not appearing
```bash
# Check DAG file for errors
docker-compose exec airflow-scheduler python /opt/airflow/dags/data_ingestion.py

# Refresh DAGs
docker-compose restart airflow-scheduler
```

### API rate limits
- **Alpha Vantage**: 25 calls/day, 5 calls/minute
- **News API**: 100 calls/day
- Wait between calls or use demo mode

## Next Steps

1. ✅ **Explore the Roadmap**: `cat warp.md`
2. ✅ **Read Full README**: `cat README.md`
3. ✅ **Review DAGs**: Check out `airflow/dags/` folder
4. ✅ **Customize Features**: Edit feature engineering in `src/features/`
5. ✅ **Add Models**: Implement ML models in `src/models/`
6. ✅ **Run Benchmarks**: `python benchmarks/pandas_vs_fireducks.py`

## Getting Help

- **Project Issues**: Check `warp.md` for common issues
- **Airflow Docs**: https://airflow.apache.org/docs/
- **MLflow Docs**: https://mlflow.org/docs/
- **Python Errors**: Check logs with `docker-compose logs -f`

---

**🎉 You're all set! Happy coding!**

For a deeper dive, check out the full [README.md](README.md) and [warp.md](warp.md).
