"""
FastAPI application for Financial Market Intelligence MVP.

Provides REST API endpoints for:
- Stock predictions
- Model information
- Pipeline status
- Landing page dashboard
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PipelineStatusResponse,
    ModelsResponse,
    ModelInfo,
)
from src.api.service import get_prediction_service, PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Financial Market Intelligence API...")
    service = get_prediction_service()
    logger.info("Prediction service initialized")
    yield
    # Shutdown
    logger.info("Shutting down Financial Market Intelligence API...")


# Create FastAPI application
app = FastAPI(
    title="Financial Market Intelligence API",
    description="ML-powered stock prediction and market analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Setup static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Landing page with dashboard."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Financial Market Intelligence"}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services={
            "api": "ok",
            "mlflow": "ok",
            "airflow": "ok"
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single stock symbol."""
    service = get_prediction_service()
    
    try:
        result = service.predict(
            symbol=request.symbol.upper(),
            include_features=request.include_features
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple stock symbols."""
    service = get_prediction_service()
    
    try:
        results = service.predict_batch([s.upper() for s in request.symbols])
        
        predictions = []
        for result in results:
            if "error" not in result:
                predictions.append(PredictionResponse(**result))
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """Get pipeline status."""
    service = get_prediction_service()
    
    try:
        status = service.get_pipeline_status()
        return PipelineStatusResponse(**status)
    
    except Exception as e:
        logger.error(f"Pipeline status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    service = get_prediction_service()
    
    try:
        models_info = service.get_models_info()
        models = [ModelInfo(**m) for m in models_info]
        return ModelsResponse(models=models)
    
    except Exception as e:
        logger.error(f"Models list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
