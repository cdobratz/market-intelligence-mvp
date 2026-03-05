"""Pydantic models for API request/response schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single stock prediction."""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, GOOGL)")
    include_features: bool = Field(default=False, description="Include feature details in response")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    symbols: List[str] = Field(..., description="List of stock symbols")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    symbol: str
    prediction: float = Field(..., description="Predicted value (price change or direction)")
    direction: str = Field(..., description="bullish or bearish")
    confidence: float = Field(..., description="Confidence score 0-1")
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    features: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str = "1.0.0"
    services: Dict[str, str]


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""
    data_ingestion: str
    feature_engineering: str
    model_training: str
    last_run: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    accuracy: Optional[float] = None
    last_trained: Optional[str] = None
    features: List[str] = []


class ModelsResponse(BaseModel):
    """Response model for models list."""
    models: List[ModelInfo]
