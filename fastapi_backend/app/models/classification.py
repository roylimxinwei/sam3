"""Pydantic models for food classification endpoints."""
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single prediction result."""
    rank: int = Field(..., ge=1, description="Prediction rank (1 = top prediction)")
    label: str = Field(..., description="Predicted food class label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class ClassificationResponse(BaseModel):
    """Response for single image classification."""
    success: bool = True
    predictions: List[PredictionResult]
    top_prediction: str = Field(..., description="Top predicted food class")
    model_name: str = Field(..., description="Name of the model used")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "predictions": [
                    {"rank": 1, "label": "chicken_rice", "confidence": 0.85},
                    {"rank": 2, "label": "nasi_lemak", "confidence": 0.10},
                    {"rank": 3, "label": "fried_rice", "confidence": 0.03},
                ],
                "top_prediction": "chicken_rice",
                "model_name": "sgfood233_convnext_base"
            }
        }
    }


class BatchImageResult(BaseModel):
    """Result for a single image in batch processing."""
    filename: str
    success: bool = True
    top_prediction: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    """Response for batch image classification."""
    success: bool = True
    total_processed: int
    successful: int
    failed: int
    results: List[BatchImageResult]


class ClassificationRequest(BaseModel):
    """Request body for classification with base64 image."""
    image_base64: str = Field(..., description="Base64 encoded image")
    topk: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "topk": 5
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Response with model information."""
    model_name: str
    num_classes: int
    classes: List[str]
    is_loaded: bool
