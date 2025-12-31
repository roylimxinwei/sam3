"""Pydantic models for API request/response schemas."""
from app.models.classification import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationResponse,
    BatchImageResult,
    PredictionResult,
    ModelInfoResponse,
)
from app.models.nutrition import (
    NutritionSearchResponse,
    NutritionSearchResult,
    NutritionDetailResponse,
    NutrientInfo,
)
from app.models.segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    SegmentationStatusResponse,
    DetectionResult,
    BoundingBox,
)

__all__ = [
    # Classification
    "ClassificationRequest",
    "ClassificationResponse",
    "BatchClassificationResponse",
    "BatchImageResult",
    "PredictionResult",
    "ModelInfoResponse",
    # Nutrition
    "NutritionSearchResponse",
    "NutritionSearchResult",
    "NutritionDetailResponse",
    "NutrientInfo",
    # Segmentation
    "SegmentationRequest",
    "SegmentationResponse",
    "SegmentationStatusResponse",
    "DetectionResult",
    "BoundingBox",
]
