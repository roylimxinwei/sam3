"""Pydantic models for SAM3 segmentation endpoints."""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    """Single detection result from segmentation."""
    item_id: int = Field(..., description="Detection item number")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bounding_box: BoundingBox


class SegmentationRequest(BaseModel):
    """Request body for segmentation with base64 image."""
    image_base64: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(default="food", description="Text prompt for detection")
    confidence_threshold: float = Field(
        default=0.5, 
        ge=0.1, 
        le=1.0, 
        description="Minimum confidence threshold"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "prompt": "egg tart",
                "confidence_threshold": 0.5
            }
        }
    }


class SegmentationResponse(BaseModel):
    """Response for image segmentation."""
    success: bool = True
    prompt: str
    num_detections: int
    detections: List[DetectionResult]
    annotated_image_base64: Optional[str] = Field(
        None, 
        description="Base64 encoded annotated image with masks and boxes"
    )
    message: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "prompt": "egg tart",
                "num_detections": 3,
                "detections": [
                    {
                        "item_id": 1,
                        "confidence": 0.92,
                        "bounding_box": {"x1": 100, "y1": 150, "x2": 250, "y2": 300}
                    },
                    {
                        "item_id": 2,
                        "confidence": 0.87,
                        "bounding_box": {"x1": 300, "y1": 140, "x2": 450, "y2": 290}
                    },
                ],
                "annotated_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
                "message": "Found 3 'egg tart' item(s)"
            }
        }
    }


class SegmentationStatusResponse(BaseModel):
    """Response for segmentation service status."""
    available: bool
    message: str
