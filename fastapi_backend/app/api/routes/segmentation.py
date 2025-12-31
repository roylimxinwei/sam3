"""
Segmentation API routes.
Handles SAM3 food segmentation endpoints.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from PIL import Image
import io

from app.core.segmentation import get_segmentation_service
from app.models.segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    SegmentationStatusResponse,
    DetectionResult,
    BoundingBox,
)
from app.utils.exceptions import SAM3NotAvailableError, SegmentationError
from app.utils.image_processing import (
    decode_base64_image,
    encode_image_base64,
)

router = APIRouter(prefix="/segment", tags=["Segmentation"])


@router.get("/status", response_model=SegmentationStatusResponse)
async def get_segmentation_status():
    """Check if SAM3 segmentation is available."""
    service = get_segmentation_service()
    
    return SegmentationStatusResponse(
        available=service.is_available,
        message=service.status_message
    )


@router.post("", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """
    Segment food items in an image using text prompt.
    
    Use this endpoint from React Native by sending base64 encoded image data.
    
    - **image_base64**: Base64 encoded image
    - **prompt**: Text description of what to detect (e.g., "food", "egg tart")
    - **confidence_threshold**: Minimum confidence for detections (0.1-1.0)
    """
    service = get_segmentation_service()
    
    if not service.is_available:
        raise HTTPException(
            status_code=503,
            detail=service.status_message
        )
    
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Run segmentation
        result = service.segment(
            image=image,
            prompt=request.prompt,
            confidence_threshold=request.confidence_threshold
        )
        
        # Encode annotated image to base64
        annotated_base64 = None
        if result["annotated_image"] is not None:
            annotated_base64 = encode_image_base64(result["annotated_image"])
        
        return SegmentationResponse(
            success=True,
            prompt=request.prompt,
            num_detections=result["num_detections"],
            detections=[
                DetectionResult(
                    item_id=det["item_id"],
                    confidence=det["confidence"],
                    bounding_box=BoundingBox(**det["bounding_box"])
                )
                for det in result["detections"]
            ],
            annotated_image_base64=annotated_base64,
            message=result["message"]
        )
        
    except SAM3NotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SegmentationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.post("/upload", response_model=SegmentationResponse)
async def segment_uploaded_image(
    file: UploadFile = File(...),
    prompt: str = Query(default="food", min_length=1),
    confidence_threshold: float = Query(default=0.5, ge=0.1, le=1.0)
):
    """
    Segment food items in an uploaded image.
    
    - **file**: Image file to segment
    - **prompt**: Text description of what to detect
    - **confidence_threshold**: Minimum confidence for detections
    """
    service = get_segmentation_service()
    
    if not service.is_available:
        raise HTTPException(
            status_code=503,
            detail=service.status_message
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run segmentation
        result = service.segment(
            image=image,
            prompt=prompt,
            confidence_threshold=confidence_threshold
        )
        
        # Encode annotated image to base64
        annotated_base64 = None
        if result["annotated_image"] is not None:
            annotated_base64 = encode_image_base64(result["annotated_image"])
        
        return SegmentationResponse(
            success=True,
            prompt=prompt,
            num_detections=result["num_detections"],
            detections=[
                DetectionResult(
                    item_id=det["item_id"],
                    confidence=det["confidence"],
                    bounding_box=BoundingBox(**det["bounding_box"])
                )
                for det in result["detections"]
            ],
            annotated_image_base64=annotated_base64,
            message=result["message"]
        )
        
    except SAM3NotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SegmentationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
