"""
Classification API routes.
Handles food classification endpoints for React Native and web clients.
"""
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from PIL import Image

from app.config import settings
from app.core.classifier import get_classifier
from app.models.classification import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationResponse,
    BatchImageResult,
    PredictionResult,
    ModelInfoResponse,
)
from app.utils.exceptions import ModelNotLoadedError, ImageProcessingError
from app.utils.image_processing import (
    decode_base64_image,
    validate_image_file,
)

router = APIRouter(prefix="/classify", tags=["Classification"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    classifier = get_classifier()
    
    return ModelInfoResponse(
        model_name=classifier.model_name or "Not loaded",
        num_classes=classifier.num_classes,
        classes=classifier.labels or [],
        is_loaded=classifier.is_loaded
    )


@router.post("", response_model=ClassificationResponse)
async def classify_image(
    request: ClassificationRequest
):
    """
    Classify a food image from base64 encoded data.
    
    Use this endpoint from React Native by sending base64 encoded image data.
    
    - **image_base64**: Base64 encoded image (with or without data URI prefix)
    - **topk**: Number of top predictions to return (1-10)
    """
    classifier = get_classifier()
    
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Run prediction
        predictions, top_prediction = classifier.predict(image, topk=request.topk)
        
        return ClassificationResponse(
            success=True,
            predictions=[
                PredictionResult(**pred) for pred in predictions
            ],
            top_prediction=top_prediction,
            model_name=classifier.model_name
        )
        
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/upload", response_model=ClassificationResponse)
async def classify_uploaded_image(
    file: UploadFile = File(...),
    topk: int = Query(default=5, ge=1, le=10)
):
    """
    Classify a food image from file upload.
    
    Use this endpoint for traditional file uploads.
    
    - **file**: Image file (JPEG, PNG, WebP, BMP)
    - **topk**: Number of top predictions to return (1-10)
    """
    classifier = get_classifier()
    
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate file type
    if not validate_image_file(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run prediction
        predictions, top_prediction = classifier.predict(image, topk=topk)
        
        return ClassificationResponse(
            success=True,
            predictions=[
                PredictionResult(**pred) for pred in predictions
            ],
            top_prediction=top_prediction,
            model_name=classifier.model_name
        )
        
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    topk: int = Query(default=1, ge=1, le=10)
):
    """
    Classify multiple food images.
    
    - **files**: Multiple image files
    - **topk**: Number of top predictions per image (1-10)
    """
    classifier = get_classifier()
    
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.max_batch_size} images"
        )
    
    # Process all images
    images = []
    for file in files:
        if not validate_image_file(file.filename, settings.allowed_extensions):
            images.append((file.filename, None))
            continue
        
        try:
            contents = await file.read()
            import io
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append((file.filename, image))
        except Exception:
            images.append((file.filename, None))
    
    # Run batch prediction
    results = []
    successful = 0
    failed = 0
    
    for filename, image in images:
        if image is None:
            results.append(BatchImageResult(
                filename=filename,
                success=False,
                error="Failed to load image"
            ))
            failed += 1
            continue
        
        try:
            predictions, top_prediction = classifier.predict(image, topk=topk)
            results.append(BatchImageResult(
                filename=filename,
                success=True,
                top_prediction=top_prediction,
                confidence=predictions[0]["confidence"] if predictions else None
            ))
            successful += 1
        except Exception as e:
            results.append(BatchImageResult(
                filename=filename,
                success=False,
                error=str(e)
            ))
            failed += 1
    
    return BatchClassificationResponse(
        success=True,
        total_processed=len(files),
        successful=successful,
        failed=failed,
        results=results
    )
