
"""
FastAPI application entry point.
Configures the application, middleware, and routes.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your segmentation functions
try:
    from testing.gradio_seg_white import (
        initialize_sam3, 
        initialize_model, 
        sam3_processor,
        predict_segmented_items,
        models
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import segmentation modules: {e}")
    MODELS_AVAILABLE = False
    sam3_processor = None
    models = {}

app = FastAPI(title="Food Segmentation API", version="1.0.0")

# Enable CORS - CRITICAL for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL: ["http://localhost:3000", "http://localhost:19006"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize SAM3 and food classifier on startup."""
    if MODELS_AVAILABLE:
        print("🚀 Initializing models...")
        initialize_sam3()
        initialize_model()
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Models not available - running in demo mode")


@app.get("/")
async def root():
    return {
        "message": "Food Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Check API health",
            "/segment": "POST - Segment and classify food items"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the API and models are ready."""
    sam3_ready = sam3_processor is not None
    return {
        "status": "healthy" if sam3_ready else "unhealthy",
        "sam3_available": sam3_ready,
        "classifier_loaded": len(models) > 0
    }


@app.post("/segment")
async def segment_food(
    file: UploadFile = File(...),
    prompt: str = "food",
    confidence: float = 0.5,
    topk: int = 3,
    return_masks: bool = False
):
    """
    Segment and classify food items in an image.
    
    Args:
        file: Image file (JPEG, PNG)
        prompt: Text prompt for segmentation (default: "food")
        confidence: Confidence threshold (0.0-1.0)
        topk: Number of top predictions per item
        return_masks: Include base64-encoded masks
    """
    if not MODELS_AVAILABLE or sam3_processor is None:
        raise HTTPException(status_code=503, detail="SAM3 not initialized. Check server logs.")
    
    try:
        # Read image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Run segmentation
        sam3_processor.confidence_threshold = confidence
        inference_state = sam3_processor.set_image(image)
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        if len(masks) == 0:
            return {
                "items": [],
                "total": 0,
                "message": f"No '{prompt}' items detected"
            }
        
        # Classify items
        cache = {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "prompt": prompt
        }
        
        predictions = predict_segmented_items(cache, list(range(len(masks))), topk=topk)
        
        # Build response
        items = []
        for i, pred in enumerate(predictions):
            box = boxes[i].detach().cpu().numpy().astype(int).tolist()
            score = float(scores[i].item()) if hasattr(scores[i], "item") else float(scores[i])
            
            item_data = {
                "index": i,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "bounding_box": {
                    "x1": box[0], "y1": box[1],
                    "x2": box[2], "y2": box[3]
                },
                "detection_score": score
            }
            
            if return_masks:
                mask_np = masks[i].detach().cpu().numpy().squeeze()
                mask_uint8 = (mask_np * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_uint8, mode='L')
                
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                item_data["mask_base64"] = base64.b64encode(buffer.getvalue()).decode()
            
            items.append(item_data)
        
        return {
            "items": items,
            "total": len(items),
            "image_size": {"width": image.width, "height": image.height},
            "prompt": prompt
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")