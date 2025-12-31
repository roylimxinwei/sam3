"""
Core SAM3 segmentation service.
Handles food segmentation using SAM3 model.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.utils.exceptions import SAM3NotAvailableError, SegmentationError


class SegmentationService:
    """
    Food segmentation service using SAM3.
    
    This class handles model loading and inference for food segmentation.
    """
    
    def __init__(self):
        self._processor = None
        self._is_available: bool = False
        self._error_message: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Check if SAM3 is available."""
        return self._is_available
    
    @property
    def status_message(self) -> str:
        """Get status message."""
        if self._is_available:
            return "SAM3 segmentation model is available"
        return self._error_message or "SAM3 model not loaded"
    
    def initialize(self) -> bool:
        """
        Initialize SAM3 model.
        
        Returns:
            True if successful, False otherwise
        """
        if not settings.sam3_enabled:
            self._error_message = "SAM3 is disabled in configuration"
            return False
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print("Loading SAM3 model...")
            model = build_sam3_image_model()
            self._processor = Sam3Processor(model)
            self._is_available = True
            print("✅ SAM3 model loaded successfully")
            return True
            
        except ImportError as e:
            self._error_message = f"SAM3 not installed: {str(e)}"
            print(f"⚠️ SAM3 not available: {self._error_message}")
            return False
        except Exception as e:
            self._error_message = f"Failed to load SAM3: {str(e)}"
            print(f"⚠️ SAM3 not available: {self._error_message}")
            return False
    
    def segment(
        self,
        image: Image.Image,
        prompt: str,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Run segmentation on an image with a text prompt.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for detection (e.g., "food", "egg tart")
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary containing:
                - annotated_image: numpy array with masks and boxes drawn
                - detections: list of detection results
                - num_detections: number of items found
                - message: status message
                
        Raises:
            SAM3NotAvailableError: If SAM3 is not available
            SegmentationError: If segmentation fails
        """
        if not self._is_available:
            raise SAM3NotAvailableError(self._error_message)
        
        if not prompt or not prompt.strip():
            raise SegmentationError("Please provide a text prompt")
        
        try:
            # Set confidence threshold
            self._processor.confidence_threshold = confidence_threshold
            
            # Run SAM3 inference
            inference_state = self._processor.set_image(image)
            output = self._processor.set_text_prompt(
                state=inference_state, 
                prompt=prompt.strip()
            )
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            if len(masks) == 0:
                return {
                    "annotated_image": np.array(image),
                    "detections": [],
                    "num_detections": 0,
                    "message": f"No '{prompt}' items detected"
                }
            
            # Create annotated image
            annotated_image = self._draw_annotations(
                np.array(image).copy(), 
                masks, 
                boxes, 
                scores
            )
            
            # Build detection results
            detections = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                detections.append({
                    "item_id": i + 1,
                    "confidence": float(score),
                    "bounding_box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                })
            
            return {
                "annotated_image": annotated_image,
                "detections": detections,
                "num_detections": len(masks),
                "message": f"Found {len(masks)} '{prompt}' item(s)"
            }
            
        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {str(e)}")
    
    def _draw_annotations(
        self,
        img_array: np.ndarray,
        masks,
        boxes,
        scores
    ) -> np.ndarray:
        """
        Draw masks and bounding boxes on image.
        
        Args:
            img_array: Image as numpy array
            masks: Detection masks
            boxes: Bounding boxes
            scores: Confidence scores
            
        Returns:
            Annotated image as numpy array
        """
        # Generate colors for each detection
        np.random.seed(42)
        colors = [
            (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            for _ in range(len(masks))
        ]
        
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            color = colors[i]
            
            # Apply mask overlay
            mask_np = mask.cpu().numpy().squeeze()
            overlay = img_array.copy()
            overlay[mask_np > 0] = [color[0], color[1], color[2]]
            img_array = np.where(
                mask_np[:, :, None] > 0,
                (0.6 * img_array + 0.4 * overlay).astype(np.uint8),
                img_array
            )
            
            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            thickness = 3
            img_array[y1:y1+thickness, x1:x2] = color  # Top
            img_array[y2-thickness:y2, x1:x2] = color  # Bottom
            img_array[y1:y2, x1:x1+thickness] = color  # Left
            img_array[y1:y2, x2-thickness:x2] = color  # Right
        
        return img_array


# Global instance (singleton pattern)
_segmentation_instance: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """Get the global segmentation service instance."""
    global _segmentation_instance
    
    if _segmentation_instance is None:
        _segmentation_instance = SegmentationService()
        _segmentation_instance.initialize()
    
    return _segmentation_instance


def initialize_segmentation() -> bool:
    """
    Initialize segmentation service on application startup.
    
    Returns:
        True if successful, False otherwise
    """
    service = get_segmentation_service()
    return service.is_available
