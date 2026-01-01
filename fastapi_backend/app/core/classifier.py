"""
Core food classification service.
Handles model inference and prediction logic.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, ImageProcessingError
from app.utils.image_processing import preprocess_image
from app.utils.model_downloader import download_classifier_model

class FoodClassifier:
    """
    Food classification service using ONNX Runtime.
    
    This class handles model loading and inference for food classification.
    It's designed to be used as a singleton across the application.
    """
    
    def __init__(self):
        self._session: Optional[ort.InferenceSession] = None
        self._labels: Optional[List[str]] = None
        self._model_name: Optional[str] = None
        self._is_loaded: bool = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self._session is not None
    
    @property
    def model_name(self) -> Optional[str]:
        """Get loaded model name."""
        return self._model_name
    
    @property
    def labels(self) -> Optional[List[str]]:
        """Get class labels."""
        return self._labels
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self._labels) if self._labels else 0
    
    def _get_model_path(self) -> Path:
        """Get model path, downloading from HF if needed."""
        if settings.model_path.exists():
            return settings.model_path
    
        print("📥 Downloading model from Hugging Face...")
        return download_classifier_model(
            repo_id=settings.hf_classifier_repo,
            filename=settings.hf_classifier_filename,
        )
    def load_model(
        self, 
        model_path: Optional[Path] = None, 
        labels_path: Optional[Path] = None
    ) -> bool:
        """
        Load ONNX model and class labels.
        
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to labels JSON file
            
        Returns:
            True if successful, False otherwise
        """
        # model_path = model_path or settings.model_path
        model_path = model_path if (model_path and model_path.exists()) else self._get_model_path()
        labels_path = labels_path or settings.labels_path
        
        try:
            # Load class labels
            with open(labels_path, "r") as f:
                self._labels = json.load(f)
            
            # Create ONNX Runtime session with GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            
            self._model_name = model_path.stem
            self._is_loaded = True
            
            print(f"✅ Model loaded: {self._model_name} ({len(self._labels)} classes)")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            self._is_loaded = False
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._is_loaded = False
            return False
    
    def predict(
        self, 
        image: Image.Image, 
        topk: int = 5
    ) -> Tuple[List[Dict], str]:
        """
        Run prediction on a single image.
        
        Args:
            image: PIL Image object
            topk: Number of top predictions to return
            
        Returns:
            Tuple of (predictions list, top prediction label)
            
        Raises:
            ModelNotLoadedError: If model is not loaded
            ImageProcessingError: If image preprocessing fails
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Run inference
        outputs = self._session.run(None, {"input": input_tensor})
        logits = outputs[0][0]
        
        # Apply softmax to get probabilities
        probs = self._softmax(logits)
        
        # Get top-k predictions
        topk = min(topk, len(self._labels))
        topk_indices = np.argsort(probs)[::-1][:topk]
        
        predictions = []
        for rank, idx in enumerate(topk_indices, 1):
            predictions.append({
                "rank": rank,
                "label": self._labels[idx],
                "confidence": float(probs[idx])
            })
        
        top_prediction = self._labels[topk_indices[0]]
        
        return predictions, top_prediction
    
    def predict_batch(
        self, 
        images: List[Tuple[str, Image.Image]], 
        topk: int = 1
    ) -> List[Dict]:
        """
        Run predictions on multiple images.
        
        Args:
            images: List of (filename, PIL Image) tuples
            topk: Number of top predictions per image
            
        Returns:
            List of result dictionaries for each image
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()
        
        results = []
        
        for filename, image in images:
            try:
                predictions, top_prediction = self.predict(image, topk=topk)
                results.append({
                    "filename": filename,
                    "success": True,
                    "top_prediction": top_prediction,
                    "confidence": predictions[0]["confidence"] if predictions else None,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "top_prediction": None,
                    "confidence": None,
                    "error": str(e)
                })
        
        return results
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()


# Global classifier instance (singleton pattern)
_classifier_instance: Optional[FoodClassifier] = None


def get_classifier() -> FoodClassifier:
    """
    Get the global classifier instance.
    Creates and loads the model on first call.
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = FoodClassifier()
        _classifier_instance.load_model()
    
    return _classifier_instance


def initialize_classifier() -> bool:
    """
    Initialize the classifier on application startup.
    
    Returns:
        True if successful, False otherwise
    """
    classifier = get_classifier()
    return classifier.is_loaded
