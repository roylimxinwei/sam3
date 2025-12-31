"""API routes."""
from app.api.routes.classification import router as classification_router
from app.api.routes.nutrition import router as nutrition_router
from app.api.routes.segmentation import router as segmentation_router

__all__ = [
    "classification_router",
    "nutrition_router",
    "segmentation_router",
]
