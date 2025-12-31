"""Utility functions and exceptions."""
from app.utils.exceptions import (
    FoodClassifierError,
    ModelNotLoadedError,
    ImageProcessingError,
    SegmentationError,
    NutritionLookupError,
    InvalidFileError,
    SAM3NotAvailableError,
)
from app.utils.image_processing import (
    preprocess_image,
    decode_base64_image,
    encode_image_base64,
    validate_image_file,
    resize_for_display,
)

__all__ = [
    # Exceptions
    "FoodClassifierError",
    "ModelNotLoadedError",
    "ImageProcessingError",
    "SegmentationError",
    "NutritionLookupError",
    "InvalidFileError",
    "SAM3NotAvailableError",
    # Image processing
    "preprocess_image",
    "decode_base64_image",
    "encode_image_base64",
    "validate_image_file",
    "resize_for_display",
]
