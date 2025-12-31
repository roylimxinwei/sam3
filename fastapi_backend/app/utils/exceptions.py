"""Custom exceptions for the food classifier application."""


class FoodClassifierError(Exception):
    """Base exception for food classifier errors."""
    pass


class ModelNotLoadedError(FoodClassifierError):
    """Raised when model is not loaded."""
    def __init__(self, message: str = "Model not loaded. Please check configuration."):
        super().__init__(message)


class ImageProcessingError(FoodClassifierError):
    """Raised when image processing fails."""
    pass


class SegmentationError(FoodClassifierError):
    """Raised when segmentation fails."""
    pass


class NutritionLookupError(FoodClassifierError):
    """Raised when nutrition lookup fails."""
    pass


class InvalidFileError(FoodClassifierError):
    """Raised when file validation fails."""
    pass


class SAM3NotAvailableError(FoodClassifierError):
    """Raised when SAM3 model is not available."""
    def __init__(self, message: str = "SAM3 segmentation model is not available."):
        super().__init__(message)
