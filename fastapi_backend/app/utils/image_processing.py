"""Image processing utilities for food classification."""
import base64
import io
from typing import Tuple, Union

import numpy as np
from PIL import Image
from torchvision import transforms

from app.utils.exceptions import ImageProcessingError


# Standard ImageNet normalization transform
def get_transform() -> transforms.Compose:
    """Get the standard preprocessing transform for food classification."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


_transform = get_transform()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL image for ONNX model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array with shape (1, 3, 224, 224)
    """
    try:
        img = image.convert("RGB")
        input_tensor = _transform(img).unsqueeze(0).numpy()
        return input_tensor
    except Exception as e:
        raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 encoded image string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string (with or without data URI prefix)
        
    Returns:
        PIL Image object
    """
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception as e:
        raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}")


def encode_image_base64(image: Union[Image.Image, np.ndarray], format: str = "JPEG") -> str:
    """
    Encode an image to base64 string.
    
    Args:
        image: PIL Image or numpy array
        format: Output format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        raise ImageProcessingError(f"Failed to encode image: {str(e)}")


def validate_image_file(filename: str, allowed_extensions: list) -> bool:
    """
    Validate that a file has an allowed image extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (without dots)
        
    Returns:
        True if valid, False otherwise
    """
    if not filename:
        return False
    
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in allowed_extensions


def resize_for_display(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize image for display while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Get image dimensions as (width, height)."""
    return image.size
