"""
Hugging Face Model Downloader Utility.
Downloads ONNX models from Hugging Face Hub if not present locally.
"""
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


class ModelDownloader:
    """Handles downloading models from Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the downloader.
        
        Args:
            token: Hugging Face API token for private repos.
                   If None, uses HF_TOKEN environment variable.
        """
        self.token = token or os.getenv("HF_TOKEN")
    
    def download_model(
        self,
        repo_id: str,
        filename: str,
        local_dir: Optional[Path] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download a model file from Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "xinwei6969/sgfood233_convnext_base")
            filename: Name of the file to download (e.g., "sgfood233_convnext_base.onnx")
            local_dir: Local directory to save the model. If None, uses HF cache.
            force_download: If True, re-download even if file exists.
            
        Returns:
            Path to the downloaded model file.
            
        Raises:
            HfHubHTTPError: If download fails (e.g., unauthorized, not found)
        """
        print(f"📥 Checking for model: {repo_id}/{filename}")
        
        try:
            # Download from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self.token,
                local_dir=local_dir,
                force_download=force_download,
            )
            
            print(f"✅ Model ready: {model_path}")
            return Path(model_path)
            
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                raise RuntimeError(
                    f"❌ Authentication failed for {repo_id}. "
                    "Please check your HF_TOKEN environment variable."
                ) from e
            elif "404" in str(e):
                raise RuntimeError(
                    f"❌ Model not found: {repo_id}/{filename}. "
                    "Please check the repository and filename."
                ) from e
            else:
                raise
    
    def ensure_model_available(
        self,
        repo_id: str,
        filename: str,
        local_path: Optional[Path] = None,
    ) -> Path:
        """
        Ensure a model is available locally, downloading if necessary.
        
        First checks if the model exists at local_path. If not, downloads from HF.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Name of the file to download
            local_path: Optional local path to check first
            
        Returns:
            Path to the model file (local or downloaded)
        """
        # Check if model exists locally first
        if local_path and local_path.exists():
            print(f"✅ Using local model: {local_path}")
            return local_path
        
        # Download from Hugging Face
        return self.download_model(repo_id, filename)


# Global downloader instance
_downloader: Optional[ModelDownloader] = None


def get_model_downloader() -> ModelDownloader:
    """Get or create the global model downloader instance."""
    global _downloader
    if _downloader is None:
        _downloader = ModelDownloader()
    return _downloader


def download_classifier_model(
    repo_id: str = "xinwei6969/sgfood233_convnext_base",
    filename: str = "sgfood233_convnext_base.onnx",
    local_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function to download the food classifier model.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename
        local_path: Optional local path to check first
        
    Returns:
        Path to the model file
    """
    downloader = get_model_downloader()
    return downloader.ensure_model_available(repo_id, filename, local_path)