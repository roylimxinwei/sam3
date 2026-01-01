"""
Configuration management using Pydantic Settings.
Supports environment variables and .env files.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    app_name: str = "Food Classifier API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS - Origins allowed to access the API
    cors_origins: List[str] = [
        "http://localhost:3000",      # React web dev
        "http://localhost:8081",      # React Native web
        "exp://localhost:19000",      # Expo development
        "http://10.0.2.2:8000",       # Android emulator
    ]

    # ==========================================
    # Hugging Face Model Settings
    # ==========================================
    hf_token: Optional[str] = None  # Set via HF_TOKEN env var
    
    # Food Classifier Model
    hf_classifier_repo: str = "xinwei6969/sgfood233_convnext_base"
    hf_classifier_filename: str = "sgfood233_convnext_base.onnx"
    
    # Supabase Configuration
    supabase_url: str = "https://miwbvvmcdgndbmwqtzzx.supabase.co"
    supabase_key: str = "sb_publishable_ZjzqdEybcPeUWCzcmAlkAg_SRkxQL-i"

    # Model paths
    model_path: Path = Path("../../testing/sgfood233_convnext_base.onnx")
    labels_path: Path = Path("../../testing//foodsg233_labels.json")
    
    # SAM3 Segmentation
    sam3_enabled: bool = True
    sam3_confidence_threshold: float = 0.5
    
    # Inference settings
    default_topk: int = 5
    max_topk: int = 10
    max_batch_size: int = 50
    
    # Nutrition scraping
    nutrition_headless: bool = True
    nutrition_cache_ttl: int = 3600  # 1 hour
    
    # File upload limits
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
