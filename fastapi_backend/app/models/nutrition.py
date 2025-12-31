"""Pydantic models for nutrition lookup endpoints."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NutritionSearchResult(BaseModel):
    """Single search result item."""
    index: int
    name: str


class NutritionSearchResponse(BaseModel):
    """Response for nutrition search."""
    success: bool = True
    query: str
    count: int
    results: List[NutritionSearchResult]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "query": "chicken rice",
                "count": 5,
                "results": [
                    {"index": 0, "name": "Chicken Rice, Hainanese"},
                    {"index": 1, "name": "Chicken Rice, Roasted"},
                    {"index": 2, "name": "Chicken Rice, Steamed"},
                ]
            }
        }
    }


class NutrientInfo(BaseModel):
    """Single nutrient information."""
    name: str
    per_100g: str
    per_serving: Optional[float] = None


class NutritionDetailResponse(BaseModel):
    """Detailed nutrition information response."""
    success: bool = True
    name: str
    description: Optional[str] = None
    default_serving_size: Optional[str] = None
    nutrition_per_100g: Dict[str, str]
    nutrition_per_serving: Optional[Dict[str, float]] = None
    extra_info: Optional[Dict[str, str]] = None
    source: Optional[str] = None  # "database" or "web"
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "name": "Chicken Rice, Hainanese",
                "description": "Steamed chicken served with fragrant rice...",
                "default_serving_size": "300g",
                "nutrition_per_100g": {
                    "Energy (kcal)": "168",
                    "Protein (g)": "12.5",
                    "Carbohydrate (g)": "18.2",
                    "Total Fat (g)": "5.3",
                },
                "nutrition_per_serving": {
                    "Energy (kcal)": 504.0,
                    "Protein (g)": 37.5,
                    "Carbohydrate (g)": 54.6,
                    "Total Fat (g)": 15.9,
                }
            }
        }
    }


class NutritionErrorResponse(BaseModel):
    """Error response for nutrition endpoints."""
    success: bool = False
    error: str
    query: Optional[str] = None
