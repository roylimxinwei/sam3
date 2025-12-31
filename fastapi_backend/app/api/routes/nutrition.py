"""
Nutrition API routes.
Handles nutrition lookup endpoints.
"""
from fastapi import APIRouter, HTTPException, Query

from app.core.nutrition import get_nutrition_service
from app.models.nutrition import (
    NutritionSearchResponse,
    NutritionSearchResult,
    NutritionDetailResponse,
)
from app.utils.exceptions import NutritionLookupError

router = APIRouter(prefix="/nutrition", tags=["Nutrition"])


@router.get("/search", response_model=NutritionSearchResponse)
async def search_nutrition(
    query: str = Query(..., min_length=1, description="Food search term"),
    max_results: int = Query(default=5, ge=1, le=10)
):
    """
    Search for food items by name.
    
    - **query**: Search term (e.g., "chicken rice")
    - **max_results**: Maximum number of results (1-10)
    """
    service = get_nutrition_service()
    
    try:
        results = service.search(query, max_results=max_results)
        
        return NutritionSearchResponse(
            success=True,
            query=query,
            count=len(results),
            results=[
                NutritionSearchResult(**r) for r in results
            ]
        )
        
    except NutritionLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/cached")
async def get_cached_nutrition(
    name: str = Query(..., min_length=1, description="Exact food name to look up")
):
    """
    Check if food exists in database cache.
    Call this BEFORE /details/{index} to avoid unnecessary web scraping.
    
    Returns:
        - Cached nutrition data if found
        - 404 if not in cache (then call /details/{index})
    """
    service = get_nutrition_service()
    
    cached = service.get_cached(name)
    
    if cached:
        return NutritionDetailResponse(
            success=True,
            name=cached["name"],
            description=cached.get("description"),
            default_serving_size=cached.get("default_serving_size"),
            nutrition_per_100g=cached.get("nutrition_per_100g", {}),
            nutrition_per_serving=cached.get("nutrition_per_serving"),
            extra_info=cached.get("extra_info"),
            source="database"
        )
    
    raise HTTPException(
        status_code=404,
        detail=f"'{name}' not found in cache. Use /details/{{index}} to fetch."
    )


@router.get("/details/{index}", response_model=NutritionDetailResponse)
async def get_nutrition_details(
    index: int,
    query: str = Query(..., min_length=1, description="Original search term")
):
    """
    Get detailed nutrition information for a specific food item.
    
    - **index**: Index of the food item from search results (0-based)
    - **query**: Original search term used to find the food
    """
    service = get_nutrition_service()
    
    if index < 0:
        raise HTTPException(status_code=400, detail="Index must be non-negative")
    
    try:
        details = service.get_nutrition_details(query, index)
        
        return NutritionDetailResponse(
            success=True,
            name=details["name"],
            description=details.get("description"),
            default_serving_size=details.get("default_serving_size"),
            nutrition_per_100g=details.get("nutrition_per_100g", {}),
            nutrition_per_serving=details.get("nutrition_per_serving"),
            extra_info=details.get("extra_info")
        )
        
    except NutritionLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get details: {str(e)}")


@router.get("/quick", response_model=NutritionDetailResponse)
async def quick_nutrition_lookup(
    query: str = Query(..., min_length=1, description="Food name"),
    result_index: int = Query(default=0, ge=0, le=4, description="Which search result to use")
):
    """
    Quick nutrition lookup - searches and returns details in one call.
    
    This is a convenience endpoint that combines search and details retrieval.
    Useful for getting nutrition info directly from a food name.
    
    - **query**: Food name (e.g., "chicken rice")
    - **result_index**: Which search result to use (0 = first/best match)
    """
    service = get_nutrition_service()
    
    try:
        # First search
        results = service.search(query, max_results=5)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No results found for '{query}'"
            )
        
        if result_index >= len(results):
            result_index = 0
        
        # Then get details
        details = service.get_nutrition_details(query, result_index)
        
        return NutritionDetailResponse(
            success=True,
            name=details["name"],
            description=details.get("description"),
            default_serving_size=details.get("default_serving_size"),
            nutrition_per_100g=details.get("nutrition_per_100g", {}),
            nutrition_per_serving=details.get("nutrition_per_serving"),
            extra_info=details.get("extra_info")
        )
        
    except NutritionLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)}")
