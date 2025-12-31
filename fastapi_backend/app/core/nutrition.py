# app/core/nutrition.py
"""
Core nutrition lookup service.
Handles web scraping for nutrition information.
"""
import re
from typing import Dict, List, Optional

from app.config import settings
from app.database.nutrition_db import get_nutrition_db
from app.utils.exceptions import NutritionLookupError


class NutritionService:
    """
    Nutrition lookup service using web scraping.
    """
    
    KEY_NUTRIENTS = [
        "Energy (kcal)",
        "Protein (g)",
        "Carbohydrate (g)",
        "Total Fat (g)",
        "Dietary Fibre (g)"
    ]
    
    def __init__(self):
        self._cache: Dict = {}
        self.db = get_nutrition_db()
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for food items matching the query.
        
        Args:
            query: Search term
            max_results: Maximum number of results
            
        Returns:
            List of search results with index and name
        """
        if not query or not query.strip():
            raise NutritionLookupError("Search query cannot be empty")
        
        try:
            # Import here to avoid loading selenium at startup
            from app.services.web_scraper import search_foods
            
            results = search_foods(
                search_term=query.strip(),
                headless=settings.nutrition_headless,
                max_results=max_results
            )
            
            if not results:
                return []
            
            # Cache for later use
            self._cache["last_search"] = {
                "query": query.strip(),
                "results": results
            }
            
            return [
                {"index": i, "name": name}
                for i, name in enumerate(results)
            ]
            
        except ImportError:
            raise NutritionLookupError("Web scraper dependencies not installed")
        except Exception as e:
            raise NutritionLookupError(f"Search failed: {str(e)}")
        
    def get_cached(self, name: str) -> Optional[Dict]:
        """
        Check if food exists in database cache.
        
        Args:
            name: Exact food name to look up
            
        Returns:
            Cached nutrition data or None if not found
        """
        result = self.db.get_by_name(name)
        if result:
            result["source"] = "database"
            return result
        return None
    
    def get_nutrition_details(self, query: str, index: int) -> Dict:
        """
        Get detailed nutrition information for a specific food item.
        
        Args:
            query: Search term used to find the food
            index: Index of the selected food item (0-based)
            
        Returns:
            Dictionary containing nutrition details
        """
        try:
            from app.services.web_scraper import get_nutrition_details
            
            nutrition_data = get_nutrition_details(
                search_term=query.strip(),
                result_index=index,
                headless=settings.nutrition_headless
            )
            
            if not nutrition_data:
                raise NutritionLookupError("Failed to fetch nutrition data")
            
            result = {
                "name": nutrition_data.get("name", "Unknown"),
                "description": nutrition_data.get("description"),
                "nutrition_per_100g": nutrition_data.get("nutrition", {}),
                "extra_info": nutrition_data.get("extra_info", {}),
                "source": "web"
            }
            
            # Extract serving size
            default_size = result["extra_info"].get("Default Serving Size", "")
            result["default_serving_size"] = default_size
            
            # Calculate per-serving values
            per_serving = self._calculate_per_serving(
                nutrition_data.get("nutrition", {}),
                default_size
            )
            if per_serving:
                result["nutrition_per_serving"] = per_serving

            # Cache to database for future lookups
            self.db.save(result)
            
            return result
            
        except NutritionLookupError:
            raise
        except ImportError:
            raise NutritionLookupError("Web scraper dependencies not installed")
        except Exception as e:
            raise NutritionLookupError(f"Failed to get nutrition details: {str(e)}")
    
    def _calculate_per_serving(
        self, 
        nutrition_per_100g: Dict[str, str],
        serving_size: str
    ) -> Optional[Dict[str, float]]:
        """Calculate nutrition per serving from per-100g values."""
        if not serving_size:
            return None
        
        match = re.search(r"(\d+(?:\.\d+)?)\s*g", serving_size)
        if not match:
            return None
        
        grams = float(match.group(1))
        
        per_serving = {}
        for nutrient, value in nutrition_per_100g.items():
            if nutrient not in self.KEY_NUTRIENTS:
                continue
            
            try:
                if isinstance(value, str):
                    value = value.replace(",", "")
                    if value.replace(".", "", 1).isdigit():
                        numeric_value = float(value)
                    else:
                        continue
                else:
                    numeric_value = float(value)
                
                per_serving[nutrient] = round(numeric_value * grams / 100, 2)
            except (ValueError, TypeError):
                continue
        
        return per_serving if per_serving else None
    
    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()


# Singleton instance
_nutrition_instance: Optional[NutritionService] = None


def get_nutrition_service() -> NutritionService:
    """Get the global nutrition service instance."""
    global _nutrition_instance
    
    if _nutrition_instance is None:
        _nutrition_instance = NutritionService()
    
    return _nutrition_instance