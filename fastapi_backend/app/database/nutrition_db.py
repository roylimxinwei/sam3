"""
Supabase database layer for nutrition data.
"""
from typing import Dict, Optional, Any
from supabase import create_client, Client

from app.config import settings


class NutritionDB:
    """Handles nutrition data storage in Supabase."""
    
    TABLE_NAME = "nutrition_cache"
    
    def __init__(self):
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
        return self._client
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get nutrition data by exact food name.
        
        Args:
            name: Food name to look up
            
        Returns:
            Nutrition data dict or None if not found
        """
        try:
            response = (
                self.client
                .table(self.TABLE_NAME)
                .select("*")
                .ilike("name", name)
                .limit(1)
                .execute()
            )
            
            if response.data and len(response.data) > 0:
                return self._format_db_record(response.data[0])
            return None
            
        except Exception as e:
            print(f"Database lookup error: {e}")
            return None
    
    def search(self, query: str, limit: int = 5) -> list[Dict[str, Any]]:
        """
        Search for foods by name (partial match).
        
        Args:
            query: Search term
            limit: Max results
            
        Returns:
            List of matching foods
        """
        try:
            response = (
                self.client
                .table(self.TABLE_NAME)
                .select("*")
                .ilike("name", f"%{query}%")
                .limit(limit)
                .execute()
            )
            
            return [self._format_db_record(r) for r in response.data]
            
        except Exception as e:
            print(f"Database search error: {e}")
            return []
    
    def save(self, nutrition_data: Dict[str, Any]) -> bool:
        """
        Save nutrition data to database.
        
        Args:
            nutrition_data: Full nutrition response from web scraper
            
        Returns:
            True if saved successfully
        """
        try:
            name = nutrition_data.get("name", "")
            if not name:
                return False
            
            # Check if already exists
            existing = self.get_by_name(name)
            if existing:
                print(f"⏭️  Already cached: {name}")
                return True
            
            record = {
                "name": name,
                "description": nutrition_data.get("description"),
                "default_serving_size": nutrition_data.get("default_serving_size"),
                "nutrition_per_100g": nutrition_data.get("nutrition_per_100g", {}),
                "nutrition_per_serving": nutrition_data.get("nutrition_per_serving"),
                "extra_info": nutrition_data.get("extra_info", {}),
            }
            
            response = (
                self.client
                .table(self.TABLE_NAME)
                .insert(record)
                .execute()
            )
            
            if response.data:
                print(f"✅ Cached to Supabase: {name}")
                return True
            return False
            
        except Exception as e:
            print(f"Database save error: {e}")
            return False
    
    def _format_db_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Format database record to match web scraper response format."""
        return {
            "name": record.get("name"),
            "description": record.get("description"),
            "default_serving_size": record.get("default_serving_size"),
            "nutrition_per_100g": record.get("nutrition_per_100g", {}),
            "nutrition_per_serving": record.get("nutrition_per_serving"),
            "extra_info": record.get("extra_info", {}),
        }


# Singleton
_db_instance: Optional[NutritionDB] = None


def get_nutrition_db() -> NutritionDB:
    """Get the global NutritionDB instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = NutritionDB()
    return _db_instance