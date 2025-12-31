# scripts/populate_supabase.py
import sys
import time
import os
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from app.database.nutrition_db import get_nutrition_db
from app.services.web_scraper import search_foods, get_nutrition_details
import re

# Common Singapore foods
COMMON_FOODS = [
    "egg tart"
#     "chicken rice",
#     "nasi lemak", 
#     "char kway teow",
#     "laksa",
#     "mee goreng",
#     "roti prata",
#     "fish ball noodle",
#     "bak chor mee",
#     "hokkien mee",
#     "satay",
#     "curry puff",
#     "kaya toast",
#     "carrot cake",
#     "popiah",
#     "chilli crab",
#     "ban mian",
#     "yong tau foo",
#     "duck rice",
#     "wanton mee",
#     "prawn noodle",
#     "beef noodle",
#     "fried rice",
#     "white rice",
#     "brown rice",
#     "egg",
#     "vegetables",
#     "tofu",
#     "chicken breast",
#     "fish fillet",
#     "pork belly",
#     "beef steak",
#     "nasi briyani",
#     "mee siam",
#     "mee rebus",
#     "ice kachang",
#     "chendol",
#     "tau huay",
#     "you tiao",
#     "chwee kueh",
#     "soon kueh",
#     "kueh lapis",
#     "ondeh ondeh",
#     "ang ku kueh",
]

def parse_number(value: str) -> float:
    if not value:
        return None
    try:
        match = re.search(r"[\d.]+", str(value))
        return float(match.group()) if match else None
    except:
        return None

def populate():
    db = get_nutrition_db()
    
    total = len(COMMON_FOODS)
    success = 0
    skipped = 0
    failed = 0
    
    print(f"🚀 Starting population of {total} foods to Supabase...\n")
    
    for i, food_name in enumerate(COMMON_FOODS):
        print(f"[{i+1}/{total}] {food_name}")
        
        # Check if exists
        if db.food_exists(food_name):
            print(f"  ⏭️  Already exists")
            skipped += 1
            continue
        
        try:
            # Search
            print(f"  🔍 Searching...")
            results = search_foods(food_name, max_results=1)
            
            if not results:
                print(f"  ❌ No results")
                failed += 1
                continue
            
            # Get details
            print(f"  📊 Getting details...")
            details = get_nutrition_details(food_name, result_index=0)
            
            if not details:
                print(f"  ❌ No details")
                failed += 1
                continue
            
            # Parse and save
            nutrition = details.get("nutrition_per_100g", {})
            extra_info = details.get("extra_info", {})
            
            food_data = {
                "name": details.get("name", food_name),
                "description": details.get("description"),
                "food_group": extra_info.get("Food Group"),
                "default_serving_size": extra_info.get("Serving Size"),
                "energy_kcal": parse_number(nutrition.get("Energy", "")),
                "protein_g": parse_number(nutrition.get("Protein", "")),
                "total_fat_g": parse_number(nutrition.get("Total Fat", "")),
                "saturated_fat_g": parse_number(nutrition.get("Saturated Fat", "")),
                "carbohydrate_g": parse_number(nutrition.get("Carbohydrate", "")),
                "dietary_fibre_g": parse_number(nutrition.get("Dietary Fibre", "")),
                "sugars_g": parse_number(nutrition.get("Sugars", "")),
                "sodium_mg": parse_number(nutrition.get("Sodium", "")),
                "cholesterol_mg": parse_number(nutrition.get("Cholesterol", "")),
                "source": "hpb_web",
            }
            
            result = db.add_food(food_data)
            if result:
                print(f"  ✅ Saved: {food_data['name']}")
                success += 1
            else:
                print(f"  ❌ Failed to save")
                failed += 1
            
            # Rate limiting
            time.sleep(3)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Success: {success}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    populate()