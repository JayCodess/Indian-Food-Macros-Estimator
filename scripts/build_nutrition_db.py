import os
import json
import requests
import time

# Directory containing the class folders
dataset_dir = "data/Indian Food Images/Indian Food Images"
output_file = "data/nutrition_table.json"

USDA_API_KEY = "DEMO_KEY"
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Common unit types: mostly weight. A few might be count.
# We will heuristically assign them, or fallback to weight.
count_based_keywords = ["naan", "samosa", "idli", "dosa", "chapati", "puri", "bhature", "roll", "momos", "vada", "tikka"]

def get_nutrition_from_usda(query):
    params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": 1,
        "dataType": ["Branded", "Foundation", "SR Legacy"]
    }
    response = requests.get(USDA_SEARCH_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            food = data["foods"][0]
            nutrients = food.get("foodNutrients", [])
            
            # Extract macronutrients (per 100g by default in USDA API for most entries)
            calories = 0
            protein = 0
            carbs = 0
            fat = 0
            
            for nutrient in nutrients:
                name = nutrient.get("nutrientName", "").lower()
                val = nutrient.get("value", 0)
                if "energy" in name and "kcal" in nutrient.get("unitName", "").lower():
                    calories = val
                elif "protein" in name:
                    protein = val
                elif "carbohydrate" in name:
                    carbs = val
                elif "total lipid (fat)" in name or "fat" in name:
                    fat = val
            
            return {
                "calories": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat
            }
    return None

def build_db():
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Found {len(classes)} classes.")

    db = {}
    
    for cls in classes:
        print(f"Fetching nutrition for {cls}...")
        clean_name = cls.replace("_", " ")
        
        # Determine unit type based on keywords
        unit_type = "weight"
        for kw in count_based_keywords:
            if kw in clean_name.lower():
                unit_type = "count"
                break
                
        macros = get_nutrition_from_usda(clean_name)
        
        if not macros:
            print(f"  -> Not found in USDA, using default fallback values.")
            macros = {
                "calories": 250,
                "protein_g": 10,
                "carbs_g": 30,
                "fat_g": 10
            }
            
        db[cls] = {
            "dish_name": clean_name,
            "unit_type": unit_type,
            **macros
        }
        
        # Respect rate limits
        time.sleep(1)
        
    with open(output_file, 'w') as f:
        json.dump(db, f, indent=4)
        
    print(f"Saved nutrition DB to {output_file}")

if __name__ == "__main__":
    build_db()
