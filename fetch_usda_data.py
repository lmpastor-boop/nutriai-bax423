"""
fetch_usda_data.py
------------------
NutriAI — BAX-423 Final Project
Data Pipeline: USDA FoodData Central → SQLite snapshot

Usage:
    python fetch_usda_data.py --api-key YOUR_KEY_HERE
    python fetch_usda_data.py --api-key YOUR_KEY_HERE --max-items 10000

Get a free API key at: https://fdc.nal.usda.gov/api-guide.html
(instant, no credit card)

Output:
    data/foods.db       — SQLite database (primary, for offline use)
    data/foods.csv      — CSV snapshot (for grader convenience)

Pipeline stages (logged):
    1. Fetch Foundation foods  (~2,000 items, best nutrient coverage)
    2. Fetch SR Legacy foods   (~8,000 items, broad variety)
    3. Parse & normalise nutrients
    4. Deduplicate (by lowercased description + food category)
    5. Write SQLite + CSV
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nutriai.pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# USDA nutrient IDs we care about
# Full list: https://fdc.nal.usda.gov/fdc-app.html#/food-details/171705/nutrients
NUTRIENT_MAP = {
    1008: "calories",          # Energy (kcal)
    1003: "protein_g",         # Protein
    1005: "carbs_g",           # Carbohydrate, by difference
    1004: "fat_g",             # Total lipid (fat)
    1079: "fiber_g",           # Fiber, total dietary
    1089: "iron_mg",           # Iron, Fe
    1087: "calcium_mg",        # Calcium, Ca
    1178: "vitamin_b12_mcg",   # Vitamin B-12
    1114: "vitamin_d_mcg",     # Vitamin D (D2 + D3)
    1095: "zinc_mg",           # Zinc, Zn
    1093: "sodium_mg",         # Sodium, Na  (needed for DASH/hypertension)
    1092: "potassium_mg",      # Potassium, K (DASH)
    1090: "magnesium_mg",      # Magnesium, Mg (DASH)
    1404: "omega3_g",          # Fatty acids, total omega-3
}

# Food data types with best nutrient coverage
DATA_TYPES = ["Foundation", "SR Legacy", "Survey (FNDDS)"]

# ── Schema ────────────────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS foods (
    fdc_id          INTEGER PRIMARY KEY,
    description     TEXT    NOT NULL,
    food_category   TEXT,
    data_type       TEXT,
    brand_owner     TEXT,
    -- Macronutrients (per 100g)
    calories        REAL,
    protein_g       REAL,
    carbs_g         REAL,
    fat_g           REAL,
    fiber_g         REAL,
    -- Micronutrients (per 100g)
    iron_mg         REAL,
    calcium_mg      REAL,
    vitamin_b12_mcg REAL,
    vitamin_d_mcg   REAL,
    zinc_mg         REAL,
    sodium_mg       REAL,
    potassium_mg    REAL,
    magnesium_mg    REAL,
    omega3_g        REAL,
    -- Metadata
    ingredients_text TEXT,
    dedup_key       TEXT UNIQUE   -- lowercase(description) + category
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_category ON foods(food_category);
CREATE INDEX IF NOT EXISTS idx_calories  ON foods(calories);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_page(session: requests.Session, api_key: str, data_type: str,
               page: int, page_size: int = 200) -> dict:
    """Fetch one page of food list results."""
    resp = session.get(
        f"{BASE_URL}/foods/list",
        params={
            "api_key":   api_key,
            "dataType":  data_type,
            "pageSize":  page_size,
            "pageNumber": page,
            "nutrients": list(NUTRIENT_MAP.keys()),  # only pull nutrients we need
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def backfill_nutrients(session: requests.Session, api_key: str, fdc_ids: list) -> dict:
    """Fetch full nutrient data for a batch of fdc_ids via POST /foods endpoint.
    Returns dict of {fdc_id: nutrients_dict}."""
    result = {}
    batch_size = 20
    for i in range(0, len(fdc_ids), batch_size):
        batch = fdc_ids[i:i+batch_size]
        try:
            resp = session.post(
                f"{BASE_URL}/foods",
                params={"api_key": api_key},
                json={"fdcIds": batch, "format": "full"},
                timeout=30,
            )
            resp.raise_for_status()
            for food in resp.json():
                fid = food.get("fdcId")
                nutrients = {}
                for n in food.get("foodNutrients", []):
                    nid = n.get("nutrient", {}).get("id")
                    if nid in NUTRIENT_MAP:
                        nutrients[NUTRIENT_MAP[nid]] = n.get("amount")
                result[fid] = nutrients
        except Exception as e:
            log.warning(f"Backfill failed for batch {i}: {e}")
        time.sleep(0.15)
    return result


def parse_nutrients(food_nutrients: list) -> dict:
    """Extract nutrient values from the food nutrients list."""
    result = {col: None for col in NUTRIENT_MAP.values()}
    for n in food_nutrients:
        nid = n.get("nutrientId") or n.get("nutrient", {}).get("id")
        if nid in NUTRIENT_MAP:
            result[NUTRIENT_MAP[nid]] = n.get("value") or n.get("amount")
    return result


def make_dedup_key(description: str, category: str) -> str:
    desc = (description or "").lower().strip()
    cat = (category or "").lower().strip()
    return f"{desc}|{cat}"


def fetch_all_foods(api_key: str, max_items: int) -> list[dict]:
    """Fetch Foundation + SR Legacy foods up to max_items total."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_foods = []
    seen_keys = set()

    for data_type in DATA_TYPES:
        if len(all_foods) >= max_items:
            break

        log.info(f"Fetching {data_type} foods …")
        page = 1

        with tqdm(desc=f"  {data_type}", unit=" items", dynamic_ncols=True) as pbar:
            while len(all_foods) < max_items:
                try:
                    items = fetch_page(session, api_key, data_type, page)
                except requests.HTTPError as e:
                    log.error(f"HTTP {e.response.status_code} on page {page}: {e}")
                    break
                except requests.RequestException as e:
                    log.error(f"Request error on page {page}: {e}")
                    time.sleep(5)
                    continue

                if not items:
                    break  # no more pages

                for food in items:
                    key = make_dedup_key(
                        food.get("description", ""),
                        food.get("foodCategory", "") or food.get("foodCategoryLabel", "")
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    nutrients = parse_nutrients(
                        food.get("foodNutrients", [])
                    )

                    all_foods.append({
                        "fdc_id":          food.get("fdcId"),
                        "description":     food.get("description", ""),
                        "food_category":   food.get("foodCategory") or food.get("foodCategoryLabel", ""),
                        "data_type":       data_type,
                        "brand_owner":     food.get("brandOwner", ""),
                        "ingredients_text": food.get("ingredients", ""),
                        "dedup_key":       key,
                        **nutrients,
                    })
                    pbar.update(1)

                    if len(all_foods) >= max_items:
                        break

                if len(items) < 200:
                    break  # last page
                page += 1
                time.sleep(0.1)  # be polite to the API

    # Backfill nutrients for foods missing calorie data (Foundation + FNDDS)
    missing = [f for f in all_foods if not f.get("calories")]
    if missing:
        log.info(f"Backfilling nutrients for {len(missing):,} foods missing calorie data...")
        fdc_ids = [f["fdc_id"] for f in missing if f.get("fdc_id")]
        nutrient_map = backfill_nutrients(session, api_key, fdc_ids)
        for food in all_foods:
            if not food.get("calories") and food.get("fdc_id") in nutrient_map:
                food.update(nutrient_map[food["fdc_id"]])
        filled = sum(1 for f in all_foods if f.get("calories"))
        log.info(f"After backfill: {filled:,}/{len(all_foods):,} foods have calorie data")

    log.info(f"Fetched {len(all_foods):,} unique foods total.")
    return all_foods


# ── Write outputs ─────────────────────────────────────────────────────────────
def write_sqlite(foods: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TABLE + CREATE_INDEX)

    columns = [
        "fdc_id", "description", "food_category", "data_type", "brand_owner",
        "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
        "iron_mg", "calcium_mg", "vitamin_b12_mcg", "vitamin_d_mcg", "zinc_mg",
        "sodium_mg", "potassium_mg", "magnesium_mg", "omega3_g",
        "ingredients_text", "dedup_key",
    ]
    placeholders = ", ".join("?" * len(columns))
    sql = f"INSERT OR IGNORE INTO foods ({', '.join(columns)}) VALUES ({placeholders})"

    rows = [[f.get(c) for c in columns] for f in foods]
    conn.executemany(sql, rows)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
    conn.close()
    log.info(f"SQLite: {count:,} rows written → {db_path}")


def write_csv(foods: list[dict], csv_path: Path) -> None:
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not foods:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=foods[0].keys())
        writer.writeheader()
        writer.writerows(foods)

    log.info(f"CSV: {len(foods):,} rows written → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NutriAI — USDA FoodData Central pipeline")
    parser.add_argument("--api-key",   required=True,  help="USDA FDC API key")
    parser.add_argument("--max-items", type=int, default=8000,
                        help="Max food items to fetch (default 8000; rubric needs ≥5000)")
    parser.add_argument("--db",   default="data/foods.db",  help="Output SQLite path")
    parser.add_argument("--csv",  default="data/foods.csv", help="Output CSV path")
    args = parser.parse_args()

    log.info("=== NutriAI Data Pipeline ===")
    log.info(f"Target: {args.max_items:,} items  |  DB: {args.db}  |  CSV: {args.csv}")

    foods = fetch_all_foods(args.api_key, args.max_items)

    if not foods:
        log.error("No foods fetched. Check your API key.")
        return

    write_sqlite(foods, Path(args.db))
    write_csv(foods, Path(args.csv))

    # Quick quality report
    has_calories = sum(1 for f in foods if f.get("calories") is not None)
    has_iron     = sum(1 for f in foods if f.get("iron_mg")  is not None)
    log.info(f"Quality check: {has_calories:,}/{len(foods):,} have calories, "
             f"{has_iron:,}/{len(foods):,} have iron data")
    log.info("Done. ✓")


if __name__ == "__main__":
    main()
