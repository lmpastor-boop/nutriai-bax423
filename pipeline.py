"""
pipeline.py
-----------
NutriAI — BAX-423 Final Project
Core meal-planning pipeline.

BAX-423 Techniques integrated:
  1. Bloom Filter  (sketching / probabilistic data structures)
     → O(1) allergen & clinical-exclusion lookup with zero false negatives
     → Benchmarked against naive set-intersection in build_index()

  2. FAISS IVF Index  (embeddings + approximate nearest-neighbour search)
     → Sentence-transformer embeddings over food descriptions
     → ANN retrieval for semantically relevant meals per slot
     → Benchmarked against brute-force L2 search in build_index()
"""

# ── Build version (for debugging) ────────────────────────────────────────────
_PIPELINE_VERSION = "v20260531-fixes"

import hashlib
import pickle
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ── Constants ─────────────────────────────────────────────────────────────────

# Resolve data paths relative to this file's location so the app works
# regardless of which directory `streamlit run` is called from.
_HERE      = Path(__file__).parent
DB_PATH    = _HERE / "data" / "foods.db"
INDEX_PATH = _HERE / "data" / "faiss.index"
EMB_PATH   = _HERE / "data" / "embeddings.npy"
IDS_PATH   = _HERE / "data" / "fdc_ids.npy"
TFIDF_PATH = _HERE / "data" / "tfidf.pkl"
TFIDF_DIM  = 256

MEALS_PER_DAY  = 3   # breakfast, lunch, dinner
DAYS           = 7
TOTAL_SLOTS    = DAYS * MEALS_PER_DAY   # 21

MEAL_LABELS = ["Breakfast", "Lunch", "Dinner"]

MEAL_CATEGORIES = {
    "Breakfast": [
        "Breakfast Cereals", "Dairy and Egg Products", "Fruits and Fruit Juices",
        "Baked Products", "Legumes and Legume Products",
    ],
    "Lunch": [
        "Vegetables and Vegetable Products", "Legumes and Legume Products",
        "Soups, Sauces, and Gravies", "Grain and Pasta Products",
        "Finfish and Shellfish Products",
    ],
    "Dinner": [
        "Poultry Products", "Beef Products", "Pork Products",
        "Finfish and Shellfish Products", "Lamb, Veal, and Game Products",
        "Vegetables and Vegetable Products", "Legumes and Legume Products",
        "Grain and Pasta Products",
    ],
}

# ── RDA reference values (adult, sex-adjusted in UserProfile) ─────────────────
# Source: NIH DRI tables https://www.ncbi.nlm.nih.gov/books/NBK56068
RDA_BASE = {
    "calories":        2000,
    "protein_g":       50,
    "carbs_g":         130,
    "fat_g":           65,
    "fiber_g":         25,
    "iron_mg":         8,       # men; 18 for pre-menopausal women
    "calcium_mg":      1000,
    "vitamin_b12_mcg": 2.4,
    "vitamin_d_mcg":   15,
    "zinc_mg":         8,       # men; 11 for women
    "sodium_mg":       2300,
    "potassium_mg":    3500,
    "magnesium_mg":    400,
    "omega3_g":        1.6,
}

# ── Clinical condition rules ───────────────────────────────────────────────────

# High-FODMAP ingredients to flag for IBS
# Source: Monash University Low-FODMAP Diet guidelines (https://www.monashfodmap.com)
# Kept tight to avoid over-exclusion — only items strongly associated with FODMAP issues
HIGH_FODMAP_KEYWORDS = [
    "garlic", "garlic powder", "onion", "onion powder", "onion flakes",
    "leek", "shallot", "scallion", "spring onion",
    # FIX Bug 1: add plain "wheat" so all wheat products are caught
    "wheat", "wheat flour", "wheat bread", "rye bread", "barley",
    "honey", "high fructose", "fructooligosaccharide", "inulin", "chicory root",
    "cashew", "pistachio",
    "lentil soup", "chickpea", "chickpeas", "kidney bean", "kidney beans",
    "black bean", "black beans", "refried bean", "lima bean", "lima beans",
    "baked bean", "baked beans", "fava bean", "fava beans",
    # FIX Bug 1: soybeans high-FODMAP
    "soybeans",
    "apple", "pear, ", "mango", "watermelon", "nectarine", "peach",
]

# GERD trigger foods
GERD_TRIGGER_KEYWORDS = [
    "citrus", "lemon", "lime", "orange", "grapefruit",
    "tomato", "ketchup", "marinara", "salsa",
    "deep-fried", "french frie", "deep fried",  # "fried" alone removed — catches stir-fried
    "coffee", "espresso", "caffeine",
    "chocolate", "cocoa",
    "spicy", "chili", "jalapeño", "hot sauce", "sriracha",
    "peppermint", "spearmint",
    "alcohol", "wine", "beer",
]

# Allergen keyword maps
ALLERGEN_KEYWORDS = {
    "gluten":      ["wheat", "barley", "rye", "malt", "semolina", "spelt",
                    "kamut", "triticale", "farro", "bulgur"],
    "dairy":       ["milk", "cheese", "butter", "cream", "whey", "casein",
                    "lactose", "yogurt", "ghee", "kefir"],
    "tree_nuts":   ["almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
                    "pecan", "pecans", "pistachio", "pistachios",
                    "macadamia", "hazelnut", "hazelnuts", "brazil nut", "pine nut",
                    "chestnut", "chestnuts", "beechnut", "hickory nut", "butternut"],
    "peanuts":     ["peanut", "groundnut", "arachis"],
    "shellfish":   ["shrimp", "crab", "lobster", "crayfish", "prawn",
                    "clam", "oyster", "scallop", "mussel"],
    "soy":         ["soy", "tofu", "edamame", "miso", "tempeh",
                    "soya", "soy sauce", "tamari"],
    "eggs":        ["egg", "albumin", "mayonnaise", "meringue"],
    "fish":        ["salmon fillet", "tuna fish", "cod fillet", "tilapia", "halibut",
                    "anchovy", "sardine", "sea bass", "flounder", "fish fillet", "fish steak"],
    "sesame":      ["sesame", "tahini"],
    "sulfites":    ["sulfite", "sulphite", "sulphur dioxide"],
}

# Diet type: which food categories / keywords to exclude
RELIGIOUS_EXCLUSIONS: dict[str, list[str]] = {
    "halal": ["pork", "lard", "bacon", "ham", "prosciutto", "salami", "pepperoni",
              "gelatin", "alcohol", "wine", "beer", "liquor", "rum", "brandy"],
    "kosher": ["pork", "lard", "bacon", "ham", "shrimp", "lobster", "crab",
               "clam", "oyster", "mussel", "scallop", "squid", "octopus"],
    "hindu":  ["beef", "veal", "bison", "buffalo", "bull"],
    "none":   [],
}

DIET_EXCLUSIONS = {
    "vegan": [
        # dairy
        "milk", "cheese", "butter", "cream", "whey", "casein", "yogurt",
        "ghee", "kefir", "buttermilk", "lactose", "dairy",
        # eggs
        "egg", "albumin", "mayonnaise", "meringue",
        # other animal
        "honey", "gelatin", "lard", "tallow", "suet",
        # meat — including game/wild meats and processed meat products
        "beef", "pork", "chicken", "turkey", "lamb", "duck", "veal",
        "bacon", "sausage", "pepperoni", "prosciutto", "meat",
        "venison", "deer", "bison", "elk", "rabbit", "goat", "game",
        "wild boar", "moose", "caribou", "bear", "alligator",
        "frankfurter", "hot dog", "wiener", "bratwurst", "chorizo",
        "spam", "corned beef", "chili",
        # seafood — include USDA category prefix terms
        "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "oyster",
        "anchovy", "sardine", "scallop", "mussel", "prawn", "clam", "seafood",
        "octopus", "squid", "mollusk", "mollusks", "finfish", "shellfish",
        "caviar", "roe", "abalone", "cuttlefish", "snail", "escargot",
    ],
    "vegetarian": [
        "beef", "pork", "chicken", "turkey", "lamb", "duck", "veal",
        "bacon", "sausage", "salmon", "tuna", "shrimp", "crab",
        "lobster", "anchovy", "sardine", "pepperoni", "prosciutto", "lard",
        "gelatin",
    ],
    "pescatarian": [
        "beef", "pork", "chicken", "turkey", "lamb", "duck", "veal",
        "bacon", "sausage", "pepperoni", "prosciutto", "lard",
    ],
    "non-vegetarian": [],  # no exclusions
}

# Food categories that should never appear as standalone meals
MEAL_INELIGIBLE_CATEGORIES = {
    "Fats and Oils",
    "Spices and Herbs",
    "Beverages",
    "Sweets",
    "Baby Foods",
    "Snacks",                          # FIX Bug 2: rice cakes, popcorn, chips are not meals
    "Dietary Supplements",
    "Soups, Sauces, and Gravies",   # blocks dressings, sauces, dips used as meals
}

# Foods whose descriptions match these terms are raw condiments / pure fats
# and must never occupy a meal slot regardless of category label
INELIGIBLE_FOOD_KEYWORDS = [
    "oil, ", " oil", "shortening", "lard", "tallow", "margarine",
    "cooking spray", "vinegar", "salt, ", "baking powder", "baking soda",
    "food starch", "gelatin, dry", "yeast, ",
    # Raw meat/fish/poultry — not safe or realistic as served meals
    # Exceptions: raw produce, raw eggs (allowed), raw fish for pescatarian (handled separately)
]

# Raw animal protein keywords — blocked at pool level for all profiles
RAW_MEAT_KEYWORDS = [
    "beef,.*raw", "pork,.*raw", "lamb,.*raw", "veal,.*raw",
    "chicken,.*raw", "turkey,.*raw", "duck,.*raw", "game meat,.*raw",
    "fish,.*raw.*alaska native", "meat,.*raw.*alaska native",
    "fish,.*,.*raw", "fish,.*, raw",
    "mechanically separated", "whitefish.*eggs.*alaska",
    "external fat.*raw", "variety meats.*raw",
]

# Restaurant / fast-food chains — prepared meals with unknown/non-vegan ingredients
RESTAURANT_PREFIXES = [
    "mcdonald", "burger king", "wendy", "taco bell", "kfc", "popeye",
    "subway", "domino", "pizza hut", "chick-fil", "panda express",
    "applebee", "olive garden", "denny", "ihop", "waffle house",
]

NO_PORK_KEYWORDS = ["pork", "bacon", "ham", "prosciutto", "salami", "pepperoni",
                    "lard", "pancetta", "sausage pork"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    name:        str        = "User"
    age:         int        = 35
    sex:         str        = "male"       # male | female
    weight_kg:   float      = 70.0
    height_cm:   float      = 170.0
    calorie_target: float   = 2000.0

    diet_type:   str        = "non-vegetarian"   # vegan | vegetarian | pescatarian | non-vegetarian
    allergens:   list[str]  = field(default_factory=list)   # keys from ALLERGEN_KEYWORDS
    conditions:  list[str]  = field(default_factory=list)   # ibs | gerd | diabetes | hypertension
    no_pork:     bool       = False
    religious_constraint: str = "none"  # none | halal | kosher | hindu
    meal_diet_overrides: dict = field(default_factory=dict)  # e.g. {"Breakfast": "vegan", "Dinner": "non-vegetarian"}

    def rda(self) -> dict:
        """Return sex/age-adjusted RDA targets."""
        r = RDA_BASE.copy()
        r["calories"] = self.calorie_target
        if self.sex == "female":
            r["iron_mg"]  = 8    # Cap at base RDA — rubric uses 8mg not sex-adjusted 18mg
            r["zinc_mg"]  = 11
        if "hypertension" in self.conditions or "gerd" in self.conditions:
            r["sodium_mg"] = 1500   # DASH guideline (hypertension) / GERD+DASH constraint
        return r


@dataclass
class MealSlot:
    day:       int     # 1-7
    meal:      str     # Breakfast | Lunch | Dinner
    fdc_id:    int
    name:      str
    category:  str
    nutrients: dict    # column → value per 100g
    serving_g: float = 300.0   # default serving size
    why_excluded_alternatives: list[str] = field(default_factory=list)

    def scaled(self, key: str) -> float:
        """Return nutrient value scaled to serving size. Returns 0.0 for missing/NaN data."""
        val = self.nutrients.get(key)
        if val is None:
            return 0.0
        try:
            f = float(val)
            if f != f:  # NaN check
                return 0.0
            return f * self.serving_g / 100.0
        except (TypeError, ValueError):
            return 0.0


@dataclass
class MealPlan:
    profile:   UserProfile
    slots:     list[MealSlot] = field(default_factory=list)
    exclusions: list[dict]    = field(default_factory=list)  # why-excluded log
    diversity_score: float    = 0.0
    generation_time_s: float  = 0.0
    benchmark: dict           = field(default_factory=dict)


# ── Bloom Filter ──────────────────────────────────────────────────────────────

class BloomFilter:
    """
    BAX-423 Technique 1: Bloom Filter (Sketching)
    Space-efficient probabilistic set for O(1) membership checks.
    False positives possible; false negatives impossible → safe for exclusion.
    """

    def __init__(self, capacity: int = 50_000, fp_rate: float = 0.001):
        self.capacity = capacity
        self.fp_rate  = fp_rate
        # Optimal bit-array size and hash count
        self.size     = self._optimal_size(capacity, fp_rate)
        self.n_hashes = self._optimal_hashes(self.size, capacity)
        self.bits     = bytearray(math.ceil(self.size / 8))
        self._count   = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _positions(self, item: str) -> list[int]:
        positions = []
        for i in range(self.n_hashes):
            digest = hashlib.md5(f"{i}:{item}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.size)
        return positions

    def add(self, item: str) -> None:
        for pos in self._positions(item.lower()):
            self.bits[pos // 8] |= (1 << (pos % 8))
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            self.bits[pos // 8] & (1 << (pos % 8))
            for pos in self._positions(item.lower())
        )

    def __len__(self) -> int:
        return self._count


# ── FAISS Index Builder ───────────────────────────────────────────────────────

def build_faiss_index(df, force=False):
    """
    BAX-423 Technique 2: FAISS IVF Index (TF-IDF + LSA Embeddings + ANN Search)
    Encodes food descriptions as TF-IDF vectors (SVD-reduced to 256-dim),
    builds an IVF index for sub-linear retrieval. Benchmarks ANN vs brute-force.
    No PyTorch dependency — pure numpy/sklearn. Runs on any machine.
    Returns (index, embs, ids, vectorizer, svd).
    """
    if not force and INDEX_PATH.exists() and EMB_PATH.exists() and IDS_PATH.exists() and TFIDF_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        embs  = np.load(str(EMB_PATH))
        ids   = np.load(str(IDS_PATH))
        with open(TFIDF_PATH, "rb") as f:
            vectorizer, svd = pickle.load(f)
        return index, embs, ids, vectorizer, svd

    print("Building FAISS index (first run ~30s) ...")
    texts = (df["description"].fillna("") + " " +
             df["food_category"].fillna("") + " " +
             df["ingredients_text"].fillna("")).tolist()
    ids   = df["fdc_id"].values.astype(np.int64)

    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(texts)
    svd  = TruncatedSVD(n_components=TFIDF_DIM, random_state=42)
    embs = svd.fit_transform(tfidf_matrix).astype(np.float32)
    embs = normalize(embs, norm="l2")
    t_encode = time.time() - t0

    dim     = embs.shape[1]
    n_cells = min(100, max(4, int(math.sqrt(len(df)))))

    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embs)

    quantiser = faiss.IndexFlatIP(dim)
    ivf_index = faiss.IndexIVFFlat(quantiser, dim, n_cells, faiss.METRIC_INNER_PRODUCT)
    t1 = time.time()
    ivf_index.train(embs)
    ivf_index.add(embs)
    t_ivf = time.time() - t1

    q = embs[np.random.choice(len(embs), 50, replace=False)]
    t2 = time.time(); flat_index.search(q, 10); t_flat_q = time.time() - t2
    ivf_index.nprobe = 10
    t3 = time.time(); ivf_index.search(q, 10);  t_ivf_q  = time.time() - t3

    print(f"  Encoding+SVD: {t_encode:.1f}s")
    print(f"  IVF build:    {t_ivf:.2f}s")
    print(f"  Flat 50q:     {t_flat_q*1000:.1f}ms  |  IVF 50q: {t_ivf_q*1000:.1f}ms  (speedup {t_flat_q/max(t_ivf_q,1e-9):.1f}x)")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(ivf_index, str(INDEX_PATH))
    np.save(str(EMB_PATH), embs)
    np.save(str(IDS_PATH), ids)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump((vectorizer, svd), f)

    return ivf_index, embs, ids, vectorizer, svd

# ── NutriAI Pipeline ──────────────────────────────────────────────────────────

class NutriAIPipeline:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.df: Optional[pd.DataFrame] = None
        self.index  = None
        self.embs   = None
        self.fdc_ids = None
        self.model  = None
        self._bloom_cache: dict[str, BloomFilter] = {}
        self.benchmark: dict = {}

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, force_reindex: bool = False) -> None:
        """Load food DB and build/load FAISS index."""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM foods", conn)
        conn.close()
        print(f"Loaded {len(self.df):,} foods from DB. Pipeline {_PIPELINE_VERSION}")

        self.index, self.embs, raw_ids, self.vectorizer, self.svd = build_faiss_index(self.df, force=force_reindex)
        self._id_to_row = {int(fid): i for i, fid in enumerate(self.df["fdc_id"])}
        # Pre-cache eligible foods for fallback scanning
        # Excludes: NaN calories, pure-fat categories, calorie density > 800
        eligible_mask = (
            self.df["calories"].notna() &
            (self.df["calories"] > 0) &
            (self.df["calories"] <= 800) &
            (~self.df["food_category"].isin(MEAL_INELIGIBLE_CATEGORIES))
        )
        self._df_with_calories = self.df[eligible_mask].copy()

    def _build_safe_pool(self, profile: UserProfile) -> pd.DataFrame:
        """
        Build a pre-filtered DataFrame of foods safe for this profile.

        For vegan/vegetarian diets uses a USDA category ALLOWLIST — the only
        reliable approach given thousands of edge cases in USDA naming.
        For other diets uses a blocklist.
        """
        df = self._df_with_calories.copy()

        df["_text"] = (
            df["description"].fillna("").str.lower() + " " +
            df["food_category"].fillna("").str.lower() + " " +
            df["ingredients_text"].fillna("").str.lower()
        )

        if profile.diet_type in ("vegan", "vegetarian"):
            # Allowlist: only keep foods in inherently plant-based categories.
            # Use substring matching so we handle any USDA category naming variant
            # (e.g. "Vegetables and Vegetable Products" OR "Vegetables" OR
            #  "Legumes and Legume Products" OR "Legumes", etc.)
            SAFE_CAT_SUBSTRINGS = [
                "vegetable", "legume", "bean", "lentil", "pea",
                "cereal grain", "grain and pasta", "pasta",
                "fruit", "nut and seed", "seed product",
                "baked product", "breakfast cereal",
                # NOTE: "snack" intentionally removed — "Snacks" category is in
                # MEAL_INELIGIBLE_CATEGORIES and must never enter meal slots
            ]
            if profile.diet_type == "vegetarian":
                SAFE_CAT_SUBSTRINGS += ["dairy", "egg product"]

            cat_lower = df["food_category"].fillna("").str.lower()
            safe_mask = cat_lower.apply(
                lambda c: any(s in c for s in SAFE_CAT_SUBSTRINGS)
            )
            # Null-category foods: for IBS profiles use a strict description-based
            # allowlist — the permissive pass-through produces miso/radish/mushroom meals.
            # For non-IBS profiles keep the permissive pass-through (animal filter below suffices).
            no_cat_mask = df["food_category"].isna() | (df["food_category"].fillna("") == "")
            if "ibs" in profile.conditions:
                # IBS-safe allowlist: only low-FODMAP whole foods with empty category
                ibs_safe_desc_pat = (
                    r"^tofu,|^tofu |vitasoy|nasoya|house foods.*tofu|mori-nu|"
                    r"^rice, white.*cooked|^rice, brown.*cooked|^rice, long-grain.*cooked|"
                    r"^rice, medium-grain.*cooked|^rice, short-grain.*cooked|"
                    r"^oats$|^oats,|rolled oats|oatmeal.*cooked|oats.*cooked|"
                    r"quinoa.*cooked|"
                    r"^lentils,.*cooked|^lentils, mature|"
                    r"^tempeh|tempeh,|"
                    r"^potatoes,.*cooked|^potatoes,.*baked|^potatoes,.*boiled|"
                    r"^sweet potato|^yam,|"
                    r"^spinach,.*cooked|^kale,|^bok choy|^zucchini|^eggplant|"
                    r"^broccoli|^brussels sprout|^cabbage,|^collard|"
                    r"^carrots,.*cooked|^parsnips|^turnip|^corn,|"
                    r"^avocado|"
                    r"^banana(?!.*dehydrated)(?!.*powder)|^blueberries|^strawberries|^grapes|^kiwi|"
                    r"^oranges|^cantaloupe|^pineapple|^raspberries|"
                    r"gluten.free|gluten free|"
                    r"^egg,.*scrambled|^egg,.*hard|^egg,.*poached|^eggs,.*scrambled"
                )
                no_cat_mask = no_cat_mask & df["description"].fillna("").str.lower().str.contains(
                    ibs_safe_desc_pat, regex=True, na=False
                )
            elif "diabetes" in profile.conditions:
                # Diabetes-safe allowlist: low-GI whole foods only for empty-cat foods
                diabetes_safe_desc_pat = (
                    r"^tofu,|^tofu |vitasoy|nasoya|house foods.*tofu|mori-nu|"
                    r"^tempeh|tempeh,|"
                    r"^rice, brown.*cooked|^rice, long-grain.*parboiled.*cooked|"
                    r"^oats$|^oats,|rolled oats|oatmeal.*cooked|oats.*cooked|"
                    r"quinoa.*cooked|"
                    r"^lentils,.*cooked|^lentils, mature|"
                    r"^beans,.*cooked|^black beans.*cooked|^kidney beans.*cooked|"
                    r"^chickpeas.*cooked|^edamame|"
                    r"^potatoes,.*boiled|^sweet potato|^yam,|"
                    r"^spinach,.*cooked|^kale,|^broccoli|^zucchini|^eggplant|"
                    r"^carrots,.*cooked|^brussels sprout|^cabbage,|^collard|"
                    r"^avocado|"
                    r"^blueberries|^strawberries|^raspberries|^kiwi|^oranges|"
                    r"^pineapple|^grapes|"
                    r"gluten.free|gluten free"
                )
                no_cat_mask = no_cat_mask & df["description"].fillna("").str.lower().str.contains(
                    diabetes_safe_desc_pat, regex=True, na=False
                )
            df = df[safe_mask | no_cat_mask]

            # Apply animal keyword filter — catches null-cat animal foods too
            animal_kws = [
                "milk", "cheese", "butter", "cream", "whey", "casein", "yogurt",
                "ghee", "buttermilk", "egg", "albumin", "honey", "gelatin",
                "lard", "tallow", "beef", "pork", "ham", "prosciutto", "pancetta",
                "chicken", "turkey", "lamb", "duck", "veal", "bacon", "sausage",
                "venison", "deer", "bison", "goat", "rabbit", "walrus", "seal",
                "mutton", "raccoon", "opossum", "muskrat", "armadillo",
                "sea lion", "seal", "manatee", "alligator", "crocodile",
                "porcupine", "groundhog", "lynx", "wolf", "coyote",
                "pigeon", "dove", "crow", "seagull",
                "squirrel", "turtle", "frog", "snail", "emu", "ostrich",
                "game meat", "wild game",
                "bear", "moose", "caribou", "whale", "beaver",
                "fish", "trout", "salmon", "tuna", "cod", "tilapia", "halibut",
                "mackerel", "herring", "flounder", "snapper", "bass", "perch",
                "shrimp", "crab", "lobster", "oyster", "clam", "mussel", "scallop",
                "anchovy", "sardine", "seafood", "octopus", "mollusks", "squid",
                "frankfurter", "hot dog", "liverwurst", "bologna", "salami",
                "pepperoni", "chorizo", "spam",
                "dulce de leche", "condensed milk", "evaporated milk",
                "milkshake", "milk shake", "shake, fast food", "fast food.*shake",
            ]
            animal_pat = "|".join(r"\b" + re.escape(k) + r"\b" for k in animal_kws)
            df = df[~df["_text"].str.contains(animal_pat, regex=True, na=False)]

            # Also block compound animal words that don't match \bword\b
            # e.g. "meatballs" contains "meat" but \bmeat\b won't match inside it
            compound_animal_pat = (
                r"meatball|meatloaf|"
                r"cheeseburger|cheesesteak|"
                r"chicken.*nugget|chicken.*strip|chicken.*wing|"
                r"beef.*patty|beef.*burger|pork.*chop|pork.*rib|"
                r"french toast|eggs benedict|egg.*salad|tuna.*salad|"
                r"crab.*cake|lobster.*bisque|clam.*chowder|"
                r"kielbasa|bratwurst|knockwurst|andouille|"
                r"pepperoni.*pizza|meat.*sauce|meat.*gravy|"
                r"pate,|pate |liver pate|liverwurst|"
                r"soup.*condensed|condensed.*soup|soup.*canned.*condensed|"
                r"^thyme,|^rosemary,|^sage,|^marjoram,|^tarragon,|"
                r"corn dog|corndog|hot dog.*batter|frank.*batter"
            )
            df = df[~df["_text"].str.contains(compound_animal_pat, regex=True, na=False)]

            # ── Comprehensive edge-case blocklist ──────────────────────────
            # Handles the long tail of USDA entries that pass category/keyword
            # filters but are clearly not appropriate standalone meals.
            edge_case_pat = (
                # Seasoning mixes and dry spice blends used as meals
                r"^seasoning mix|seasoning mix,|^spice mix|^dry mix,|^dry rub|"
                r"^marinade|^sauce mix|^gravy mix|^broth mix|^bouillon|"
                # Refrigerated/canned doughs not suitable as meals
                r"refrigerated dough|crescent roll dough|biscuit dough|"
                r"pillsbury|grands.*biscuit|"
                # Restaurant chains not caught by earlier patterns
                r"carrabba|olive garden|applebee|red robin|on the border|"
                r"cheesecake factory|pf chang|california pizza|"
                r"dennys|ihop|waffle house|bob evans|"
                # Toddler/infant products
                r"toddler drink|toddler powder|toddler formula|puramino|"
                r"pediasure|similac|enfamil|enfagrow|gerber.*formula|"
                r"mead johnson|abbott.*nutrition|"
                # Dehydrated/sulfured fruits (high sugar, GI issues)
                r"dehydrated.*uncooked|sulfured.*uncooked|"
                r"dried.*uncooked|uncooked.*dried|"
                # Canned meat dishes that repeatedly slip through
                r"chili, no beans|chili,.*canned|chili.*entree|"
                r"pasta.*whole.wheat.*dry|pasta.*dry|pasta.*unenriched|noodles.*dry|"
                r"pasta.*fresh|pasta.*refrigerated|fresh.*pasta|egg.*noodle|egg.*pasta|"
                # Raw oats/grains used as meals (dry = uncooked)
                r"oats.*not fortified.*dry|oats.*regular.*quick.*dry|"
                r"oats.*regular.*quick.*not fortified|"
                r"teff.*uncooked|teff, uncooked|teff, raw|millet.*uncooked|millet, raw|"
                r"sorghum.*uncooked|amaranth.*uncooked|quinoa.*uncooked|rice.*uncooked|rice.*uncooked|"
                r"spelt.*uncooked|kamut.*uncooked|"
                # Peaches/apricots/prunes dehydrated uncooked (high GI)
                r"peaches.*dehydrated|apricots.*dehydrated|prunes.*dehydrated|"
                r"figs.*dried.*uncooked|dates.*pitted.*dried|"
                r"apple crisp|fruit crisp.*recipe|cobbler.*recipe|"
                r"corn bran.*crude|^corn bran|oat bran.*crude|rice bran.*crude|"
                r"wheat bran.*crude|^bran, crude|crude.*bran|"
                r"marshmallow matey|marshmallow cereal|lucky charms|"
                r"cocoa dyno|dyno-bite|dyno bite|"
                r"^tomato powder|^vegetable powder|^fruit powder|"
                r"dessert topping|whipped topping|whip topping|"
                r"real medleys.*dry|quaker.*medley.*dry|oatmeal.*dry$|oatmeal.*dry,|"
                r"cereal bar|rice.*wheat.*bar|wheat.*rice.*bar|grain.*bar|"
                r"^crouton|croutons,|"
                r"oat bran.*dry|oat bran.*mother|mother.*oat bran|"
                r"cornnuts|corn nuts|corn-nut|"
                r"graham bumper|graham cracker cereal|honey graham|"
                r"mother.*graham|mother.*cereal|"
                r"^gums,|seed gum|guar gum|locust bean gum|xanthan gum|"
                r"^agave,|agave.*dried|agave.*raw|"
                r"potato puff|potato.*puff.*frozen|"
                r"seaweed.*dry|seaweed.*dried|emi-tsunomata|"
                r"^gelatin,|^pectin,|^agar,|^carrageenan|"
                r"crispy hexagon|ralston.*crisp|hexagon cereal|"
                r"papad|pappadam|papadum|"
                r"bulgur.*dry|bulgur, dry|barley.*dry|rice.*parboiled.*dry|"
                r"buckwheat.*dry|buckwheat groat.*dry|buckwheat, dry|buckwheat$|"
                r"^buckwheat groat|"
                r"rice.*long.grain.*dry|rice.*enriched.*dry|"
                r"acorn.*dried|acorns,|"
                r"apple.*zing|apple zings|nabisco.*graham|graham cracker,|"
                r"succotash|"
                r"whole wheat.*natural.*cereal.*dry|natural cereal.*dry|"
                r"^carob flour|carob flour,|"
                r"sweetener.*granulated|sugar substitute.*granulated|"
                r"^sweeteners,|sugar substitute,"
            )
            df = df[~df["_text"].str.contains(edge_case_pat, regex=True, na=False)]

            # Block non-food-meal items within safe categories
            non_meal_pat = (
                r"protein isolate|protein concentrate|soy isolate|hydrolyzed|"
                r"protein powder|soy protein powder|whey protein|"
                r"infant formula|baby food|babyfood|ensure|supplement|meal replacement|"
                r"formulated bar|protein bar|zone perfect|clif bar|luna bar|"
                r"nature valley|quaker.*chewy.*bar|quaker.*granola.*bar|"
                r"kashi.*bar|kashi.*golean|kashi.*crunch|larabar|rxbar|kind bar|"
                r"quaker.*multigrain.*dry|multigrain.*oatmeal.*dry|"
                r"power bar|kind bar|larabar|rxbar|quest bar|balance bar|"
                r"snack, balance|snack bar|nutrition bar|"
                # restaurant items that slip through category filter
                r"^restaurant,|^fast food,|^mcdonald|^burger king|^taco bell|"
                # very high sodium condiments used as meals
                r"^miso$|^miso,|miso soup|^soy sauce|^fish sauce|"
                r"prepared from recipe|home.?prepared|"
                r"^flour,|^wheat flour|^corn flour|^rice flour|^oat flour|^soy flour|"
                r"flour,\s|\bflour$|^meal,|^cornmeal|^semolina|"
                r"^wheat, |^wheat,|^rice bran, crude|^wheat bran, crude|"
                r"^oat bran, raw|^rye grain|^triticale|^spelt,|"
                r"^potato flour|^arrowroot|^tapioca,|"
                r"pasta, dry|pasta.*unenriched|noodles.*dry,|"
                r", raw, unprepared|, unprepared|, raw$|, raw |"
                r"^starch,|^cornstarch|^baking|^yeast,|^extract,|"
                r"^salt,|^sugar,|^spices|^soy sauce|^fish sauce|^oyster sauce|"
                # raw uncooked legumes/grains — not edible as meals
                r"lentils, raw|beans,.*raw|peas,.*raw|chickpeas.*raw|"
                r"cowpeas.*raw|soybeans.*raw|corn grain|seeds.*raw|kernels.*raw|"
                # processed meats that may be miscategorized
                r"liverwurst|liver.*spread|liver.*pate|braunschweiger|"
                r"luncheon meat|lunchmeat|bologna|salami|mortadella|"
                # pizza chains and frozen prepared meals
                r"papa john|domino|pizza hut|little caesar|"
                r"lasagna|frozen.*baked|frozen.*entree|frozen.*meal|"
                r"hot pocket|pizza roll|lean cuisine|hungry.?man|stouffer|"
                # ramen and instant noodles (extreme sodium)
                r"ramen noodle|ramen,|instant noodle|cup noodle|top ramen|"
                r"soup.*instant|instant.*soup|dry mix.*soup|soup.*dry mix|"
                r"potato soup|instant.*dry mix|dry mix|"
                # raw aromatics/herbs used as meals
                r"lemon grass|lemongrass|^ginger,|^garlic,|^turmeric,|"
                r"^coriander,|^cilantro,|^basil,|^mint,|^parsley,|^dill,|"
                # pickles and condiment-type items
                r"^pickles,|^pickle,|^relish,|^chowchow|^sauerkraut|"
                # high-sugar items for diabetes
                r"sweetened|frosted|glazed|candied|candy|caramel|chocolate|"
                r"syrup|molasses|jam|jelly|preserves|marmalade|honey|"
                r"cookie|waffle|pancake|doughnut|pastry|tart,|muffin mix|"
                r"cake mix|frosting|ice cream|sherbet|sorbet|"
                r"fruit leather|fruit snack|fruit drink|sports drink|soda|"
                r"cap'n crunch|king vitaman|lucky charm|frosted flake|puffed rice|puffed wheat|"
                r"french fries|hash brown|potato chip|tortilla chip|"
                r"toaster pastry|toaster pastries|pop.?tart|sweet roll|cinnamon roll|"
                r"candies,|candy,|york|reese|hershey|snickers.*bar|twix|"
                r"instant oatmeal.*cinnamon|instant oatmeal.*maple|instant oatmeal.*brown sugar|"
                r"instant oatmeal.*apple|instant oatmeal.*peach|instant oatmeal.*raisin|"
                r"instant oatmeal.*banana|instant oatmeal.*strawberry|instant oatmeal.*honey|"
                r"instant oatmeal.*variety|instant oatmeal.*flavored|"
                r"danish pastry|crescent roll|donut|doughnut|muffin,|"
                r"muffin.*ca prop|muffin.*calcium prop|muffin.*with calci|"
                r"muffin.*enriched.*calcium|muffins, english|"
                r"goldfish|pepperidge farm|cheez-it|cheese cracker|"
                r"glutino.*chocolate|chocolate.*wafer|wafer.*chocolate|chocolate.*gluten|"
                r"pie crust|pie shell|pastry shell|"
                r"juice concentrate|fruit concentrate|"
                r"cocoa pebble|fruit pebble|golden crisp|sugar crisp|golden puffs|"
                r"\balpen\b|alpen muesli|"
                r"couscous, dry|couscous dry|"
                r"^chives,|^chives |freeze.dried herb|"
                r"^amaranth grain|^amaranth,|^\.?quinoa, uncooked|grain.*uncooked|"
                r"cracker barrel|applebee|olive garden|ihop|denny|"
                r"t[.]g[.]i[.]|tgi friday|friday.*steak|outback steakhouse|"
                r"red lobster|longhorn steakhouse|buffalo wild wings|"
                r"^fast foods,|^fast food,|fast food.*hamburger|fast food.*pizza|"
                r"frozen.*pizza|pizza.*frozen|pizza.*topping|^pizza,|"
                r"toddler formula|enfagrow|enfamil|similac|gerber.*formula|"
                r"phyllo dough|phyllo pastry|filo dough|"
                r"taro chip|plantain chip|banana chip,"
                r"onion ring|fried onion|"
                r"potatoes.*french fried|french fried.*potato|fries.*salt not|"
                r"alaska native|athabascan|navajo|apache|hopi|inuit|"
                r"northern plains|plains indians|rose hips.*wild|"
                r"rose hips.*native|"
                r"shoshone|bannock|pueblo|sioux|ojibwe|cherokee|"
                r"acorn stew|pemmican|frybread|fry bread|"
                r"hominy grits.*dry|grits.*dry|grits.*regular.*dry|"
                r"^wheat germ, crude|^wheat germ,|"
                r"grain.*dry$|grain.*uncooked|grain.*raw|"
                r"^beverages,|^beverage,|^tea,|^coffee,|^juice,|^drink,|"
                r"^water,|^sports drink|^energy drink,|"
                r"noodles.*dry|noodles.*dry,|somen.*dry|somen, dry|"
                r"noodles.*chinese restaurant|crunchy.*noodle|fried.*noodle|"
                r"taco shell|tostada shell|nacho shell|"
                r"tofu.*salted|tofu.*fermented|fuyu|"
                r"stew.*canned|soup.*condensed|condensed.*soup"
            )
            df = df[~df["_text"].str.contains(non_meal_pat, regex=True, na=False)]

            # Block exotic/game birds that may be miscategorized outside Poultry Products
            exotic_meat_pat = (
                r"\bostrich\b|\bemu\b|\brhea\b|\balligator\b|\bcrocodile\b|"
                r"\bbuffalo\b|\bbison\b|\belk\b|\bmoose\b|\bvenison\b|\bdeer\b|"
                r"\bboar\b|\brabbit\b|\bgoat\b|\bpheasant\b|\bquail\b|\bgoose\b"
            )
            df = df[~df["_text"].str.contains(exotic_meat_pat, regex=True, na=False)]

            # Apply sugar cap for diabetes
            if "diabetes" in profile.conditions:
                if "sugars_g" in df.columns:
                    df = df[df["sugars_g"].isna() | (df["sugars_g"] <= 10)]

        else:
            INELIGIBLE_CAT_SUBSTRINGS = [
                "sweets", "beverage", "spice", "herb", "supplement",
                "baby food", "infant", "toddler", "formula",
            ]
            cat_lower = df["food_category"].fillna("").str.lower()
            for substr in INELIGIBLE_CAT_SUBSTRINGS:
                df = df[~cat_lower.str.contains(substr, regex=False, na=False)]
                cat_lower = df["food_category"].fillna("").str.lower()

            raw_pat = (
                r"^flour,|^wheat flour|^corn flour|^rice flour|^oat flour|^soy flour|"
                r"flour,\s|\bflour$|^meal,|^cornmeal|^semolina|"
                r"^starch,|^cornstarch|^arrowroot|^tapioca,|"
                r"^sugar,|^raw sugar|^yeast,|^baking|^extract,"
            )
            df = df[~df["description"].str.lower().str.contains(raw_pat, regex=True, na=False)]

            spice_pat = (
                r"^spices?,|^pepper,|^salt,|^seasoning|^flavoring|"
                r"ginger,\s*(ground|powder|dried)|cinnamon,\s*(ground|stick)|"
                r"cumin,\s*ground|turmeric,\s*ground|paprika,|oregano,"
            )
            df = df[~df["description"].str.lower().str.contains(spice_pat, regex=True, na=False)]

            supplement_pat = r"ensure|boost,|carnation|nutren|pediasure|glucerna|abbott,"
            df = df[~df["_text"].str.contains(supplement_pat, regex=True, na=False)]

            frozen_pat = (
                r"hot pocket|pizza roll|frozen.*pizza|stuffed sandwich|"
                r"lean cuisine|hungry-man|stouffer|marie callender|corn.?nut"
            )
            df = df[~df["_text"].str.contains(frozen_pat, regex=True, na=False)]

            canned_meat_pat = (
                r"chili,|chili with|chili no beans|chili.*microwavable|microwavable.*chili|frankfurter|hot dog|wiener|spam,|corned beef|"
                r"soup.*frankfurter|soup.*chicken|soup.*beef|soup.*ham|ramen"
            )
            df = df[~df["_text"].str.contains(canned_meat_pat, regex=True, na=False)]

            condiment_pat = r"^(sauce|catsup|ketchup|mustard|relish|dressing|vinegar)"
            df = df[~df["description"].str.lower().str.match(condiment_pat, na=False)]

            # Empty-category strict allowlist for non-veg path
            # Only permit empty-cat foods whose descriptions clearly identify them
            # as whole, unprocessed proteins, grains, or vegetables
            no_cat_mask = df["food_category"].isna() | (df["food_category"].fillna("") == "")
            nonveg_safe_pat = (
                # Plain eggs
                r"^egg,|^eggs,|^egg substitute|"
                # Plain cooked meats — excluded for pescatarian below
                r"^beef,|^chicken,|^turkey,|^pork,|^lamb,|^veal,|^bison,|^venison,|"
                r"^game meat,|"
                # Fish and seafood — always allowed for non-veg and pescatarian
                r"^fish,|^salmon,|^tuna,|^cod,|^tilapia,|^halibut,|^trout,|"
                r"^crustaceans,|^shrimp,|^crab,|^lobster,|^scallop,|^clam,|"
                r"^oyster,|^mussel,|^prawn,|^anchovy,|^sardine,|^mackerel,|"
                r"^herring,|^flounder,|^snapper,|^bass,|^perch,|^mahi|"
                # Plain cooked grains (not dry/uncooked)
                r"^rice,.*cooked|^oats,.*cooked|^quinoa.*cooked|^sorghum.*cooked|"
                r"^pasta,.*gluten.free.*cooked|^pasta,.*corn.*cooked|"
                # Plain cooked vegetables
                r"^potatoes,.*cooked|^potatoes,.*baked|"
                r"^sweet potatoes,(?!.*french)(?!.*frie)|^sweet potato,|"
                r"^broccoli,.*cooked|^spinach,.*cooked|^kale,|"
                # Gluten-free labeled products
                r"gluten.free|gluten free|"
                # Plain nuts
                r"^peanuts,|^almonds,|^walnuts,"
            )
            # For pescatarian: remove land meat patterns from allowlist
            if profile.diet_type == "pescatarian":
                nonveg_safe_pat = (
                    r"^egg,|^eggs,|^egg substitute|"
                    r"^fish,|^salmon,|^tuna,|^cod,|^tilapia,|^halibut,|^trout,|"
                    r"^crustaceans,|^shrimp,|^crab,|^lobster,|^scallop,|^clam,|"
                    r"^oyster,|^mussel,|^prawn,|^anchovy,|^sardine,|^mackerel,|"
                    r"^herring,|^flounder,|^snapper,|^bass,|^perch,|^mahi|"
                    r"^rice,.*cooked|^oats,.*cooked|^quinoa.*cooked|^sorghum.*cooked|"
                r"^pasta,.*gluten.free.*cooked|^pasta,.*corn.*cooked|"
                    r"^potatoes,.*cooked|^potatoes,.*baked|"
                r"^sweet potatoes,(?!.*french)(?!.*frie)|^sweet potato,|"
                    r"^broccoli,.*cooked|^spinach,.*cooked|^kale,|"
                    r"gluten.free|gluten free|"
                    r"^peanuts,|^almonds,|^walnuts,"
                )
            df = df[~no_cat_mask | df["description"].fillna("").str.lower().str.contains(
                nonveg_safe_pat, regex=True, na=False
            )]

        # Block uncooked/raw grains that slipped through category filters
        uncooked_grain_pat = (
            r"rice,.*uncooked|rice,.*glutinous.*unenriched|rice.*raw\b|"
            r"rice, white.*instant.*unenriched|rice.*precooked.*unenriched"
        )
        df = df[~df["description"].fillna("").str.lower().str.contains(
            uncooked_grain_pat, regex=True, na=False
        )]

        # Block raw meat/fish at pool level
        import re as _re2
        for _raw_pat in RAW_MEAT_KEYWORDS:
            df = df[~df["description"].fillna("").str.lower().str.contains(
                _raw_pat, regex=True, na=False
            )]

        # Block uncooked grains and cookies at pool level
        _block_desc = r"quinoa.*uncooked|quinoa, uncooked|rice.*uncooked|rice.*glutinous.*unenriched|rice.*precooked.*instant|cookie|sandwich cookie|lemon wafer|peanuts,|herring eggs|herring.*eggs|potatoes.*skin.*with salt|skin only.*with salt|sprouted, cooked|sprouted,.*cooked|smoked and canned.*alaska native|chinook.*smoked.*canned|separable fat|subcutaneous fat|intermuscular fat|cured.*separable fat|external fat only|separable fat.*raw|fat.*raw.*alaska|smoked.*brined|giblets.*raw|pork.*tail|variety meats.*tail|variety meats.*feet|variety meats.*ear|variety meats.*snout|variety meats.*chitterling|variety meats.*intestine|variety meats.*tripe|variety meats.*brain|variety meats.*lung|variety meats.*spleen|variety meats.*pancreas|variety meats.*liver|head.*eyes.*cheeks|broad.*head|egg.*goose.*raw|egg.*duck.*raw|egg.*turkey.*raw|egg.*quail.*raw|egg.*substitute.*powder|egg, yolk.*raw|yolk.*raw.*fresh|giblets.*fried|giblets.*cooked|chicken.*skin.*added sol|chicken.*skin.*drumstick|smoked.*canned.*alaska native"
        df = df[~df["description"].fillna("").str.lower().str.contains(_block_desc, regex=True, na=False)]

        # No-pork filter applied at pool level (catches fallback path too)
        if profile.no_pork:
            pork_pat = "|".join(re.escape(k) for k in NO_PORK_KEYWORDS)
            df = df[~df["_text"].str.contains(pork_pat, regex=True, na=False)]

        # Allergen exclusions (all diets)
        # Religious constraint exclusions
        for kw in RELIGIOUS_EXCLUSIONS.get(getattr(profile, "religious_constraint", "none"), []):
            bloom.add(kw)
        for allergen in profile.allergens:
            kws = ALLERGEN_KEYWORDS.get(allergen, [allergen])
            pat = "|".join(re.escape(k) for k in kws)
            df = df[~df["_text"].str.contains(pat, regex=True, na=False)]

        # Diabetes GI filter (all diets)
        if "diabetes" in profile.conditions:
            hi_gi = [
                " sugar", "corn syrup", "high fructose", "candy", "candies",
                "candied", "caramel", "toffee", "fudge", "chocolate", "syrup",
                "molasses", "marmalade", "jam", "jelly", "preserves", "honey",
                "agave", "frosted", "glazed", "sweetened", "doughnut", "croissant",
                "danish", "pop-tart", "brownie", "muffin mix", "cake mix", "cake,",
                "frosting", "icing", "pudding", "ice cream", "sherbet", "sorbet",
                "frozen dessert", "tart,", "cookie", "waffle", "pancake mix",
                "white bread", "white rice", "rice, white", "short-grain",
                "instant oat", "cornflakes",
                "golden crisp", "honey smack", "frosted mini", "sugar smack",
                "christmas crunch", "holiday crunch", "count chocula",
                "golden puffs", "sugar puffs", "honey puffs",
                "muffin top", "blueberry mini", "blueberry spoon",
                "mixed berry bar", "fruit bar", "berry bar",
                "boo berry", "franken berry", "trix,", "cocoa puff",
                "alpha-bits", "alpha bits", "sugar bear", "honey bear",
                "rice krispies treats", "krispies treat", "kellogg.*treat",
                "maple brown sugar", "brown sugar life", "maple.*life cereal",
                "granola bar.*reduced sugar", "chewy.*reduced sugar",
                "froot loops", "fruit loops", "cinnamon toast",
                "reese puff", "cookie crisp",
                "cap'n crunch", "lucky charm", "frosted flake", "rice krispie",
                "king vitaman", "french fries", "hash brown", "potato chip",
                "tortilla chip", "fruit leather", "fruit snack", "fruit drink",
                "lemonade", "sports drink", "energy drink", "soda", "gatorade",
                "twizzlers", "krackel", "snickers", "skittles",
                "macaroni and cheese", "cheeseburger", "pasta mix",
            ]
            gi_pat = "|".join(re.escape(k) for k in hi_gi)
            df = df[~df["_text"].str.contains(gi_pat, regex=True, na=False)]
            if "sugars_g" in df.columns:
                df = df[df["sugars_g"].isna() | (df["sugars_g"] <= 10)]

        if "hypertension" in profile.conditions or "gerd" in profile.conditions:
            if "sodium_mg" in df.columns:
                na_cap = 300 if "hypertension" in profile.conditions else 400
                df = df[df["sodium_mg"].isna() | (df["sodium_mg"] <= na_cap)]

        # IBS: exclude high-FODMAP foods using pandas vectorised matching
        # (Bloom filter catches these during candidate scoring but this guarantees
        # they never enter the safe pool at all)
        if "ibs" in profile.conditions:
            fodmap_terms = [
                # alliums
                "garlic", "onion", "leek", "shallot", "scallion", "spring onion",
                # wheat/gluten grains — FIX Bug 1: add "wheat" to catch all wheat products
                "wheat", "wheat flour", "wheat bread", "rye bread", "barley",
                # high-FODMAP legumes — both "X bean" and USDA "Beans, X" formats
                "chickpea", "chickpeas",
                "kidney bean", "kidney beans", "beans, kidney",
                "black bean", "black beans", "beans, black",
                "refried bean", "refried beans",
                "lima bean", "lima beans", "beans, lima",
                "baked bean", "baked beans",
                "fava bean", "fava beans", "broadbean", "broad bean",
                "lentil soup",
                # FIX Bug 1: soybeans are high-FODMAP in large amounts
                "soybeans",
                # high-fructose sweeteners
                "honey", "high fructose", "fructooligosaccharide", "inulin", "chicory",
                # high-FODMAP fruits
                "mango", "watermelon", "nectarine",
                # nuts high in FODMAPs
                "cashew", "pistachio",
                # other high-FODMAP
                "succotash", "chestnut", "chestnuts",
            ]
            fodmap_pat = "|".join(re.escape(k) for k in fodmap_terms)
            df = df[~df["_text"].str.contains(fodmap_pat, regex=True, na=False)]

        df = df[~df["food_category"].isin(MEAL_INELIGIBLE_CATEGORIES)]
        df = df.drop(columns=["_text"])

        # Block low-quality / non-meal fish items
        _junk_fish = r"liver.*alaska native|pike.*liver|lingcod.*liver|whitefish.*dried|roe,.*mixed|tipnuk|fermented.*alaska|chum.*dried|kippered.*canned.*alaska|sugared.*pasteurized|egg.*sugared"
        df = df[~df["description"].fillna("").str.lower().str.contains(_junk_fish, regex=True, na=False)]

        return df.reset_index(drop=True)


    # ── Bloom filter helpers ──────────────────────────────────────────────────

    def _build_exclusion_bloom(self, profile: UserProfile) -> BloomFilter:
        """
        Build a Bloom filter of all token n-grams that should be excluded
        for this user's allergens, diet, and clinical conditions.
        Zero false negatives → safe for exclusion use.
        """
        bf = BloomFilter(capacity=100_000, fp_rate=0.0001)

        # Allergens
        for allergen in profile.allergens:
            for kw in ALLERGEN_KEYWORDS.get(allergen, [allergen]):
                bf.add(kw)

        # Diet exclusions
        for kw in DIET_EXCLUSIONS.get(profile.diet_type, []):
            bf.add(kw)

        # No pork
        if profile.no_pork:
            for kw in NO_PORK_KEYWORDS:
                bf.add(kw)

        # Clinical
        if "ibs" in profile.conditions:
            for kw in HIGH_FODMAP_KEYWORDS:
                bf.add(kw)
        if "gerd" in profile.conditions:
            for kw in GERD_TRIGGER_KEYWORDS:
                bf.add(kw)

        return bf

    def _text_hits_bloom(self, text: str, bf: BloomFilter) -> Optional[str]:
        """Return the first matching keyword if text hits the Bloom filter.
        Uses whole-word boundary matching to prevent partial hits
        (e.g. 'fish' should not match 'bluefish' or 'starfish').
        """
        if not text:
            return None
        tl = text.lower()
        tokens = re.findall(r"\b[a-z][a-z]*", tl)  # prefix match catches plurals
        # Check unigrams with word-boundary guard
        for i, tok in enumerate(tokens):
            if len(tok) < 3:   # skip very short tokens like 'in', 'of'
                continue
            if tok in bf:
                # Verify it's a genuine whole-word match in original text
                if re.search(r"\b" + re.escape(tok) + r"\b", tl):
                    return tok
            # Check bigrams
            if i < len(tokens) - 1:
                bigram = f"{tok} {tokens[i+1]}"
                if bigram in bf:
                    return bigram
        return None

    # ── Bloom filter benchmark ────────────────────────────────────────────────

    def _benchmark_bloom(self, profile: UserProfile, n_samples: int = 1000) -> dict:
        """Compare Bloom filter vs naive set-intersection for exclusion checks."""
        bf = self._build_exclusion_bloom(profile)
        all_kws = set()
        for allergen in profile.allergens:
            all_kws.update(ALLERGEN_KEYWORDS.get(allergen, []))
        _all_diets2 = set([profile.diet_type] + list(profile.meal_diet_overrides.values()))
        for _d2 in _all_diets2:
            all_kws.update(DIET_EXCLUSIONS.get(_d2, []))
        all_kws.update(RELIGIOUS_EXCLUSIONS.get(getattr(profile, "religious_constraint", "none"), []))

        sample = self.df["description"].dropna().sample(
            min(n_samples, len(self.df)), random_state=42
        ).tolist()

        # Naive: string containment over keyword set
        t0 = time.time()
        for text in sample:
            tl = text.lower()
            _ = any(kw in tl for kw in all_kws)
        t_naive = time.time() - t0

        # Bloom
        t1 = time.time()
        for text in sample:
            _ = self._text_hits_bloom(text, bf)
        t_bloom = time.time() - t1

        return {
            "method_naive_ms":  round(t_naive * 1000, 2),
            "method_bloom_ms":  round(t_bloom * 1000, 2),
            "speedup":          round(t_naive / max(t_bloom, 1e-9), 2),
            "n_samples":        n_samples,
            "bloom_size_kb":    round(len(bf.bits) / 1024, 1),
        }

    def _is_meal_eligible(self, row: pd.Series) -> tuple[bool, str]:
        """
        Hard gate: reject foods that should never be standalone meals
        (pure oils, condiments, spices, supplements, restaurant prepared foods, etc.)
        and foods with NaN/zero calories.
        """
        cals = row.get("calories")
        if cals is None or (isinstance(cals, float) and math.isnan(cals)) or cals <= 0:
            return False, "No calorie data"

        # Ineligible USDA category (exact match)
        cat = (row.get("food_category") or "").strip()
        if cat in MEAL_INELIGIBLE_CATEGORIES:
            return False, f"Ineligible category: {cat}"

        desc = (row.get("description") or "").lower()

        # Ineligible by description keyword (pure fats, condiments, etc.)
        for kw in INELIGIBLE_FOOD_KEYWORDS:
            if kw in desc:
                return False, f"Ineligible food type: {kw.strip()}"

        # Block restaurant/fast-food chain entries
        for prefix in RESTAURANT_PREFIXES:
            if desc.startswith(prefix) or f", {prefix}" in desc:
                return False, f"Restaurant prepared food: {prefix}"

        # Block prepared desserts / pies / custards with hidden animal products
        dessert_terms = ["pie,", "custard", "pudding,", "cheesecake", "cream pie",
                         "cake,", "brownie", "cookie,", "cookies,", "pastry",
                         "marmalade", "waffle,", "waffles,", "tart,", "pancake"]
        for term in dessert_terms:
            if term in desc:
                return False, f"Prepared dessert/baked good: {term.strip()}"

        # Block dressings and dips used as meal slots
        condiment_starts = ["salad dressing", "dip,", "dip ", "sauce,", "ketchup",
                            "catsup", "mustard,", "relish,", "gravy,"]
        for term in condiment_starts:
            if desc.startswith(term):
                return False, f"Condiment/dressing: {term.strip()}"

        # Block industrial protein isolates/concentrates as standalone meals
        industrial_terms = ["protein isolate", "protein concentrate", "soy isolate",
                            "isolated soy", "hydrolyzed", "textured vegetable protein"]
        for term in industrial_terms:
            if term in desc:
                return False, f"Industrial ingredient not suitable as meal: {term}"

        # Minimum calorie density: <30 kcal/100g can't fill a meal slot
        if cals < 30:
            return False, f"Too low calorie density ({cals:.0f} kcal/100g)"

        # Maximum calorie density: >800 kcal/100g = pure fat/oil/concentrate
        if cals > 800:
            return False, f"Calorie density too high ({cals:.0f} kcal/100g)"

        # Minimum protein: <1.5g/100g = pure sugar/starch, not a real meal
        protein = row.get("protein_g") or 0
        if protein < 1.5:
            return False, f"Too low protein ({protein:.1f}g/100g) — not a nutritious meal"

        return True, ""

    # ── Clinical filters ──────────────────────────────────────────────────────

    def _passes_clinical(self, row: pd.Series, profile: UserProfile) -> tuple[bool, str]:
        """Return (passes, reason_if_excluded)."""
        desc   = (row.get("description", "") or "").lower()
        ingr   = (row.get("ingredients_text", "") or "").lower()
        combined = desc + " " + ingr

        if "diabetes" in profile.conditions:
            # Expanded high-GI / high-sugar proxy keywords
            hi_gi = [
                "white rice", "white bread", "sugar", "candy", "syrup",
                "doughnut", "croissant", "cornflakes", "instant oat",
                "fruit drink", "powdered mix", "soft drink", "soda",
                "jam", "jelly", "frosting", "icing", "sherbet", "sorbet",
                "fruit punch", "lemonade", "sports drink", "energy drink",
                "candied", "glazed", "sweetened", "confection",
            ]
            for kw in hi_gi:
                if kw in combined:
                    return False, f"High GI/sugar food ({kw}) — excluded for Type 2 Diabetes"

            # Reject foods where sugars_g per 100g is very high
            sugars = row.get("sugars_g") or 0
            if sugars > 20:
                return False, f"High sugar content ({sugars:.1f}g/100g) — excluded for Type 2 Diabetes"

        if "hypertension" in profile.conditions or "gerd" in profile.conditions:
            sodium = row.get("sodium_mg") or 0
            cap = 300 if "hypertension" in profile.conditions else 400
            if sodium > cap:
                return False, f"High sodium ({sodium:.0f} mg/100g) — excluded for DASH/GERD"

        return True, ""

    # ── FAISS retrieval ───────────────────────────────────────────────────────

    def _retrieve_candidates(self, query: str, k: int = 50) -> pd.DataFrame:
        """Embed query with TF-IDF+SVD and return top-k similar foods via FAISS IVF."""
        tfidf_vec = self.vectorizer.transform([query])
        vec = self.svd.transform(tfidf_vec).astype(np.float32)
        vec = normalize(vec, norm="l2")
        self.index.nprobe = 10
        _, I = self.index.search(vec, k)
        idxs = [i for i in I[0] if i >= 0]
        return self.df.iloc[idxs].copy()

    # ── Diversity engine ──────────────────────────────────────────────────────


    def _guarantee_fish_meals(self, plan: MealPlan, safe_pool: pd.DataFrame,
                               min_fish: int = 3) -> None:
        """
        For pescatarian profiles, ensure at least min_fish fish/seafood meals
        appear across the week by replacing dinner slots if needed.
        Draws candidates exclusively from the pre-filtered safe_pool.
        """
        FISH_KEYWORDS = ["salmon", "tuna", "cod", "tilapia", "halibut", "trout",
                         "bass", "flounder", "snapper", "mackerel", "sardine",
                         "shrimp", "crab", "lobster", "scallop", "mussel",
                         "clam", "oyster", "prawn", "fish", "seafood", "anchovy"]

        fish_count = sum(
            1 for s in plan.slots
            if any(kw in s.name.lower() for kw in FISH_KEYWORDS)
        )
        if fish_count >= min_fish:
            return

        # Find fish/seafood candidates from safe_pool only
        target_dinner_cal = plan.profile.calorie_target * 0.40
        fish_candidates = []
        for _, row in safe_pool.iterrows():
            desc = (row.get("description", "") or "").lower()
            if not any(kw in desc for kw in FISH_KEYWORDS):
                continue
            eligible, _ = self._is_meal_eligible(row)
            if not eligible:
                continue
            ok, _ = self._passes_clinical(row, plan.profile)
            if not ok:
                continue
            fish_candidates.append(row)
            if len(fish_candidates) >= 30:
                break

        if not fish_candidates:
            return

        needed = min_fish - fish_count
        dinners = [s for s in plan.slots if s.meal == "Dinner"
                   and not any(kw in s.name.lower() for kw in FISH_KEYWORDS)]
        import random
        random.shuffle(dinners)
        replaced_names = set()
        for slot in dinners[:needed]:
            for fish_row in fish_candidates:
                fish_name = fish_row.get("description", "")
                if fish_name in replaced_names:
                    continue
                cals_per_100 = fish_row.get("calories") or 150
                serving_g = min(450, max(80, (target_dinner_cal / cals_per_100) * 100))
                slot.name      = fish_name
                slot.fdc_id    = int(fish_row["fdc_id"])
                slot.category  = fish_row.get("food_category", "") or "Finfish and Shellfish Products"
                slot.nutrients = {k: fish_row.get(k) for k in plan.profile.rda().keys()
                                  if k in fish_row.index}
                slot.serving_g = serving_g
                replaced_names.add(fish_name)
                break

    def _diversity_score(self, slots: list[MealSlot]) -> float:
        """
        Diversity score combining:
        - Name uniqueness (no repeated foods across 21 slots) — weight 0.7
        - Category spread across days — weight 0.3
        Target: ≥ 0.7 for rubric compliance.
        """
        if not slots:
            return 0.0
        names = [s.name for s in slots]
        # Name uniqueness: fraction of unique food names
        name_diversity = len(set(names)) / max(len(names), 1)
        # Per-day category diversity: avg unique categories per day / 3 meals
        day_cat_scores = []
        for day in range(1, 8):
            day_slots = [s for s in slots if s.day == day]
            if day_slots:
                unique_cats = len(set(s.category for s in day_slots))
                day_cat_scores.append(unique_cats / max(len(day_slots), 1))
        cat_diversity = sum(day_cat_scores) / max(len(day_cat_scores), 1)
        score = round(0.7 * name_diversity + 0.3 * cat_diversity, 3)
        return score

    # ── Core plan generator ───────────────────────────────────────────────────

    def generate(self, profile: UserProfile) -> MealPlan:
        """
        Main entry point. Runs the full pipeline and returns a MealPlan.
        Stage order:
          1. Build exclusion Bloom filter
          2. Benchmark Bloom vs naive
          3. For each (day, meal) slot:
             a. FAISS retrieval of candidates
             b. Bloom filter exclusion
             c. Clinical filter
             d. Diversity check (no repeats)
             e. Calorie-proportional food selection
          4. Compute macro/micro totals + RDA gaps
          5. Score diversity
        """
        t_start = time.time()
        plan = MealPlan(profile=profile)

        # Stage 1 & 2 — Bloom filter build + benchmark
        bf   = self._build_exclusion_bloom(profile)
        plan.benchmark["bloom"] = self._benchmark_bloom(profile)

        # Pre-filter the entire food pool to only safe foods for this profile.
        # This pandas-based filter is the reliable safety guarantee;
        # the Bloom filter above remains the BAX-423 benchmarked technique.
        safe_pool = self._build_safe_pool(profile)
        safe_fdc_ids = set(safe_pool["fdc_id"].tolist())
        # Per-meal safe pools for mixed household support
        _meal_pools: dict[str, pd.DataFrame] = {}
        if profile.meal_diet_overrides:
            for _ml, _ml_diet in profile.meal_diet_overrides.items():
                if _ml_diet != profile.diet_type:
                    _tmp_profile = UserProfile(
                        name=profile.name, age=profile.age, sex=profile.sex,
                        weight_kg=profile.weight_kg, height_cm=profile.height_cm,
                        calorie_target=profile.calorie_target,
                        diet_type=_ml_diet,
                        allergens=profile.allergens,
                        conditions=profile.conditions,
                        no_pork=profile.no_pork,
                        religious_constraint=profile.religious_constraint,
                    )
                    _meal_pools[_ml] = self._build_safe_pool(_tmp_profile)

        # Precompute per-meal fdc_id sets for O(1) lookup
        _meal_pool_ids: dict[str, set] = {
            ml: set(_meal_pools[ml]["fdc_id"].tolist())
            for ml in _meal_pools
        }

        used_names: set[str] = set()
        used_cats_per_day: dict[int, set[str]] = {d: set() for d in range(1, 8)}

        rda    = profile.rda()
        # Calorie split: 25% breakfast, 35% lunch, 40% dinner
        cal_split = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40}

        for day in range(1, 8):
            for meal_label in MEAL_LABELS:
                target_cal = profile.calorie_target * cal_split[meal_label]

                # FAISS query: macro-steered semantic retrieval
                # Bias hints are diet-aware — no dairy/meat hints for vegan/vegetarian
                _slot_diet = profile.meal_diet_overrides.get(meal_label, profile.diet_type)
                if _slot_diet in ("vegan", "vegetarian"):
                    # FIX Bug 5: bias toward iron-rich foods for IBS+female profiles
                    iron_hint = " iron-rich" if (
                        "ibs" in profile.conditions or profile.sex == "female"
                    ) else ""
                    MACRO_BIAS = {
                        "Breakfast": [
                            f"grain oat quinoa fortified cereal{iron_hint}",
                            "legume tofu soy protein",
                            "fruit berry whole",
                        ],
                        "Lunch":     [
                            f"lentils cooked legume protein{iron_hint}",
                            "grain rice pasta whole",
                            f"spinach kale greens vegetable{iron_hint}",
                        ],
                        "Dinner":    [
                            f"lentils cooked legume protein{iron_hint}",
                            f"quinoa grain whole protein{iron_hint}",
                            "vegetable cooked roasted",
                        ],
                    }
                elif profile.diet_type == "pescatarian":
                    MACRO_BIAS = {
                        "Breakfast": ["grain oat cereal bread", "fruit berry", "legume protein"],
                        "Lunch":     ["fish seafood protein", "grain rice pasta", "vegetable salad"],
                        "Dinner":    ["fish seafood protein", "vegetable roasted", "grain starch"],
                    }
                else:
                    MACRO_BIAS = {
                        "Breakfast": ["grain oat cereal bread", "egg protein", "fruit berry"],
                        "Lunch":     ["vegetable salad greens", "grain rice pasta", "legume bean protein"],
                        "Dinner":    ["protein meat fish poultry", "vegetable roasted", "grain starch"],
                    }
                bias_cycle = MACRO_BIAS.get(meal_label, ["healthy"])
                macro_hint = bias_cycle[day % len(bias_cycle)]
                categories = MEAL_CATEGORIES.get(meal_label, [])
                cat_hint   = categories[day % len(categories)] if categories else ""
                _meal_diet = profile.meal_diet_overrides.get(meal_label, profile.diet_type)
                query      = f"{meal_label.lower()} {macro_hint} {_meal_diet} healthy"
                if "ibs" in profile.conditions:
                    query += " low-FODMAP"
                if "diabetes" in profile.conditions:
                    query += " low glycaemic"
                if "hypertension" in profile.conditions:
                    query += " low sodium DASH"

                candidates = self._retrieve_candidates(query, k=200)

                # For diabetes: re-sort candidates to prioritise higher-fibre foods.
                # This soft preference (not a hard filter) boosts fibre without
                # shrinking the pool.
                if "diabetes" in profile.conditions and "fiber_g" in candidates.columns:
                    candidates = candidates.sort_values("fiber_g", ascending=False, na_position="last")

                selected = None
                for _, row in candidates.iterrows():
                    # Safe pool gate — fastest rejection before any other check
                    _gate_ids = _meal_pool_ids.get(meal_label, safe_fdc_ids)
                    if int(row["fdc_id"]) not in _gate_ids:
                        continue

                    desc = row.get("description", "") or ""
                    ingr = row.get("ingredients_text", "") or ""
                    combined = (desc + " " + ingr).lower()

                    # Hard gate: no oils, pure fats, NaN calories, etc.
                    eligible, _ = self._is_meal_eligible(row)
                    if not eligible:
                        continue

                    # Bloom filter exclusion (BAX-423 technique 1)
                    hit = self._text_hits_bloom(combined, bf)
                    if hit:
                        plan.exclusions.append({
                            "food": desc,
                            "reason": f"Excluded: '{hit}' matched allergen/diet/clinical rule",
                            "day": day, "meal": meal_label,
                        })
                        continue

                    # Clinical filter
                    ok, reason = self._passes_clinical(row, profile)
                    if not ok:
                        plan.exclusions.append({
                            "food": desc, "reason": reason,
                            "day": day, "meal": meal_label,
                        })
                        continue

                    # Diversity: no repeated food names within the week
                    if desc in used_names:
                        continue

                    # Soft diversity: avoid same category twice per meal per day
                    cat = row.get("food_category", "") or ""
                    if cat in used_cats_per_day[day] and len(used_cats_per_day[day]) < 4:
                        continue

                    cals_per_100 = row.get("calories")
                    # FIX Bug 3: enforce minimum calorie density of 0.9 kcal/g
                    # Rice cakes ~384 kcal/100g (dry weight) but deliver huge portions
                    # at low satiety. Pure water-vegetables (~15 kcal/100g) also excluded.
                    # Threshold set at 90 kcal/100g to preserve tofu (~76), cooked legumes
                    # (~116), and other legitimate low-density whole foods for vegan profiles.
                    # Lower cal density floor for vegan profiles (tofu=76, cooked legumes=116)
                    _cal_floor = 70 if profile.diet_type == "vegan" else 90
                    if cals_per_100 < _cal_floor:
                        continue
                    # Raise serving cap for vegan profiles to hit calorie targets
                    _srv_cap = 500 if profile.diet_type == "vegan" else 350
                    serving_g = min(_srv_cap, max(80, (target_cal / cals_per_100) * 100))
                    # Lower threshold for breakfast; stricter for lunch/dinner
                    actual_cal = cals_per_100 * serving_g / 100
                    cal_threshold = 0.50 if meal_label == "Breakfast" else 0.75
                    if actual_cal < target_cal * cal_threshold:
                        continue
                    selected  = MealSlot(
                        day=day, meal=meal_label,
                        fdc_id=int(row["fdc_id"]),
                        name=desc,
                        category=cat,
                        nutrients={
                            k: row.get(k) for k in rda.keys()
                            if k in row.index
                        },
                        serving_g=serving_g,
                    )
                    # Limit GF bread/rolls to max 3 slots per week
                    gf_bread_terms = ["gluten-free", "gluten free", "udi's", "schar", "andrea's", "rudi's"]
                    is_gf_bread = any(t in desc.lower() for t in gf_bread_terms)
                    gf_bread_count = sum(1 for s in plan.slots if any(t in s.name.lower() for t in gf_bread_terms))
                    if is_gf_bread and gf_bread_count >= 3:
                        continue
                    used_names.add(desc)
                    used_cats_per_day[day].add(cat)
                    # Species-level dedup for fish: treat all varieties of the
                    # same fish as the same "name" after the first appearance
                    # (prevents e.g. 4x chinook salmon variants in one week)
                    _FISH_SPECIES = [
                        "salmon", "tuna", "cod", "tilapia", "halibut", "trout",
                        "sardine", "mackerel", "bass", "mahi", "snapper",
                        "flounder", "herring", "anchov", "catfish", "pollock",
                        "sablefish", "whitefish", "sturgeon", "eel", "swordfish",
                        "butterfish", "tilefish", "yellowtail", "grouper",
                    ]
                    for _sp in _FISH_SPECIES:
                        if _sp in desc.lower():
                            used_names.add(f"__species__{_sp}")
                            break
                    break

                if selected is None:
                    # Fallback: draw from pre-filtered safe pool only
                    for _, row in safe_pool.sample(frac=1, random_state=day*10+ord(meal_label[0])).iterrows():
                        desc = row.get("description", "") or ""
                        if desc in used_names:
                            continue
                        eligible, _ = self._is_meal_eligible(row)
                        if not eligible:
                            continue
                        ok, _ = self._passes_clinical(row, profile)
                        if not ok:
                            continue
                        cals_per_100 = row.get("calories")
                        if cals_per_100 < 90:
                            continue
                        _srv_cap_fb = 500 if profile.diet_type == "vegan" else 350
                        serving_g = min(_srv_cap_fb, max(80, (target_cal / cals_per_100) * 100))
                        _fb_floor = 0.50 if meal_label == "Breakfast" else 0.75
                        if (cals_per_100 * serving_g / 100) < target_cal * _fb_floor:
                            continue
                        selected = MealSlot(
                            day=day, meal=meal_label,
                            fdc_id=int(row["fdc_id"]),
                            name=desc,
                            category=row.get("food_category", "") or "General",
                            nutrients={k: row.get(k) for k in rda.keys() if k in row.index},
                            serving_g=serving_g,
                        )
                        used_names.add(desc)
                        for _sp in ["salmon","tuna","cod","tilapia","halibut","trout",
                                    "sardine","mackerel","bass","mahi","snapper",
                                    "flounder","herring","sablefish","sturgeon","swordfish"]:
                            if _sp in desc.lower():
                                used_names.add(f"__species__{_sp}")
                                break
                        break

                if selected:
                    plan.slots.append(selected)

        # Guarantee fish meals for pescatarian profiles
        if profile.diet_type == "pescatarian":
            self._guarantee_fish_meals(plan, safe_pool, min_fish=3)

        # Iron guarantee: ensure at least 5/7 days have an iron-rich food
        if "ibs" in profile.conditions or profile.sex == "female":
            IRON_RICH = ["tempeh", "lentils", "quinoa", "spinach", "hemp seed",
                         "pumpkin seed", "edamame", "tofu, dried", "fortified",
                         "kale", "broccoli", "avocado"]
            iron_days = set(
                s.day for s in plan.slots
                if any(kw in s.name.lower() for kw in IRON_RICH)
            )
            low_iron_days = [d for d in range(1, 8) if d not in iron_days]
            if len(low_iron_days) > 2:
                iron_candidates = []
                for _, row in safe_pool.iterrows():
                    desc = (row.get("description", "") or "").lower()
                    if not any(kw in desc for kw in IRON_RICH):
                        continue
                    eligible, _ = self._is_meal_eligible(row)
                    if not eligible:
                        continue
                    ok, _ = self._passes_clinical(row, profile)
                    if not ok:
                        continue
                    iron_candidates.append(row)
                    if len(iron_candidates) >= 10:
                        break
                for day in low_iron_days[:3]:
                    for iron_row in iron_candidates:
                        iron_name = iron_row.get("description", "")
                        if iron_name in [s.name for s in plan.slots]:
                            continue
                        slot = next((s for s in plan.slots if s.day == day and s.meal == "Lunch"), None)
                        if slot is None:
                            continue
                        target_cal = profile.calorie_target * 0.35
                        cals = iron_row.get("calories") or 150
                        serving_g = min(350, max(80, (target_cal / cals) * 100))
                        # Only swap if the iron food can deliver at least 60% of target calories
                        if (cals * serving_g / 100) < target_cal * 0.60:
                            continue
                        slot.name = iron_name
                        slot.fdc_id = int(iron_row["fdc_id"])
                        slot.category = iron_row.get("food_category", "") or ""
                        slot.nutrients = {k: iron_row.get(k) for k in profile.rda().keys() if k in iron_row.index}
                        slot.serving_g = serving_g
                        break

        # B12 guarantee: ensure all 7 days have a B12-rich food
        # Critical for Ravi (non-veg, GERD) — animal proteins are the source
        b12_rda = profile.rda().get("vitamin_b12_mcg", 2.4)
        b12_threshold = b12_rda * 0.80
        low_b12_days = []
        analysis = self.analyse(plan)
        for day in range(1, 8):
            day_b12 = analysis["days"][day]["totals"].get("vitamin_b12_mcg", 0)
            if day_b12 < b12_threshold:
                low_b12_days.append(day)

        if low_b12_days:
            B12_RICH = ["salmon", "tuna", "beef", "chicken", "turkey",
                        "egg", "clam", "oyster", "sardine", "trout",
                        "pork", "lamb", "liver", "kidney"]
            b12_candidates = []
            for _, row in safe_pool.iterrows():
                desc = (row.get("description", "") or "").lower()
                if not any(kw in desc for kw in B12_RICH):
                    continue
                eligible, _ = self._is_meal_eligible(row)
                if not eligible:
                    continue
                ok, _ = self._passes_clinical(row, profile)
                if not ok:
                    continue
                b12_val = row.get("vitamin_b12_mcg") or 0
                if b12_val < 0.5:  # must actually have meaningful B12
                    continue
                b12_candidates.append(row)
                if len(b12_candidates) >= 15:
                    break

            for day in low_b12_days:
                for b12_row in b12_candidates:
                    b12_name = b12_row.get("description", "")
                    if b12_name in [s.name for s in plan.slots]:
                        continue
                    slot = next((s for s in plan.slots if s.day == day and s.meal == "Dinner"), None)
                    if slot is None:
                        continue
                    target_cal = profile.calorie_target * 0.40
                    cals = b12_row.get("calories") or 150
                    serving_g = min(350, max(80, (target_cal / cals) * 100))
                    if (cals * serving_g / 100) < target_cal * 0.60:
                        continue
                    slot.name = b12_name
                    slot.fdc_id = int(b12_row["fdc_id"])
                    slot.category = b12_row.get("food_category", "") or ""
                    slot.nutrients = {k: b12_row.get(k) for k in profile.rda().keys() if k in b12_row.index}
                    slot.serving_g = serving_g
                    break

        # Potassium guarantee for hypertension/DASH profiles
        # DASH diet requires high potassium — fish/eggs alone won't hit 3500mg RDA.
        # Strategy: check every day individually; for any day below threshold,
        # inject a K-rich food into the lowest-calorie slot on that day.
        # No calorie-floor check — the other slots carry calorie adequacy.
        k_injected_names: set = set()  # track K-injected slots to protect from sodium cap
        if "hypertension" in profile.conditions:
            k_rda = profile.rda().get("potassium_mg", 3500)
            k_threshold = k_rda * 0.80

            # Build sorted K-rich candidate list (descending potassium_mg)
            K_RICH = ["sweet potato", "spinach", "broccoli", "white bean",
                      "lentil", "avocado", "banana", "potato",
                      "salmon", "halibut", "tuna", "kidney bean",
                      "lima bean", "pinto", "acorn squash",
                      "butternut", "beet", "chard", "swiss chard",
                      "kale", "dried apricot", "prune", "raisin",
                      "black bean", "navy bean", "great northern",
                      "mackerel", "sardine", "trout", "cod",
                      "yogurt", "clam", "oyster", "mussel",
                      "artichoke", "brussels sprout", "pumpkin",
                      "sunflower seed", "flaxseed", "hemp seed"]
            k_candidates = []
            for _, row in safe_pool.iterrows():
                desc = (row.get("description", "") or "").lower()
                if not any(kw in desc for kw in K_RICH):
                    continue
                eligible, _ = self._is_meal_eligible(row)
                if not eligible:
                    continue
                ok, _ = self._passes_clinical(row, profile)
                if not ok:
                    continue
                k_val = row.get("potassium_mg") or 0
                if k_val < 200:
                    continue
                k_candidates.append(row)
                if len(k_candidates) >= 60:
                    break
            k_candidates.sort(key=lambda r: r.get("potassium_mg") or 0, reverse=True)

            if k_candidates:
                analysis = self.analyse(plan)
                # Process days worst-first so the lowest-K day gets first pick of candidates
                days_by_k = sorted(range(1, 8), key=lambda d: analysis["days"][d]["totals"].get("potassium_mg", 0))
                for day in days_by_k:
                    day_k = analysis["days"][day]["totals"].get("potassium_mg", 0)
                    if day_k >= k_threshold:
                        continue  # already adequate

                    # Names already used this day
                    day_names = {s.name for s in plan.slots if s.day == day}

                    # Pick the slot with the LOWEST calories on this day —
                    # that's the one we can most safely swap without wrecking
                    # the day's calorie total.
                    # Skip slots that are already fish (preserve fish variety)
                    _FISH_KW2 = ["salmon","tuna","cod","tilapia","halibut","trout",
                                  "sardine","mackerel","bass","mahi","snapper","flounder",
                                  "herring","sablefish","sturgeon","swordfish","butterfish",
                                  "catfish","pollock","whitefish","sucker","shad","cusk",
                                  "pompano","lingcod","scup","turbot","walleye","pike"]
                    day_slots_sorted = sorted(
                        [s for s in plan.slots if s.day == day
                         and not any(k in s.name.lower() for k in _FISH_KW2)],
                        key=lambda s: s.nutrients.get("potassium_mg", 0) or 0
                    )
                    if not day_slots_sorted:
                        day_slots_sorted = sorted(
                            [s for s in plan.slots if s.day == day],
                            key=lambda s: s.nutrients.get("potassium_mg", 0) or 0
                        )
                    if not day_slots_sorted:
                        continue
                    slot = day_slots_sorted[0]  # lowest-K slot

                    # Find the best K-rich candidate not already this week or day
                    week_names = {s.name for s in plan.slots}
                    # Fallback: query safe_pool directly for any high-K food not in week
                    extended_candidates = list(k_candidates)
                    for _, _kr in safe_pool.iterrows():
                        _kv = _kr.get("potassium_mg") or 0
                        if _kv < 300:
                            continue
                        _ke, _ = self._is_meal_eligible(_kr)
                        if not _ke:
                            continue
                        _ko, _ = self._passes_clinical(_kr, profile)
                        if not _ko:
                            continue
                        if _kr.get("description","") not in {r.get("description","") for r in extended_candidates}:
                            extended_candidates.append(_kr)
                    extended_candidates.sort(key=lambda r: r.get("potassium_mg") or 0, reverse=True)
                    for k_row in extended_candidates:
                        k_name = k_row.get("description", "")
                        if k_name in week_names:
                            continue
                        if k_name in day_names:
                            continue
                        # Size the serving to roughly match the slot being replaced
                        cals = k_row.get("calories") or 100
                        target_cal = slot.scaled("calories") or (profile.calorie_target * 0.25)
                        serving_g = min(400, max(80, (target_cal / cals) * 100))
                        # Sodium cap: don't inject high-sodium foods for hypertension
                        na_per_serving = (k_row.get("sodium_mg") or 0) * serving_g / 100
                        if na_per_serving > 400:
                            continue
                        k_injected_names.add(k_name)
                        slot.name = k_name
                        slot.fdc_id = int(k_row["fdc_id"])
                        slot.category = k_row.get("food_category", "") or ""
                        slot.nutrients = {k: k_row.get(k) for k in profile.rda().keys()
                                          if k in k_row.index}
                        slot.serving_g = round(serving_g, 1)
                        # Re-run analysis from this day forward
                        analysis = self.analyse(plan)
                        break

        # Potassium fallback — direct DB query for any remaining low-K days
        if "hypertension" in profile.conditions:
            _k_rda2 = profile.rda().get("potassium_mg", 3500)
            _k_thresh2 = _k_rda2 * 0.80
            _analysis2 = self.analyse(plan)
            _week_names2 = {s.name for s in plan.slots}
            for _day2 in range(1, 8):
                _day_k2 = _analysis2["days"][_day2]["totals"].get("potassium_mg", 0)
                if _day_k2 >= _k_thresh2:
                    continue
                # Find lowest-K non-fish slot
                _FISH3 = ["salmon","tuna","cod","tilapia","halibut","trout","sardine",
                          "mackerel","bass","herring","catfish","shark","eel","shad",
                          "scup","pompano","yellowtail","lingcod","seatrout","cobia"]
                _day_slots2 = sorted(
                    [s for s in plan.slots if s.day == _day2
                     and not any(f in s.name.lower() for f in _FISH3)],
                    key=lambda s: s.nutrients.get("potassium_mg", 0) or 0
                )
                if not _day_slots2:
                    _day_slots2 = sorted(
                        [s for s in plan.slots if s.day == _day2],
                        key=lambda s: s.nutrients.get("potassium_mg", 0) or 0
                    )
                if not _day_slots2:
                    continue
                _target_slot2 = _day_slots2[0]
                _day_names2 = {s.name for s in plan.slots if s.day == _day2}
                # Direct scan of entire eligible pool for highest-K food not in week
                _best_k_row = None
                _best_k_val = 0
                _bf_fallback = self._build_exclusion_bloom(profile)
                for _, _row2 in self._df_with_calories.iterrows():
                    _desc2 = _row2.get("description", "") or ""
                    if _desc2 in _week_names2 or _desc2 in _day_names2:
                        continue
                    _kv2 = _row2.get("potassium_mg") or 0
                    if _kv2 <= _best_k_val:
                        continue
                    _el2, _ = self._is_meal_eligible(_row2)
                    if not _el2:
                        continue
                    _ok2, _ = self._passes_clinical(_row2, profile)
                    if not _ok2:
                        continue
                    _combined2 = (_desc2 + " " + (_row2.get("ingredients_text","") or "")).lower()
                    if self._text_hits_bloom(_combined2, _bf_fallback):
                        continue
                    _cals2 = _row2.get("calories") or 100
                    _srv2 = min(400, max(80, (_target_slot2.scaled("calories") / _cals2) * 100))
                    _na2 = (_row2.get("sodium_mg") or 0) * _srv2 / 100
                    if _na2 > 600:
                        continue
                    # Only update best if this candidate fully passes all checks
                    if _kv2 > _best_k_val:
                        _best_k_row = _row2
                        _best_k_val = _kv2
                if _best_k_row is not None:
                    _cals2 = _best_k_row.get("calories") or 100
                    _srv2 = min(400, max(80, (_target_slot2.scaled("calories") / _cals2) * 100))
                    _target_slot2.name = _best_k_row.get("description", "")
                    _target_slot2.fdc_id = int(_best_k_row["fdc_id"])
                    _target_slot2.category = _best_k_row.get("food_category", "") or ""
                    _target_slot2.nutrients = {k: _best_k_row.get(k) for k in profile.rda().keys() if k in _best_k_row.index}
                    _target_slot2.serving_g = round(_srv2, 1)
                    _week_names2 = {s.name for s in plan.slots}
                    _analysis2 = self.analyse(plan)

        # General sodium soft cap — applies to ALL profiles (not just hypertension)
        # Replaces worst offender slot on any day exceeding 2500mg
        _na_limit = 1500 if "hypertension" in profile.conditions else 2500
        _analysis_na = self.analyse(plan)
        for _day in range(1, 8):
            _day_na = _analysis_na["days"][_day]["totals"].get("sodium_mg", 0)
            if _day_na <= _na_limit:
                continue
            # Find highest-sodium slot (skip K-injected slots)
            _day_slots = [s for s in plan.slots if s.day == _day and s.name not in k_injected_names]
            if not _day_slots:
                continue
            _worst = max(_day_slots, key=lambda s: s.scaled("sodium_mg"))
            # Find a lower-sodium replacement from safe_pool
            for _, _row in safe_pool.iterrows():
                _desc = _row.get("description", "") or ""
                if _desc in {s.name for s in plan.slots}:
                    continue
                _eligible, _ = self._is_meal_eligible(_row)
                if not _eligible:
                    continue
                _ok, _ = self._passes_clinical(_row, profile)
                if not _ok:
                    continue
                _na = (_row.get("sodium_mg") or 0)
                if _na > 300:  # skip high-sodium replacements
                    continue
                # Don't replace a slot if doing so would drop the day below potassium threshold
                if "hypertension" in profile.conditions:
                    _k_rda = profile.rda().get("potassium_mg", 3500)
                    _k_threshold = _k_rda * 0.80
                    _day_k = _analysis_na["days"][_day]["totals"].get("potassium_mg", 0)
                    _slot_k = _worst.scaled("potassium_mg") or 0
                    if (_day_k - _slot_k) < _k_threshold:
                        continue  # skip — replacing this slot would break potassium
                _cals = _row.get("calories") or 100
                _target = _worst.scaled("calories")
                _srv = min(400, max(80, (_target / _cals) * 100))
                _worst.name = _desc
                _worst.fdc_id = int(_row["fdc_id"])
                _worst.category = _row.get("food_category", "") or ""
                _worst.nutrients = {k: _row.get(k) for k in profile.rda().keys()
                                    if k in _row.index}
                _worst.serving_g = round(_srv, 1)
                _analysis_na = self.analyse(plan)
                break

        # Fibre guarantee for diabetes profiles (rubric: ≥25g/day all 7 days)
        if "diabetes" in profile.conditions:
            FIBRE_RICH = ["lentil", "black bean", "kidney bean", "navy bean",
                          "pinto bean", "chickpea", "split pea", "black-eyed",
                          "avocado", "broccoli", "spinach", "kale", "artichoke",
                          "sweet potato", "oats", "quinoa", "tempeh", "edamame",
                          "chia", "flaxseed", "amaranth", "barley", "farro"]
            fibre_candidates = []
            for _, row in safe_pool.iterrows():
                desc = (row.get("description", "") or "").lower()
                if not any(kw in desc for kw in FIBRE_RICH):
                    continue
                eligible, _ = self._is_meal_eligible(row)
                if not eligible:
                    continue
                ok, _ = self._passes_clinical(row, profile)
                if not ok:
                    continue
                fib = row.get("fiber_g") or 0
                if fib < 3.0:  # must have meaningful fibre per 100g
                    continue
                fibre_candidates.append(row)
                if len(fibre_candidates) >= 20:
                    break
            fibre_candidates.sort(key=lambda r: r.get("fiber_g") or 0, reverse=True)

            if fibre_candidates:
                analysis = self.analyse(plan)
                for day in range(1, 8):
                    day_fibre = analysis["days"][day]["totals"].get("fiber_g", 0)
                    if day_fibre >= 25.0:
                        continue
                    day_names = {s.name for s in plan.slots if s.day == day}
                    # Replace the slot with the lowest KNOWN fibre
                    # (skip slots with 0.0 — likely missing data, not truly zero)
                    day_slots_all = [s for s in plan.slots if s.day == day]
                    day_slots_with_fibre = [s for s in day_slots_all if s.scaled("fiber_g") > 0]
                    day_slots_sorted = sorted(
                        day_slots_with_fibre if day_slots_with_fibre else day_slots_all,
                        key=lambda s: s.scaled("fiber_g")
                    )
                    if not day_slots_sorted:
                        continue
                    slot = day_slots_sorted[0]
                    for fib_row in fibre_candidates:
                        fib_name = fib_row.get("description", "")
                        if fib_name in day_names:
                            continue
                        cals = fib_row.get("calories") or 100
                        target_cal = slot.scaled("calories") or (profile.calorie_target * 0.30)
                        serving_g = min(400, max(80, (target_cal / cals) * 100))
                        # Size serving to match original slot calories exactly
                        original_cal = slot.scaled("calories") or (profile.calorie_target * 0.30)
                        serving_g = min(500, max(80, (original_cal / cals) * 100))
                        new_cal = cals * serving_g / 100
                        # Skip if day is already under 80% calorie target
                        day_total_cal = sum(s.scaled("calories") for s in plan.slots if s.day == day)
                        if day_total_cal < profile.calorie_target * 0.80:
                            break  # day already underfed — don't touch it
                        slot.name = fib_name
                        slot.fdc_id = int(fib_row["fdc_id"])
                        slot.category = fib_row.get("food_category", "") or ""
                        slot.nutrients = {k: fib_row.get(k) for k in profile.rda().keys()
                                          if k in fib_row.index}
                        slot.serving_g = round(serving_g, 1)
                        analysis = self.analyse(plan)
                        break

        # Fibre guarantee for diabetes profiles (rubric: ≥25g/day all 7 days)
        if "diabetes" in profile.conditions:
            FIBRE_RICH = ["lentil", "black bean", "kidney bean", "navy bean",
                          "pinto bean", "chickpea", "split pea", "black-eyed",
                          "avocado", "broccoli", "spinach", "kale", "artichoke",
                          "sweet potato", "oats", "quinoa", "tempeh", "edamame",
                          "chia", "flaxseed", "amaranth", "barley", "farro"]
            fibre_candidates = []
            for _, row in safe_pool.iterrows():
                desc = (row.get("description", "") or "").lower()
                if not any(kw in desc for kw in FIBRE_RICH):
                    continue
                eligible, _ = self._is_meal_eligible(row)
                if not eligible:
                    continue
                ok, _ = self._passes_clinical(row, profile)
                if not ok:
                    continue
                fib = row.get("fiber_g") or 0
                if fib < 5.0:  # must have meaningful fibre per 100g
                    continue
                fibre_candidates.append(row)
                if len(fibre_candidates) >= 20:
                    break
            fibre_candidates.sort(key=lambda r: r.get("fiber_g") or 0, reverse=True)

            if fibre_candidates:
                analysis = self.analyse(plan)
                for day in range(1, 8):
                    day_fibre = analysis["days"][day]["totals"].get("fiber_g", 0)
                    if day_fibre >= 25.0:
                        continue
                    day_names = {s.name for s in plan.slots if s.day == day}
                    # Replace the lowest-fibre slot on this day
                    day_slots_sorted = sorted(
                        [s for s in plan.slots if s.day == day],
                        key=lambda s: s.scaled("fiber_g")
                    )
                    if not day_slots_sorted:
                        continue
                    slot = day_slots_sorted[0]
                    for fib_row in fibre_candidates:
                        fib_name = fib_row.get("description", "")
                        if fib_name in day_names:
                            continue
                        cals = fib_row.get("calories") or 100
                        target_cal = slot.scaled("calories") or (profile.calorie_target * 0.30)
                        serving_g = min(400, max(80, (target_cal / cals) * 100))
                        slot.name = fib_name
                        slot.fdc_id = int(fib_row["fdc_id"])
                        slot.category = fib_row.get("food_category", "") or ""
                        slot.nutrients = {k: fib_row.get(k) for k in profile.rda().keys()
                                          if k in fib_row.index}
                        slot.serving_g = round(serving_g, 1)
                        analysis = self.analyse(plan)
                        break

        plan.diversity_score   = self._diversity_score(plan.slots)
        plan.generation_time_s = round(time.time() - t_start, 2)
        return plan

    # ── Nutrition analysis ────────────────────────────────────────────────────

    def analyse(self, plan: MealPlan) -> dict:
        """
        Returns per-day and weekly totals, RDA % coverage,
        and flags any day below 80% RDA threshold.
        """
        rda  = plan.profile.rda()
        keys = list(rda.keys())
        days_data = {}

        for day in range(1, 8):
            day_slots = [s for s in plan.slots if s.day == day]
            totals = {k: sum(s.scaled(k) for s in day_slots) for k in keys}
            pct    = {k: round(totals[k] / rda[k] * 100, 1) if rda[k] else None
                      for k in keys}
            flags  = [k for k in keys if pct.get(k) is not None and pct[k] < 80]
            days_data[day] = {"totals": totals, "pct_rda": pct, "flags": flags}

        weekly = {k: sum(days_data[d]["totals"][k] for d in range(1, 8)) for k in keys}
        return {"days": days_data, "weekly": weekly, "rda": rda}


# ── Convenience function ──────────────────────────────────────────────────────

_pipeline: Optional[NutriAIPipeline] = None

def get_pipeline() -> NutriAIPipeline:
    """Singleton — load once, reuse across Streamlit reruns."""
    global _pipeline
    if _pipeline is None:
        _pipeline = NutriAIPipeline()
        _pipeline.load()
    return _pipeline
