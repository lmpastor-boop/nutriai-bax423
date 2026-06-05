# NutriAI — Automated Diet Plan Builder
**BAX-423 Big Data · Spring 2026 · UC Davis GSM**

> Generates a personalized, clinically-safe 7-day meal plan in under 60 seconds.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull USDA food data (one-time, ~5 min)
#    Get a free API key at https://fdc.nal.usda.gov/api-guide.html
python fetch_usda_data.py --api-key YOUR_KEY_HERE --max-items 12000

# 3. Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Project Structure

```
code/
  app.py               — Streamlit UI (main entry point)
  pipeline.py          — Core pipeline: Bloom filter + FAISS + clinical filters
  fetch_usda_data.py   — Data pipeline: USDA API → SQLite + CSV snapshot

data/
  foods.db             — SQLite food database (offline snapshot, 8,000+ items)
  foods.csv            — CSV copy of the same data
  faiss.index          — FAISS IVF index (auto-built on first run)
  embeddings.npy       — Sentence embeddings cache
  fdc_ids.npy          — FDC ID array aligned to embeddings

brief.pdf              — Technical brief (architecture, techniques, personas)
prompts.md             — AI prompts used during development
```

---

## BAX-423 Techniques

| Technique | Where | Benchmarked? |
|---|---|---|
| **Bloom Filter** (sketching) | `pipeline.py` → `BloomFilter`, `_build_exclusion_bloom` | ✅ vs naive set-scan |
| **FAISS IVF** (embeddings + ANN) | `pipeline.py` → `build_faiss_index`, `_retrieve_candidates` | ✅ vs brute-force flat index |

Benchmark results are displayed live in the **⚡ Benchmarks** tab of the app.

---

## 6 Core Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | Clinical Condition Filtering | `_passes_clinical()` — FODMAP, GERD triggers, GI proxy, DASH sodium cap |
| 2 | Allergy Detection & Exclusion | `BloomFilter` + `ALLERGEN_KEYWORDS` — zero false negatives |
| 3 | Dietary Preference Handling | `DIET_EXCLUSIONS` — vegan / vegetarian / pescatarian / non-veg |
| 4 | Diversity Engine | `_diversity_score()` — no repeat names, category rotation per day |
| 5 | Macro & Micronutrient Analysis | `analyse()` — 14 nutrients, RDA % coverage, daily gap flags |
| 6 | Sub-60s Generation | FAISS ANN + Bloom O(1) exclusion — logged and displayed in UI |

---

## Test Personas

| Persona | Conditions | Diet | Key Constraints |
|---|---|---|---|
| Priya | IBS | Vegetarian | No dairy, no high-FODMAP, iron ≥ 80% RDA |
| Ravi | GERD | Non-vegetarian | No gluten, no GERD triggers, diversity ≥ 0.7 |
| Mei | Type 2 Diabetes | Vegan | GI ≤ 55, no tree nuts, fibre ≥ 25g/day |
| James | Hypertension | Pescatarian | Sodium ≤ 1500mg/day, no soy, potassium ≥ 80% RDA |

---

## Hosting

Deployed to **Streamlit Community Cloud** — push to GitHub, connect repo at
[share.streamlit.io](https://share.streamlit.io). No server config needed.

The `data/foods.db` offline snapshot ensures the grader can run the app
without an API key after initial setup.

---

## Data Source

**USDA FoodData Central** — Foundation + SR Legacy food types.
- API: https://fdc.nal.usda.gov/api-guide.html
- Nutrients captured: calories, protein, carbs, fat, fibre, iron, calcium,
  B12, vitamin D, zinc, sodium, potassium, magnesium, omega-3
- Deduplication: by `lowercase(description) + food_category`
