# NutriAI — AI-Assisted Development Log (`prompts.md`)

**Course:** BAX-423 Big Data · UC Davis Graduate School of Management  
**Student:** Larry Pastor  
**Project:** NutriAI — Automated 7-Day Personalized Meal Plan Generator  
**AI Tool Used:** Claude (Anthropic) via claude.ai  

---

## Overview

This document records how AI assistance (Claude) was used during the development of NutriAI. All core architectural decisions, technique selections, and design choices were made by the student. Claude was used as a coding assistant and debugging partner — analogous to using Stack Overflow, documentation, or a senior developer for pair programming.

---

## Phase 1 — Project Architecture & Setup

**Prompt category:** System design  
**What I asked:** How to structure a Python + Streamlit meal plan generator that uses Big Data techniques from BAX-423, specifically Bloom filters and FAISS indexing, with a USDA food database.

**AI contribution:**
- Suggested the 3-file architecture: `fetch_usda_data.py` (data pipeline), `pipeline.py` (core engine), `app.py` (UI)
- Recommended SQLite + CSV dual storage for offline reliability
- Outlined the generation pipeline: safe_pool → Bloom filter → FAISS retrieval → clinical rules → post-processing

**Student decisions:**
- Chose FAISS IVF + Bloom filter as the two BAX-423 techniques (motivated by rubric benchmarking requirement)
- Chose USDA FoodData Central as the data source
- Chose Python + Streamlit as the tech stack

---

## Phase 2 — USDA Data Pipeline (`fetch_usda_data.py`)

**Prompt category:** API integration  
**What I asked:** How to fetch food data from the USDA FoodData Central API and parse nutrient values correctly.

**Key debugging assistance:**
- The `/foods/list` endpoint does not return nutrient data — Claude identified that the POST `/foods` batch endpoint with `format: full` is required
- Nutrient parsing: the `foodNutrients` structure uses nested `n['nutrient']['id']` integer keys with `n['amount']` values (not `'nutrientId'`/`'value'` as the docs suggest)
- Deduplication strategy: `lowercase(description) + food_category`

**Outcome:** Custom pipeline fetching 8,000 Foundation + SR Legacy foods with 14 nutrients each, stored in `foods.db` and `foods.csv`.

---

## Phase 3 — Core Pipeline (`pipeline.py`)

**Prompt category:** Algorithm implementation

### 3a. Bloom Filter
**What I asked:** Implement a Bloom filter for allergen/diet exclusion with zero false negatives.

**AI contribution:**
- Implemented `BloomFilter` class with configurable capacity and false-positive rate
- Suggested whole-word boundary regex matching to prevent partial matches (e.g. "wheat" matching "buckwheat")
- Parameterised at capacity=50,000, fp_rate=0.001

### 3b. FAISS IVF Index
**What I asked:** Build a FAISS IVF approximate nearest-neighbour index over food descriptions.

**Key debugging assistance:**
- `sentence-transformers` caused segfaults on Python 3.13/macOS — Claude diagnosed the PyTorch incompatibility and suggested replacing with scikit-learn TF-IDF + TruncatedSVD (LSA) as a pure numpy/sklearn alternative
- Correct import block after the segfault fix: `TfidfVectorizer`, `TruncatedSVD`, `normalize` from sklearn
- Cache invalidation: FAISS index files must be deleted and rebuilt whenever pipeline indexing logic changes

### 3c. Clinical Rules
**What I asked:** Implement dietary rules for IBS (FODMAP), GERD, Type 2 Diabetes, and Hypertension.

**AI contribution:**
- FODMAP keyword blocklist (garlic, onion, wheat, high-fructose, certain legumes)
- GERD trigger list (citrus, tomato, fried, caffeine, chocolate, spicy, peppermint)
- Diabetes high-GI blocklist (white rice, sugar, candy, instant cereals)
- DASH sodium cap (≤1500mg/day) and potassium floor (≥80% RDA)

### 3d. Post-Processing Nutrient Guarantees
**What I asked:** Ensure specific micronutrients are met for clinical profiles even when FAISS retrieval doesn't naturally select nutrient-dense foods.

**AI contribution:**
- Architecture pattern: run guarantees after plan generation as a second pass
- Implemented guarantees for: iron (Priya/IBS), B12 (Ravi/GERD), potassium (James/hypertension), fibre (Mei/diabetes)
- Key fix: potassium injector must check week-wide `used_names` to avoid placing duplicate foods

---

## Phase 4 — Streamlit UI (`app.py`)

**Prompt category:** UI/UX implementation  
**What I asked:** Build a 4-tab Streamlit interface with profile intake, meal plan display, nutrient gap analysis, exclusion log, and benchmark comparison.

**AI contribution:**
- 4-tab layout: Meal Plan / Weekly Analysis / Why Excluded / Benchmarks
- Sidebar profile form with allergen multi-select and clinical condition checkboxes
- PDF export using reportlab, CSV export using pandas
- Nutrient gap analysis with RDA percentage bars
- Benchmark tab showing Bloom vs set-scan and FAISS IVF vs flat index side-by-side

---

## Phase 5 — Testing & Persona Debugging

**Prompt category:** Quality assurance  
**What I asked:** Run scorecard tests against all 4 rubric personas and fix failures.

**Key debugging sessions:**

| Issue | Root Cause | Fix |
|---|---|---|
| Ravi — pork on Day 7 | `no_pork` flag only in Bloom filter, not in `safe_pool` build | Added pool-level pork filter |
| James — potassium 4/7 days | Potassium injector had calorie floor that prevented low-cal K-rich foods | Removed calorie floor, target lowest-calorie slot |
| Mei — fibre 5/7 days | Sprouted lentils had `fiber=nan` in USDA DB; fibre guarantee picking 0-fibre slots | Blocked sprouted beans; fixed slot selection logic |
| James — diversity 0.79 | Potassium injector placing duplicate food names across days | Added week-wide `used_names` check to injector |
| Uncooked rice in plan | Rice precooked/instant variants passing the ineligible filter | Pool-level regex block for uncooked grain variants |
| Cookies in James pool | GF cookies in "Baked Products" category bypassing "Snacks" block | Pool-level description keyword block |
| Fat cuts in non-veg plan | "Separable fat", "intermuscular fat", "subcutaneous fat" not blocked | Added to pool-level `_block_desc` pattern |

**Final scorecard:** 31/31 checks passed across all 4 personas (100%).

---

## Phase 6 — Deployment

**Prompt category:** DevOps  
**What I asked:** How to deploy the Streamlit app to Streamlit Community Cloud via GitHub.

**AI contribution:**
- `.gitignore` configuration (exclude `tfidf.pkl` at 15MB, `scorecard.py`, `fix_potassium.py`)
- Git init, commit, and push sequence
- Streamlit Community Cloud deploy configuration (repo, branch, main file path)

**Outcome:** App live at `nutriai-bax423.streamlit.app`

---

## Phase 7 — Brief & Documentation

**Prompt category:** Technical writing  
**What I asked:** Generate `brief.pdf` covering all 5 rubric dimensions.

**AI contribution:**
- Full PDF layout using reportlab
- Persona pass/fail table, benchmark comparison table, architecture table, clinical rules table, rubric dimension summary

---

## Summary of AI Usage

| Phase | AI Role | Student Role |
|---|---|---|
| Architecture | Suggested file structure and pipeline stages | Chose techniques, data source, tech stack |
| Data pipeline | Diagnosed API parsing bugs | Decided to use USDA FoodData Central |
| Bloom filter | Implemented class and parameterisation | Chose Bloom filter as BAX-423 technique |
| FAISS index | Implemented index, diagnosed segfault | Chose FAISS IVF as BAX-423 technique |
| Clinical rules | Implemented keyword lists and logic | Defined which conditions to support |
| Post-processors | Implemented guarantee algorithms | Identified which nutrients needed guarantees |
| UI | Implemented 4-tab layout and exports | Designed UX flow and feature requirements |
| Testing | Wrote scorecard, diagnosed failures, wrote fixes | Ran tests, reviewed outputs, approved fixes |
| Deployment | Provided git and Streamlit commands | Executed deployment, managed GitHub account |
| Documentation | Drafted brief.pdf and prompts.md | Reviewed and approved content |

All code was reviewed, tested, and approved by the student before submission. The student ran every `python3` patch command manually and verified output at each step.
