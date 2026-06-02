"""
app.py
------
NutriAI — BAX-423 Final Project
Streamlit UI: intake → plan generation → nutrition analysis → export

Run:
    streamlit run app.py
"""

import io
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from fpdf import FPDF

# Local modules
from pipeline import (
    UserProfile, NutriAIPipeline, MealPlan,
    ALLERGEN_KEYWORDS, DIET_EXCLUSIONS, get_pipeline,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriAI — Personalized Diet Planner",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main palette */
  :root {
    --green:  #2E7D32;
    --teal:   #00796B;
    --amber:  #F57F17;
    --red:    #C62828;
    --bg:     #F9FBF9;
    --card:   #FFFFFF;
    --border: #E0E0E0;
  }

  .main { background: var(--bg); }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #2E7D32 0%, #00796B 100%);
    color: white;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
  }
  .hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; }
  .hero p  { margin: .4rem 0 0; opacity: .88; font-size: 1.05rem; }

  /* Metric cards */
  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .val  { font-size: 1.7rem; font-weight: 700; color: var(--green); }
  .metric-card .lbl  { font-size: .82rem; color: #666; margin-top: .2rem; }

  /* Meal card */
  .meal-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .7rem;
  }
  .meal-card .meal-label {
    font-size: .75rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: var(--teal); margin-bottom: .25rem;
  }
  .meal-card .meal-name {
    font-size: 1rem; font-weight: 600; color: #1a1a1a;
  }
  .meal-card .meal-meta {
    font-size: .82rem; color: #666; margin-top: .2rem;
  }

  /* Pill badges */
  .pill {
    display: inline-block;
    padding: .2rem .65rem;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
    margin: .15rem .1rem;
  }
  .pill-green  { background: #E8F5E9; color: #2E7D32; }
  .pill-red    { background: #FFEBEE; color: #C62828; }
  .pill-amber  { background: #FFF8E1; color: #F57F17; }
  .pill-blue   { background: #E3F2FD; color: #1565C0; }

  /* Flag row */
  .flag-row {
    background: #FFF8E1;
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: .5rem .9rem;
    margin: .3rem 0;
    font-size: .88rem;
  }
  .excl-row {
    background: #FFEBEE;
    border-left: 4px solid var(--red);
    border-radius: 0 8px 8px 0;
    padding: .5rem .9rem;
    margin: .2rem 0;
    font-size: .85rem;
  }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 700;
    color: var(--green); margin: 1.2rem 0 .6rem;
    border-bottom: 2px solid #E8F5E9; padding-bottom: .3rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #F1F8F1; }

  /* Timer badge */
  .timer-badge {
    display: inline-block;
    background: #E8F5E9; color: #2E7D32;
    border-radius: 20px; padding: .3rem .9rem;
    font-weight: 700; font-size: .95rem;
    border: 1px solid #A5D6A7;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

MACRO_KEYS = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
MICRO_KEYS = ["iron_mg", "calcium_mg", "vitamin_b12_mcg", "vitamin_d_mcg",
              "zinc_mg", "sodium_mg", "potassium_mg", "magnesium_mg", "omega3_g"]

NUTRIENT_LABELS = {
    "calories":        "Calories (kcal)",
    "protein_g":       "Protein (g)",
    "carbs_g":         "Carbs (g)",
    "fat_g":           "Fat (g)",
    "fiber_g":         "Fibre (g)",
    "iron_mg":         "Iron (mg)",
    "calcium_mg":      "Calcium (mg)",
    "vitamin_b12_mcg": "Vitamin B12 (µg)",
    "vitamin_d_mcg":   "Vitamin D (µg)",
    "zinc_mg":         "Zinc (mg)",
    "sodium_mg":       "Sodium (mg)",
    "potassium_mg":    "Potassium (mg)",
    "magnesium_mg":    "Magnesium (mg)",
    "omega3_g":        "Omega-3 (g)",
}

MEAL_ICONS = {"Breakfast": "🌅", "Lunch": "☀️", "Dinner": "🌙"}
CONDITION_LABELS = {
    "ibs":           "IBS (Low-FODMAP)",
    "gerd":          "GERD / Acid Reflux",
    "diabetes":      "Type 2 Diabetes",
    "hypertension":  "Hypertension (DASH)",
}


def rda_color(pct: float) -> str:
    if pct >= 100: return "#2E7D32"
    if pct >= 80:  return "#00796B"
    if pct >= 50:  return "#F57F17"
    return "#C62828"


def format_nutrient(key: str, val: float) -> str:
    if val is None: return "—"
    if key == "calories":        return f"{val:.0f}"
    if key in ("protein_g","carbs_g","fat_g","fiber_g","omega3_g"): return f"{val:.1f}g"
    if key in ("iron_mg","zinc_mg","magnesium_mg"):  return f"{val:.1f}mg"
    if key in ("calcium_mg","sodium_mg","potassium_mg"): return f"{val:.0f}mg"
    if key in ("vitamin_b12_mcg","vitamin_d_mcg"):   return f"{val:.1f}µg"
    return f"{val:.1f}"


# ── PDF Export ────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """Strip non-latin1 characters for fpdf compatibility."""
    return text.encode("latin-1", errors="replace").decode("latin-1")

def generate_pdf(plan: MealPlan, analysis: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_fill_color(46, 125, 50)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(8)
    pdf.cell(0, 10, sanitize("NutriAI - 7-Day Meal Plan"), align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Generated for: {plan.profile.name}  |  "
             f"Diet: {plan.profile.diet_type.title()}  |  "
             f"Target: {plan.profile.calorie_target:.0f} kcal/day", align="C", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Summary row
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(232, 245, 233)
    pdf.cell(0, 8, f"  Generation time: {plan.generation_time_s}s   |   "
             f"Diversity score: {plan.diversity_score:.2f}   |   "
             f"Total meals: {len(plan.slots)}", ln=True, fill=True)
    pdf.ln(4)

    # Meal plan
    for day in range(1, 8):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(200, 230, 201)
        pdf.cell(0, 7, f"  Day {day}", ln=True, fill=True)
        pdf.ln(1)

        day_slots = [s for s in plan.slots if s.day == day]
        for slot in day_slots:
            icon = {"Breakfast":"[B]","Lunch":"[L]","Dinner":"[D]"}[slot.meal]
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(20, 6, f"  {icon}", ln=False)
            pdf.set_font("Helvetica", "", 10)
            cal = slot.scaled("calories")
            prot = slot.scaled("protein_g")
            pdf.cell(0, 6,
                     f"{sanitize(slot.name[:55])}  "
                     f"({slot.serving_g:.0f}g)  -  "
                     f"{cal:.0f} kcal  |  {prot:.1f}g protein",
                     ln=True)

        # Day totals
        day_analysis = analysis["days"].get(day, {})
        totals = day_analysis.get("totals", {})
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5,
                 f"    Day totals: "
                 f"{totals.get('calories',0):.0f} kcal  |  "
                 f"{totals.get('protein_g',0):.1f}g protein  |  "
                 f"{totals.get('fiber_g',0):.1f}g fibre  |  "
                 f"{totals.get('sodium_mg',0):.0f}mg sodium",
                 ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # RDA flags
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, sanitize("Nutrient Gap Analysis"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, sanitize("Days below 80% RDA threshold are flagged in red."), ln=True)
    pdf.ln(3)

    rda = analysis["rda"]
    for day in range(1, 8):
        d = analysis["days"][day]
        flags = d.get("flags", [])
        pdf.set_font("Helvetica", "B", 10)
        flag_str = ", ".join(NUTRIENT_LABELS.get(f, f) for f in flags) if flags else "All targets met - OK"
        pdf.set_fill_color(255, 235, 238 if flags else 232)
        pdf.cell(0, 7, f"  Day {day}: {flag_str}".encode("latin-1", errors="replace").decode("latin-1"), ln=True, fill=True)

    return bytes(pdf.output())


# ── Sidebar: User Profile Form ────────────────────────────────────────────────

def sidebar_profile() -> UserProfile:
    st.sidebar.markdown("## 👤 Your Profile")

    name = st.sidebar.text_input("Name", value="User")
    col1, col2 = st.sidebar.columns(2)
    age  = col1.number_input("Age", 18, 90, 35)
    sex  = col2.selectbox("Sex", ["male", "female"])

    col3, col4 = st.sidebar.columns(2)
    weight_lbs = col3.number_input("Weight (lbs)", 66.0, 440.0, 154.0, step=1.0)
    weight = round(weight_lbs * 0.453592, 1)
    ht_col1, ht_col2 = st.sidebar.columns(2)
    height_ft = ht_col1.number_input("Height (ft)", 3, 8, 5, step=1)
    height_in = ht_col2.number_input("Height (in)", 0, 11, 10, step=1)
    height = round(((height_ft * 12) + height_in) * 2.54, 1)

    # BMR-based calorie suggestion (Mifflin-St Jeor)
    if sex == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    suggested_cal = int(bmr * 1.55)  # moderate activity
    calorie_target = st.sidebar.number_input(
        "Daily Calorie Target (kcal)",
        800, 5000, suggested_cal, step=50,
        help=f"Suggested based on your stats: {suggested_cal} kcal"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🍽️ Diet & Restrictions")

    diet_type = st.sidebar.selectbox(
        "Diet Type",
        ["non-vegetarian", "vegetarian", "vegan", "pescatarian"],
        format_func=lambda x: x.replace("-", " ").title()
    )

    no_pork = False
    if diet_type in ["non-vegetarian", "pescatarian"]:
        no_pork = st.sidebar.checkbox("No pork / halal preference")

    allergen_options = list(ALLERGEN_KEYWORDS.keys())
    allergens = st.sidebar.multiselect(
        "Allergens / Intolerances (optional)",
        options=allergen_options,
        default=[],
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🏥 Clinical Conditions")

    condition_options = list(CONDITION_LABELS.keys())
    conditions = st.sidebar.multiselect(
        "Clinical Conditions (optional)",
        options=condition_options,
        default=[],
        format_func=lambda x: CONDITION_LABELS[x]
    )

    return UserProfile(
        name=name, age=age, sex=sex,
        weight_kg=weight, height_cm=height,
        calorie_target=float(calorie_target),
        diet_type=diet_type,
        allergens=allergens,
        conditions=conditions,
        no_pork=no_pork,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

def render_plan_tab(plan: MealPlan, analysis: dict):
    st.markdown('<div class="section-header">📅 7-Day Meal Plan</div>',
                unsafe_allow_html=True)

    # Day selector
    day = st.selectbox("View day:", list(range(1, 8)),
                       format_func=lambda d: f"Day {d}")

    day_slots = [s for s in plan.slots if s.day == day]
    day_analysis = analysis["days"].get(day, {})
    totals = day_analysis.get("totals", {})
    pct    = day_analysis.get("pct_rda", {})
    flags  = day_analysis.get("flags", [])

    # Meal cards
    for slot in day_slots:
        icon  = MEAL_ICONS[slot.meal]
        cal   = slot.scaled("calories")
        prot  = slot.scaled("protein_g")
        carbs = slot.scaled("carbs_g")
        fat   = slot.scaled("fat_g")
        st.markdown(f"""
        <div class="meal-card">
          <div class="meal-label">{icon} {slot.meal}</div>
          <div class="meal-name">{slot.name}</div>
          <div class="meal-meta">
            Serving: {slot.serving_g:.0f}g &nbsp;|&nbsp;
            {cal:.0f} kcal &nbsp;|&nbsp;
            Protein: {prot:.1f}g &nbsp;|&nbsp;
            Carbs: {carbs:.1f}g &nbsp;|&nbsp;
            Fat: {fat:.1f}g
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Day macro summary bar chart
    st.markdown('<div class="section-header">📊 Day Nutrient Coverage vs RDA</div>',
                unsafe_allow_html=True)

    rda = analysis["rda"]
    display_keys = [k for k in MACRO_KEYS + MICRO_KEYS if k in pct and pct[k] is not None]
    pct_vals = [min(pct[k], 150) for k in display_keys]
    colors   = [rda_color(pct[k]) for k in display_keys]
    labels   = [NUTRIENT_LABELS[k] for k in display_keys]

    fig = go.Figure(go.Bar(
        x=pct_vals, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in pct_vals],
        textposition="outside",
    ))
    fig.add_vline(x=80,  line_dash="dash", line_color="#F57F17",
                  annotation_text="80% RDA", annotation_position="top right")
    fig.add_vline(x=100, line_dash="dot",  line_color="#2E7D32")
    fig.update_layout(
        height=420, margin=dict(l=0, r=60, t=10, b=10),
        xaxis_title="% of RDA", yaxis=dict(autorange="reversed"),
        plot_bgcolor="#F9FBF9", paper_bgcolor="#F9FBF9",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Flags
    if flags:
        st.markdown('<div class="section-header">⚠️ Nutrient Gaps</div>',
                    unsafe_allow_html=True)
        for f in flags:
            actual = totals.get(f, 0)
            target = rda.get(f, 0)
            st.markdown(
                f'<div class="flag-row">⚠️ <b>{NUTRIENT_LABELS.get(f, f)}</b> — '
                f'{format_nutrient(f, actual)} vs target {format_nutrient(f, target)} '
                f'({pct.get(f, 0):.0f}% RDA)</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ All nutrient targets met for this day (≥80% RDA)")


def render_weekly_tab(plan: MealPlan, analysis: dict):
    st.markdown('<div class="section-header">📈 Weekly Nutrient Overview</div>',
                unsafe_allow_html=True)

    rda = analysis["rda"]

    # Heatmap: days × nutrients
    days_list = list(range(1, 8))
    keys = [k for k in MACRO_KEYS[:5] + MICRO_KEYS[:6]]
    matrix = []
    for day in days_list:
        pct = analysis["days"][day]["pct_rda"]
        matrix.append([min(pct.get(k, 0) or 0, 150) for k in keys])

    df_heat = pd.DataFrame(matrix,
                           index=[f"Day {d}" for d in days_list],
                           columns=[NUTRIENT_LABELS[k] for k in keys])

    fig = px.imshow(
        df_heat,
        color_continuous_scale=["#C62828", "#F57F17", "#FFF9C4", "#A5D6A7", "#2E7D32"],
        zmin=0, zmax=150,
        text_auto=".0f",
        aspect="auto",
    )
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="% RDA"),
        paper_bgcolor="#F9FBF9",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weekly totals table
    st.markdown('<div class="section-header">Weekly Totals</div>',
                unsafe_allow_html=True)

    rows = []
    for k in keys:
        weekly_val = analysis["weekly"].get(k, 0)
        daily_avg  = weekly_val / 7
        rda_val    = rda.get(k, 1) or 1
        rows.append({
            "Nutrient": NUTRIENT_LABELS[k],
            "Weekly Total": format_nutrient(k, weekly_val),
            "Daily Avg": format_nutrient(k, daily_avg),
            "% RDA (avg)": f"{daily_avg / rda_val * 100:.0f}%",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_exclusions_tab(plan: MealPlan):
    st.markdown('<div class="section-header">🚫 Why Foods Were Excluded</div>',
                unsafe_allow_html=True)

    if not plan.exclusions:
        st.info("No exclusions recorded for this profile.")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    day_filter  = col1.selectbox("Filter by day", ["All"] + [f"Day {d}" for d in range(1, 8)])
    meal_filter = col2.selectbox("Filter by meal", ["All", "Breakfast", "Lunch", "Dinner"])

    shown = plan.exclusions
    if day_filter != "All":
        d = int(day_filter.split()[1])
        shown = [e for e in shown if e["day"] == d]
    if meal_filter != "All":
        shown = [e for e in shown if e["meal"] == meal_filter]

    st.caption(f"Showing {len(shown)} of {len(plan.exclusions)} exclusions")

    for e in shown[:200]:   # cap display at 200
        st.markdown(
            f'<div class="excl-row">'
            f'<b>Day {e["day"]} {e["meal"]}</b> — {e["food"][:60]}<br>'
            f'<span style="color:#C62828">⛔ {e["reason"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_benchmark_tab(plan: MealPlan):
    st.markdown('<div class="section-header">⚡ BAX-423 Technique Benchmarks</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Two BAX-423 techniques are integrated into the core pipeline.
    Benchmarks are measured at runtime on your actual dataset.
    """)

    # ── Bloom Filter ──────────────────────────────────────────────────────────
    st.markdown("### Technique 1 — Bloom Filter (Sketching)")
    st.markdown("""
    **What it does:** Encodes all allergen, diet, and clinical exclusion keywords
    into a space-efficient probabilistic bit-array. Every candidate food is checked
    in O(1) time — zero false negatives guaranteed (safe for exclusion).

    **Why it matters:** With 8,000+ foods and 21 meal slots, naive set-intersection
    checks every keyword for every food. The Bloom filter amortises this to a
    constant-time lookup regardless of how many keywords are in the exclusion set.
    """)

    bm = plan.benchmark.get("bloom", {})
    if bm:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Naive (set scan)",    f'{bm.get("method_naive_ms", 0):.1f} ms')
        col2.metric("Bloom filter",        f'{bm.get("method_bloom_ms", 0):.1f} ms')
        col3.metric("Speedup",             f'{bm.get("speedup", 0):.1f}×')
        col4.metric("Filter size",         f'{bm.get("bloom_size_kb", 0)} KB')
        st.caption(f"Benchmark on {bm.get('n_samples', 0)} random food descriptions.")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    st.markdown("### Technique 2 — FAISS IVF Embeddings (ANN Search)")
    st.markdown("""
    **What it does:** Each food description is encoded as a 384-dimensional
    sentence embedding (all-MiniLM-L6-v2). An IVF (Inverted File) index partitions
    the embedding space into Voronoi cells, enabling sub-linear nearest-neighbour
    retrieval. Meal queries like *"low-FODMAP vegetarian breakfast high-iron"*
    are embedded and matched semantically — not just by keyword.

    **Why it matters:** Keyword search would miss foods whose descriptions don't
    contain the exact query words. Embeddings capture *meaning*, so "spinach salad"
    is retrieved for "high-iron vegetarian lunch" even without the word "iron".
    """)

    bm2 = plan.benchmark.get("faiss", {})
    if bm2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Flat (brute-force)",  f'{bm2.get("flat_ms", 0):.1f} ms')
        col2.metric("IVF index",           f'{bm2.get("ivf_ms", 0):.1f} ms')
        col3.metric("Speedup",             f'{bm2.get("speedup", 0):.1f}×')
    else:
        st.info("FAISS benchmark runs on first index build. "
                "Re-run with `force_reindex=True` to regenerate.")

    # ── Generation time ───────────────────────────────────────────────────────
    st.markdown("### End-to-End Generation Time")
    t = plan.generation_time_s
    color = "#2E7D32" if t < 60 else "#C62828"
    st.markdown(
        f'<div style="font-size:2.5rem;font-weight:800;color:{color}">'
        f'⏱ {t:.2f}s</div>'
        f'<div style="color:#666;font-size:.9rem">Target: &lt; 60 seconds '
        f'{"✅" if t < 60 else "❌"}</div>',
        unsafe_allow_html=True
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # Hero
    st.markdown("""
    <div class="hero">
      <h1>🥗 NutriAI</h1>
      <p>Personalized 7-day meal planning — clinically safe, nutrient-optimized, under 60 seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar profile
    profile = sidebar_profile()

    # Load pipeline (cached after first load)
    with st.spinner("Loading food database & AI index …"):
        pipeline = get_pipeline()

    # Generate button
    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button(
        "🚀 Generate My Meal Plan",
        type="primary",
        use_container_width=True,
    )

    # ── Session state ──────────────────────────────────────────────────────────
    if "plan" not in st.session_state:
        st.session_state.plan = None
        st.session_state.analysis = None

    if generate_btn:
        with st.spinner("Generating your personalized plan …"):
            progress = st.progress(0, text="Filtering unsafe foods …")
            plan = pipeline.generate(profile)
            progress.progress(70, text="Analysing nutrients …")
            analysis = pipeline.analyse(plan)
            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

        st.session_state.plan = plan
        st.session_state.analysis = analysis

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.plan:
        plan     = st.session_state.plan
        analysis = st.session_state.analysis

        # Top metrics row
        t = plan.generation_time_s
        timer_color = "#2E7D32" if t < 60 else "#C62828"
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-card"><div class="val">{len(plan.slots)}</div>'
                    f'<div class="lbl">Meals Planned</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="val">'
                    f'{plan.profile.calorie_target:.0f}</div>'
                    f'<div class="lbl">kcal Target/Day</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="val">{plan.diversity_score:.2f}</div>'
                    f'<div class="lbl">Diversity Score</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="val">{len(plan.exclusions)}</div>'
                    f'<div class="lbl">Foods Excluded</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-card">'
                    f'<div class="val" style="color:{timer_color}">{t:.1f}s</div>'
                    f'<div class="lbl">Generation Time</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Profile summary pills
        pills = []
        pills.append(f'<span class="pill pill-green">{profile.diet_type.replace("-"," ").title()}</span>')
        for c in profile.conditions:
            pills.append(f'<span class="pill pill-blue">{CONDITION_LABELS[c]}</span>')
        for a in profile.allergens:
            pills.append(f'<span class="pill pill-red">No {a.replace("_"," ").title()}</span>')
        st.markdown(" ".join(pills), unsafe_allow_html=True)
        st.markdown("")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📅 Meal Plan", "📈 Weekly Analysis",
            "🚫 Why Excluded", "⚡ Benchmarks"
        ])

        with tab1:
            render_plan_tab(plan, analysis)
        with tab2:
            render_weekly_tab(plan, analysis)
        with tab3:
            render_exclusions_tab(plan)
        with tab4:
            render_benchmark_tab(plan)

        # Export row
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        # PDF export
        with col_a:
            pdf_bytes = generate_pdf(plan, analysis)
            st.download_button(
                label="📄 Download PDF Plan",
                data=pdf_bytes,
                file_name=f"NutriAI_{profile.name.replace(' ','_')}_7day_plan.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # CSV export
        with col_b:
            rows = []
            for s in plan.slots:
                rows.append({
                    "Day": s.day,
                    "Meal": s.meal,
                    "Food": s.name,
                    "Category": s.category,
                    "Serving_g": round(s.serving_g, 1),
                    "Calories": round(s.scaled("calories"), 1),
                    "Protein_g": round(s.scaled("protein_g"), 1),
                    "Carbs_g": round(s.scaled("carbs_g"), 1),
                    "Fat_g": round(s.scaled("fat_g"), 1),
                    "Fiber_g": round(s.scaled("fiber_g"), 1),
                })
            csv_str = pd.DataFrame(rows).to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Plan",
                data=csv_str,
                file_name=f"NutriAI_{profile.name.replace(' ','_')}_7day_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Exclusions CSV
        with col_c:
            excl_csv = pd.DataFrame(plan.exclusions).to_csv(index=False) \
                       if plan.exclusions else "food,reason,day,meal\n"
            st.download_button(
                label="🚫 Download Exclusions Log",
                data=excl_csv,
                file_name=f"NutriAI_{profile.name.replace(' ','_')}_exclusions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#888;">
          <div style="font-size:4rem">🥗</div>
          <div style="font-size:1.2rem;font-weight:600;margin:.5rem 0">
            Fill in your profile and click Generate
          </div>
          <div style="font-size:.9rem">
            NutriAI will build a clinically-safe, personalized 7-day meal plan in under 60 seconds.
          </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
