"""
Energy Model Regression Analyzer — Streamlit Web App
=====================================================
IPMVP-aligned baseline regression modeling tool.

Run with:
    streamlit run app.py
"""

import io
import os
import tempfile

import pandas as pd
import streamlit as st

from energy_analyzer.models import run_all_models
from energy_analyzer.output import _build_chart_png, _build_results_df, export_results

st.set_page_config(
    page_title="Energy Model Regression Analyzer",
    page_icon="⚡",
    layout="wide",
)

st.title("Energy Model Regression Analyzer")
st.caption("IPMVP-aligned baseline regression modeling")

# ---------------------------------------------------------------------------
# Step 1: File upload
# ---------------------------------------------------------------------------
st.header("1. Upload Data")
uploaded = st.file_uploader(
    "Upload your energy data file",
    type=["xlsx", "xls", "csv"],
    help="Supported formats: .xlsx, .xls, .csv",
)

if uploaded is None:
    st.info("Upload a file above to get started.")
    st.stop()

ext = os.path.splitext(uploaded.name)[1].lower()
try:
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns**")
with st.expander("Preview data (first 5 rows)"):
    st.dataframe(df.head(5), use_container_width=True)

# ---------------------------------------------------------------------------
# Step 2: Column mapping
# ---------------------------------------------------------------------------
st.header("2. Map Columns")

all_cols = list(df.columns)

col_left, col_right = st.columns(2)
with col_left:
    date_col = st.selectbox("Date / Period column", all_cols, index=0)
with col_right:
    remaining_for_energy = [c for c in all_cols if c != date_col]
    energy_col = st.selectbox(
        "Energy Consumption column (dependent variable)",
        remaining_for_energy,
        index=0,
    )

remaining = [c for c in all_cols if c not in (date_col, energy_col)]

iv_cols = st.multiselect(
    "Independent Variables — select all that apply",
    remaining,
    help="e.g. HDD, CDD, production units, occupancy",
)

mode_options = [c for c in remaining if c not in iv_cols]
mode_cols = st.multiselect(
    "Mode of Operation flag columns (optional)",
    mode_options,
    help="e.g. a weekday/weekend indicator column",
)

# Auto-detect HDD / CDD from selected IV names
hdd_col = None
cdd_col = None

if iv_cols:
    hdd_matches = [c for c in iv_cols if "hdd" in c.lower()]
    cdd_matches = [c for c in iv_cols if "cdd" in c.lower()]

    if len(hdd_matches) == 1:
        hdd_col = hdd_matches[0]
    elif len(hdd_matches) > 1:
        chosen = st.selectbox(
            "Multiple columns may represent HDD — which is correct?",
            hdd_matches + ["None / Not applicable"],
        )
        hdd_col = None if chosen == "None / Not applicable" else chosen

    if len(cdd_matches) == 1:
        cdd_col = cdd_matches[0]
    elif len(cdd_matches) > 1:
        chosen = st.selectbox(
            "Multiple columns may represent CDD — which is correct?",
            cdd_matches + ["None / Not applicable"],
        )
        cdd_col = None if chosen == "None / Not applicable" else chosen

if hdd_col or cdd_col:
    st.caption(
        f"Auto-detected: HDD = **{hdd_col or 'none'}**, CDD = **{cdd_col or 'none'}**"
    )

# ---------------------------------------------------------------------------
# Step 3: Run analysis
# ---------------------------------------------------------------------------
st.header("3. Run Analysis")

if not iv_cols:
    st.info("Select at least one independent variable above to enable the analysis.")
    st.stop()

if not st.button("Run Analysis", type="primary"):
    st.stop()

column_map = {
    "date_col":   date_col,
    "energy_col": energy_col,
    "iv_cols":    iv_cols,
    "mode_cols":  mode_cols,
    "hdd_col":    hdd_col,
    "cdd_col":    cdd_col,
}

with st.spinner("Running model permutations — this may take a moment..."):
    try:
        results, best_model = run_all_models(df, column_map)
    except Exception as e:
        st.error(f"Error during model fitting: {e}")
        st.stop()

if not results:
    st.warning("No model results were produced. Check your data and column mapping.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 4: Results
# ---------------------------------------------------------------------------
st.header("4. Results")

passing_count = sum(1 for r in results if r["Pass / Fail"] == "Pass")

m1, m2, m3 = st.columns(3)
m1.metric("Total Models Tested", len(results))
m2.metric("Models Passing", passing_count)
m3.metric("Models Failing", len(results) - passing_count)

if best_model:
    st.success(
        f"**Recommended Model: {best_model['Model ID']}** | "
        f"R² = {best_model['R²']:.4f} | "
        f"Variables: {best_model['Independent Variables']} | "
        f"Baseline: {best_model['Baseline Period']} | "
        f"Type: {best_model['Model Type']} | "
        f"Mode: {best_model['Operating Mode Segment']}"
    )
else:
    st.warning(
        "No models passed all statistical thresholds (R² > 0.75, p < 0.05, |t| > 2.0). "
        "Review data quality and consider additional variables."
    )

# Results table
st.subheader("Model Results Table")


def _highlight(row):
    if row.get("Recommended") == "Yes":
        return ["background-color: #FFEB9C"] * len(row)
    if row.get("Pass / Fail") == "Pass":
        return ["background-color: #C6EFCE"] * len(row)
    return ["background-color: #FFC7CE"] * len(row)


results_df = _build_results_df(results)
st.dataframe(
    results_df.style.apply(_highlight, axis=1),
    use_container_width=True,
    height=420,
)

# Chart for best model
if best_model:
    chart_png = _build_chart_png(best_model, energy_col)
    if chart_png:
        st.subheader("Best Model Chart")
        st.image(chart_png, use_container_width=True)

# ---------------------------------------------------------------------------
# Step 5: Download Excel
# ---------------------------------------------------------------------------
st.subheader("Download Results")

with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
    tmp_path = tmp.name

try:
    export_results(results, best_model, energy_col=energy_col, output_path=tmp_path)
    with open(tmp_path, "rb") as f:
        excel_bytes = f.read()

    st.download_button(
        label="Download Results (Excel)",
        data=excel_bytes,
        file_name="energy_model_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
except Exception as e:
    st.warning(f"Excel export failed: {e}")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
