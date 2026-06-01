"""
Simple Streamlit app for developer salary prediction.
Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── allow imports from src/ ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import (
    ED_LEVEL_ORDINAL,
    REMOTE_ORDINAL,
    AGE_ORDINAL,
    ORGSIZE_ORDINAL,
    DEV_TYPE_OPTIONS,
    INDUSTRY_OPTIONS,
    add_interaction_features,
)

# ── config ─────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "salary_pipeline.pkl"

# Countries that appear in our training data (top-25 + Other)
COUNTRY_OPTIONS = [
    "United States of America",
    "Germany",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "Canada",
    "Brazil",
    "France",
    "Netherlands",
    "Poland",
    "Australia",
    "Spain",
    "Italy",
    "Sweden",
    "Israel",
    "Ukraine",
    "Switzerland",
    "Russian Federation",
    "Turkey",
    "Austria",
    "Norway",
    "Portugal",
    "Romania",
    "Czech Republic",
    "Denmark",
    "Belgium",
    "Other",
]


# ── helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load the trained pipeline once and cache it."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model not found at `{MODEL_PATH}`.\n"
            "Run `python src/train.py` first to train the model."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


# ── page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dev Salary Predictor",
    page_icon="💰",
    layout="centered",
)

st.title("💰 Developer Salary Predictor")
st.markdown(
    "Predict your expected annual salary based on the "
    "**Stack Overflow 2025 Developer Survey** data."
)

pipeline = load_pipeline()

# ── input form ─────────────────────────────────────────────────────────────

st.header("Your Details")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", COUNTRY_OPTIONS, index=0)
    dev_type = st.selectbox("Primary Role", DEV_TYPE_OPTIONS, index=0)
    industry = st.selectbox("Industry", INDUSTRY_OPTIONS, index=0)
    ed_label = st.selectbox(
        "Education Level",
        list(ED_LEVEL_ORDINAL.keys())[:-1],  # exclude "Other" default
        index=4,  # default = Bachelor's
    )
    remote_label = st.selectbox(
        "Work Arrangement",
        ["In-person", "Hybrid", "Remote"],
        index=2,
    )

with col2:
    years_code = st.slider("Years of Coding", 0, 50, 5)
    work_exp = st.slider("Years of Professional Experience", 0, 50, 3)
    age_label = st.selectbox("Age Range", list(AGE_ORDINAL.keys()), index=2)
    org_label = st.selectbox(
        "Company Size",
        list(ORGSIZE_ORDINAL.keys()),
        index=4,  # 100-499
    )
    is_manager = st.radio(
        "Role Type",
        ["Individual Contributor", "Manager / Team Lead"],
        index=0,
    )

with st.expander("Tech Skills (optional — uses defaults if skipped)"):
    lang_count = st.slider("Programming Languages Known", 0, 30, 4)
    db_count = st.slider("Databases Worked With", 0, 20, 2)
    platform_count = st.slider("Platforms / Cloud Providers", 0, 15, 2)
    tool_count = st.slider("Tools Used at Work", 0, 30, 5)

# ── predict ────────────────────────────────────────────────────────────────

if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
    # Build a single-row DataFrame matching the training features
    input_data = pd.DataFrame([{
        "Country": country,
        "DevType": dev_type,
        "Industry": industry,
        "YearsCode": float(years_code),
        "WorkExp": float(work_exp),
        "EdLevel": ED_LEVEL_ORDINAL[ed_label],
        "RemoteWork": REMOTE_ORDINAL[remote_label],
        "Employment": 1,  # assume full-time (most common)
        "Age": AGE_ORDINAL[age_label],
        "OrgSize": ORGSIZE_ORDINAL[org_label],
        "ICorPM": 1 if is_manager == "Manager / Team Lead" else 0,
        "LanguageCount": float(lang_count),
        "DatabaseCount": float(db_count),
        "PlatformCount": float(platform_count),
        "ToolCountWork": float(tool_count),
    }])

    # Add the same interaction features used during training
    input_data = add_interaction_features(input_data)

    # Predict (model returns log-salary → expm1 to get USD)
    log_pred = pipeline.predict(input_data)[0]
    salary_usd = np.expm1(log_pred)

    st.divider()
    st.metric(
        label="Predicted Annual Salary (USD)",
        value=f"${salary_usd:,.0f}",
    )

    st.caption(
        "ℹ️ This prediction is based on a machine-learning model trained on "
        "the Stack Overflow 2025 Developer Survey. Actual salaries vary "
        "based on many factors not captured in the model."
    )

# ── footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: XGBoost on Stack Overflow 2025 Survey")
