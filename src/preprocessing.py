"""
Data cleaning utilities for developer salary prediction.

v3 changes (from v2):
────────────────────
1. EdLevel & RemoteWork converted to ordinal integers (preserves ordering)
2. Country, DevType, Industry left as strings for target encoding in pipeline
3. Employment filter: keep only Full-time & Freelance (removes noisy rows)
4. Employment mapped to binary: Full-time=1, Freelance=0
5. Interaction features: YearsCode², WorkExp², experience ratio, tech breadth
6. Exported constants so the Streamlit app can reuse the same mappings
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────
TARGET = "ConvertedCompYearly"
LOG_TARGET = "log_salary"

TOP_N_COUNTRIES = 25
SALARY_MIN = 10_000
SALARY_MAX = 500_000

# Columns pulled from the raw Stack Overflow survey CSV
SELECTED_FEATURES = [
    "Country",
    "YearsCode",
    "EdLevel",
    "Employment",
    "LanguageHaveWorkedWith",
    "DevType",
    "OrgSize",
    "RemoteWork",
    "WorkExp",
    "Industry",
    "Age",
    "ICorPM",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "ToolCountWork",
]

# These will be target-encoded inside the sklearn pipeline
TARGET_ENC_FEATURES = ["Country", "DevType", "Industry"]

# ── ordinal mappings (exported for Streamlit app) ─────────────────────────

ED_LEVEL_ORDINAL: dict[str, int] = {
    "Primary school": 0,
    "High school": 1,
    "Some college": 2,
    "Associate's": 3,
    "Bachelor's": 4,
    "Master's": 5,
    "Professional": 6,
    "Other": 2,
}

REMOTE_ORDINAL: dict[str, int] = {
    "In-person": 0,
    "Hybrid": 1,
    "Remote": 2,
    "Other": 1,
}

AGE_ORDINAL: dict[str, int] = {
    "Under 18": 0,
    "18-24": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55-64": 5,
    "65+": 6,
}

ORGSIZE_ORDINAL: dict[str, int] = {
    "Just me / freelancer": 0,
    "2-9": 1,
    "10-19": 2,
    "20-99": 3,
    "100-499": 4,
    "500-999": 5,
    "1,000-4,999": 6,
    "5,000-9,999": 7,
    "10,000+": 8,
}

DEV_TYPE_OPTIONS = [
    "Full-stack", "Back-end", "Front-end", "Data/ML",
    "DevOps/Cloud", "Mobile", "Embedded/Hardware",
    "Security", "Management", "Other",
]

INDUSTRY_OPTIONS = [
    "Information Services, IT, Software Development, or other Technology",
    "Financial Services",
    "Manufacturing, Transportation, or Supply Chain",
    "Healthcare",
    "Retail and Consumer Services",
    "Insurance",
    "Higher Education",
    "Advertising Services",
    "Government",
    "Other",
]

# Employment categories we keep (rest are filtered out)
EMPLOYMENT_KEEP = ["Full-time", "Freelance/Self-employed"]


# ── individual column cleaners ─────────────────────────────────────────────

def clean_years_code(series: pd.Series) -> pd.Series:
    """Convert YearsCode to numeric; 'More than 50 years' → 51."""
    series = series.copy()
    series = series.replace("More than 50 years", "51")
    series = series.replace("Less than 1 year", "0")
    return pd.to_numeric(series, errors="coerce")


def clean_work_exp(series: pd.Series) -> pd.Series:
    """Convert WorkExp (years of professional experience) to numeric."""
    series = series.copy()
    series = series.replace("More than 50 years", "51")
    series = series.replace("Less than 1 year", "0")
    return pd.to_numeric(series, errors="coerce")


def clean_education(series: pd.Series) -> pd.Series:
    """Map verbose education strings → short label → ordinal integer."""
    verbose_to_short = {
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Some college/university study without earning a degree": "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High school",
        "Associate degree (A.A., A.S., etc.)": "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Primary/elementary school": "Primary school",
        "Other (please specify):": "Other",
    }
    short = series.map(verbose_to_short).fillna("Other")
    return short.map(ED_LEVEL_ORDINAL)


def clean_employment(series: pd.Series) -> pd.Series:
    """Simplify multi-value employment strings to core categories."""
    def _simplify(val: str) -> str:
        if pd.isna(val):
            return np.nan
        val = str(val).lower()
        if "full-time" in val or "employed" in val:
            return "Full-time"
        if "independent contractor" in val or "freelancer" in val or "self-employed" in val:
            return "Freelance/Self-employed"
        if "part-time" in val:
            return "Part-time"
        if "student" in val:
            return "Student"
        return "Other"

    return series.apply(_simplify)


def clean_age(series: pd.Series) -> pd.Series:
    """Map verbose age bands to ordinal integers."""
    mapping = {
        "Under 18 years old": 0,
        "18-24 years old": 1,
        "25-34 years old": 2,
        "35-44 years old": 3,
        "45-54 years old": 4,
        "55-64 years old": 5,
        "65 years or older": 6,
        "Prefer not to say": np.nan,
    }
    return series.map(mapping)


def clean_org_size(series: pd.Series) -> pd.Series:
    """Map organisation-size bands to ordinal integers."""
    mapping = {
        "Just me - I am a freelancer, sole proprietor, etc.": 0,
        "2 to 9 employees": 1,
        "10 to 19 employees": 2,
        "20 to 99 employees": 3,
        "100 to 499 employees": 4,
        "500 to 999 employees": 5,
        "1,000 to 4,999 employees": 6,
        "5,000 to 9,999 employees": 7,
        "10,000 or more employees": 8,
    }
    return series.map(mapping)


def clean_icorpm(series: pd.Series) -> pd.Series:
    """Map IC-or-PM role to binary: 1 = manager/lead, 0 = IC."""
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).lower()
        if "manager" in v or "lead" in v:
            return 1
        return 0
    return series.apply(_map)


def clean_remote_work(series: pd.Series) -> pd.Series:
    """Map remote-work values to ordinal integers (0=in-person, 2=remote)."""
    verbose_to_short = {
        "Remote": "Remote",
        "Hybrid (some remote, some in-person)": "Hybrid",
        "In-person": "In-person",
    }
    short = series.map(verbose_to_short).fillna("Other")
    return short.map(REMOTE_ORDINAL)


def clean_industry(series: pd.Series) -> pd.Series:
    """Keep top industries; merge the rest into 'Other'."""
    top = INDUSTRY_OPTIONS[:-1]  # everything except the trailing "Other"
    return series.apply(lambda x: x if x in top else "Other").fillna("Other")


def clean_dev_type(series: pd.Series) -> pd.Series:
    """Extract primary developer role from potentially multi-value field."""
    def _primary(val):
        if pd.isna(val):
            return "Other"
        roles = str(val).split(";")
        first = roles[0].strip()
        low = first.lower()
        if "full-stack" in low:
            return "Full-stack"
        if "back-end" in low or "backend" in low:
            return "Back-end"
        if "front-end" in low or "frontend" in low:
            return "Front-end"
        if "data scientist" in low or "machine learning" in low or "ml" in low:
            return "Data/ML"
        if "data engineer" in low or "data analyst" in low:
            return "Data/ML"
        if "devops" in low or "site reliability" in low or "cloud" in low:
            return "DevOps/Cloud"
        if "mobile" in low:
            return "Mobile"
        if "embedded" in low or "hardware" in low:
            return "Embedded/Hardware"
        if "security" in low:
            return "Security"
        if "manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"

    return series.apply(_primary)


def count_items(series: pd.Series) -> pd.Series:
    """Count semicolon-separated items in a column; NaN if blank."""
    def _count(val):
        if pd.isna(val) or val == "":
            return np.nan
        return len(str(val).split(";"))

    return series.apply(_count)


def group_rare_countries(series: pd.Series, top_n: int = TOP_N_COUNTRIES) -> pd.Series:
    """Keep only the top-N most common countries; replace others with 'Other'."""
    top_countries = series.value_counts().head(top_n).index.tolist()
    return series.apply(lambda x: x if x in top_countries else "Other")


# ── interaction features ──────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that capture non-linear relationships.

    Why these features help:
    - YearsCode_sq   : diminishing returns on salary growth per year
    - WorkExp_sq     : same for professional experience
    - Exp_ratio      : what fraction of coding time was professional
    - Tech_breadth   : total tools known = a proxy for seniority
    """
    df = df.copy()

    if "YearsCode" in df.columns:
        df["YearsCode_sq"] = df["YearsCode"] ** 2

    if "WorkExp" in df.columns:
        df["WorkExp_sq"] = df["WorkExp"] ** 2

    if "YearsCode" in df.columns and "WorkExp" in df.columns:
        df["Exp_ratio"] = df["WorkExp"] / (df["YearsCode"].fillna(0) + 1)

    tech_cols = [c for c in ["LanguageCount", "DatabaseCount", "PlatformCount"]
                 if c in df.columns]
    if tech_cols:
        df["Tech_breadth"] = df[tech_cols].fillna(0).sum(axis=1)

    return df


# ── main loader ────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the Stack Overflow survey CSV and return a clean DataFrame
    ready for the sklearn pipeline with log1p-transformed salary.

    Parameters
    ----------
    filepath : str
        Path to the raw survey CSV.

    Returns
    -------
    pd.DataFrame
        Features + LOG_TARGET column.
    """
    # step 0 — load raw data
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw shape: {df.shape}\n")

    # step 1 — salary filter
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(SALARY_MIN, SALARY_MAX)]
    print(f"Shape after salary filter: {df.shape}")

    # step 2 — select available columns
    cols_needed = SELECTED_FEATURES + [TARGET]
    cols_available = [c for c in cols_needed if c in df.columns]
    missing = set(cols_needed) - set(cols_available)
    if missing:
        print(f"  Missing column(s) (will be skipped): {missing}")
    df = df[cols_available].copy()

    # step 3 — clean individual columns
    if "YearsCode" in df.columns:
        df["YearsCode"] = clean_years_code(df["YearsCode"])

    if "WorkExp" in df.columns:
        df["WorkExp"] = clean_work_exp(df["WorkExp"])

    if "EdLevel" in df.columns:
        df["EdLevel"] = clean_education(df["EdLevel"])       # → ordinal int

    if "Employment" in df.columns:
        df["Employment"] = clean_employment(df["Employment"])

    if "Age" in df.columns:
        df["Age"] = clean_age(df["Age"])

    if "OrgSize" in df.columns:
        df["OrgSize"] = clean_org_size(df["OrgSize"])

    if "ICorPM" in df.columns:
        df["ICorPM"] = clean_icorpm(df["ICorPM"])

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = clean_remote_work(df["RemoteWork"])  # → ordinal int

    if "Industry" in df.columns:
        df["Industry"] = clean_industry(df["Industry"])

    if "DevType" in df.columns:
        df["DevType"] = clean_dev_type(df["DevType"])

    if "LanguageHaveWorkedWith" in df.columns:
        df["LanguageCount"] = count_items(df["LanguageHaveWorkedWith"])
        df = df.drop(columns=["LanguageHaveWorkedWith"])

    if "DatabaseHaveWorkedWith" in df.columns:
        df["DatabaseCount"] = count_items(df["DatabaseHaveWorkedWith"])
        df = df.drop(columns=["DatabaseHaveWorkedWith"])

    if "PlatformHaveWorkedWith" in df.columns:
        df["PlatformCount"] = count_items(df["PlatformHaveWorkedWith"])
        df = df.drop(columns=["PlatformHaveWorkedWith"])

    if "ToolCountWork" in df.columns:
        df["ToolCountWork"] = pd.to_numeric(df["ToolCountWork"], errors="coerce")

    if "Country" in df.columns:
        df["Country"] = group_rare_countries(df["Country"])

    # step 4 — filter employment (keep Full-time & Freelance only)
    if "Employment" in df.columns:
        before = len(df)
        df = df[df["Employment"].isin(EMPLOYMENT_KEEP)]
        df["Employment"] = (df["Employment"] == "Full-time").astype(int)
        print(f"Employment filter: {before} → {len(df)} rows "
              f"(kept Full-time & Freelance)")

    # step 5 — add interaction / polynomial features
    df = add_interaction_features(df)

    # step 6 — log-transform salary
    df[LOG_TARGET] = np.log1p(df[TARGET])
    df = df.drop(columns=[TARGET])

    # step 7 — drop rows where ALL features are NaN
    df = df.dropna(how="all")

    print(f"Clean data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isna().sum().to_string()}")

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
    Return (target_enc_cols, numeric_cols) from the cleaned DataFrame.

    target_enc_cols : string columns that will be target-encoded in the pipeline
    numeric_cols    : all numeric columns (impute → scale)
    """
    non_target = [c for c in df.columns if c != LOG_TARGET]
    target_enc = [c for c in TARGET_ENC_FEATURES if c in non_target]
    num_cols = [c for c in non_target
                if c not in target_enc
                and pd.api.types.is_numeric_dtype(df[c])]
    return target_enc, num_cols
