"""
Data cleaning utilities for developer salary csv.
We will import this file in train.py and or app.py

v2 – expanded feature set:
  Added DevType, OrgSize, RemoteWork, WorkExp, Industry, Age,
  ICorPM, DatabaseHaveWorkedWith (count), PlatformHaveWorkedWith
  (count), ToolCountWork.
  Applies log1p transform to the target so XGBoost models a
  smoother distribution; inverse-transform is applied at
  predict time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────
TARGET = "ConvertedCompYearly"
LOG_TARGET = "log_salary"

TOP_N_COUNTRIES = 25          # was 15 – more granularity
SALARY_MIN = 10_000
SALARY_MAX = 500_000

# All columns we want to pull from the raw CSV
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
    """Standardise verbose education labels into short categories."""
    mapping = {
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Some college/university study without earning a degree": "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High school",
        "Associate degree (A.A., A.S., etc.)": "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Primary/elementary school": "Primary school",
        "Other (please specify):": "Other",
    }
    return series.map(mapping).fillna("Other")


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
    """Map verbose age bands to ordinal integers for the model."""
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
    """Standardise remote-work values to short categories."""
    mapping = {
        "Remote": "Remote",
        "Hybrid (some remote, some in-person)": "Hybrid",
        "In-person": "In-person",
    }
    return series.map(mapping).fillna("Other")


def clean_industry(series: pd.Series) -> pd.Series:
    """Keep top industries; merge the rest into 'Other'."""
    top = [
        "Information Services, IT, Software Development, or other Technology",
        "Financial Services",
        "Manufacturing, Transportation, or Supply Chain",
        "Healthcare",
        "Retail and Consumer Services",
        "Insurance",
        "Higher Education",
        "Advertising Services",
        "Government",
    ]
    return series.apply(lambda x: x if x in top else "Other").fillna("Other")


def clean_dev_type(series: pd.Series) -> pd.Series:
    """Extract primary developer role from potentially multi-value field."""
    # Keep only first role listed (before the semicolon)
    def _primary(val):
        if pd.isna(val):
            return "Other"
        roles = str(val).split(";")
        first = roles[0].strip()
        # Group into broad buckets
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


# ── main loader ────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the Stack Overflow survey CSV and return a clean DataFrame
    ready for the sklearn pipeline, with log1p-transformed salary.

    Parameters
    ----------
    filepath : str
        Path to the raw survey CSV.

    Returns
    -------
    pd.DataFrame
        Features + LOG_TARGET column (log1p of ConvertedCompYearly).
    """
    # step 0 — load
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw shape: {df.shape} \n")

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
        df["EdLevel"] = clean_education(df["EdLevel"])

    if "Employment" in df.columns:
        df["Employment"] = clean_employment(df["Employment"])

    if "Age" in df.columns:
        df["Age"] = clean_age(df["Age"])

    if "OrgSize" in df.columns:
        df["OrgSize"] = clean_org_size(df["OrgSize"])

    if "ICorPM" in df.columns:
        df["ICorPM"] = clean_icorpm(df["ICorPM"])

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = clean_remote_work(df["RemoteWork"])

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

    # step 4 — log-transform salary (makes target distribution more normal)
    df[LOG_TARGET] = np.log1p(df[TARGET])
    df = df.drop(columns=[TARGET])

    # step 5 — drop rows where ALL features are NaN (edge case)
    df = df.dropna(how="all")

    print(f"Clean data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isna().sum().to_string()}")

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
    Return (categorical_columns, numeric_columns) from the cleaned df,
    excluding the log-salary target.
    """
    non_target = [c for c in df.columns if c != LOG_TARGET]
    cat_cols = df[non_target].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df[non_target].select_dtypes(include=["number"]).columns.tolist()
    return cat_cols, num_cols
