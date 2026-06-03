"""
    Data cleaning utilities for developer salary csv.
    We will import this file in train.py and or app.py
"""
import pandas as pd
import numpy as np

# constants
TARGET = 'ConvertedCompYearly'
TOP_N_COUNTRIES = 25
SALARY_MIN = 10_000
SALARY_MAX = 500_000

SELECTED_FEATURES = ['Country', 'YearsCode', 'EdLevel', 'Employment', 'LanguageHaveWorkedWith',
                     #new selected features, v2
                     'DevType', # developer role 
                     'OrgSize', # company size
                     'RemoteWork', # remote/ hybrid/ in-person
                     'WorkExp', # years of professional experience.
                     'Industry', # eg tech, finance, healthcare etc
                     'Age',
                     'ICorPM', #Individual contributor or manager
                     'DatabaseHaveWorkedWith',
                     'PlatformHaveWorkedWith',
                     'ToolCountWork'
                     ]

# Cleaning functions.
def clean_years_code(series: pd.Series) -> pd.Series:
    """
        convert the column YearsCode to numeric
    """
    series = series.copy()
    series = pd.to_numeric(series, errors='coerce')
    return series

def clean_education(series: pd.Series) -> pd.Series:
    """
        Cleaning education categories, ie starndardizing them a bit.
    """
    mapping = {
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)":"Master's",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Some college/university study without earning a degree": "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High school",
        "Associate degree (A.A., A.S., etc.)": "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Primary/elementary school":"Primary school",
        "Other (please specify):": "Other"
    }
    return series.map(mapping).fillna("Other")

def clean_employment(series: pd.Series) -> pd.Series:
    """
        simplifying employment into core categories
    """
    def simplify(val):
        if pd.isna(val):
            return np.nan
        
        val = str(val)
        if 'full-time' in val.lower() or 'employed' in val.lower():
            return 'Full-time'
        elif 'Independent contractor, freelancer, or self-employed' in val.lower():
            return 'Freelance/Self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
        
    return series.apply(simplify)

def count_languages(series: pd.Series) -> pd.Series:
    """
        Convert semicolon-separated language list into a count.

        example: val = 'Python;JavaScript;SQL;HTML' 
        
        we apply val.split(';)
        
        -> val =>  [Python, JavaScript, SQL, HTML]

        return len(val) # 4
    """
    def _count(val):
        if pd.isna(val) or val == '':
            return np.nan
        return len(str(val).split(';'))
    
    return series.apply(_count)

def group_rare_countries(series: pd.Series, top_n: int = TOP_N_COUNTRIES) -> pd.Series:
    """
        Keep only the top N most common countries. Replace all others with 'Other'.
    """
    top_countries = series.value_counts().head(top_n).index.tolist()
    return series.apply(lambda x: x if x in top_countries else "Other")

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
        Load the stack overflow survey csv and return a clean df ready for the sklearn pipeline.

        parameters to pass: filepath to the survey csv

        returns the df ie features + target.
    """
    # step 0 - loading the data
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw shape: {df.shape} \n")

    #step 1 - keep only rows with a valid salary and do some filtering.
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(SALARY_MIN, SALARY_MAX)]
    print(f"Shape after the salary filter: {df.shape}")

    # step 1.5 - v2 of model
    df['log_salary'] = np.log1p(df[TARGET])

    # step 2 -  features + target and check whether they exist in the df
    cols_needed = SELECTED_FEATURES + ['log_salary']
    cols_available = [c for c in cols_needed if c in df.columns]

    missing_cols = set(cols_needed) - set(cols_available)
    if missing_cols:
        print(f" You dont have column(s): {missing_cols} in your dataset")

    df = df[cols_available].copy()
    print(f"Selected {len(cols_available)} columns, expected 6 columns")

    # step 3 - cleaning individual columns
    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_years_code(df['YearsCode'])

    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])

    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])

    if 'LanguageHaveWorkedWith' in df.columns:
        df['LanguageCount'] = count_languages(df['LanguageHaveWorkedWith'])
        df = df.drop(columns=['LanguageHaveWorkedWith'])

    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])

    # step 4 - drop rows where ALL features are NaN (edge case)
    df = df.dropna(how='all')

    print(f"Clean data shape: {df.shape}")
    print(f"Missing values per column: \n {df.isna().sum().to_string()} ")

    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
        returns categorical columns and numerical columns from the cleaned df.
        This does not include the target variable.
    """
    non_target = [c for c in df.columns if c != TARGET]
    cat_cols = df[non_target].select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df[non_target].select_dtypes(include=['number']).columns.tolist()

    return cat_cols, num_cols




