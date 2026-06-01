"""
End-to-end training script for the developer salary prediction model.

v2 improvements vs v1
─────────────────────
1. 10 extra features pulled from raw data (DevType, OrgSize,
   RemoteWork, WorkExp, Industry, Age, ICorPM, DatabaseCount,
   PlatformCount, ToolCountWork).
2. Log-transform of salary → model predicts log(salary+1);
   predictions are expm1-inverted back to USD at eval / serve time.
3. XGBoost hyperparameters tuned: added min_child_weight, gamma,
   reg_alpha, reg_lambda, max_delta_step.
4. Experiment results written to experiments/results.json for
   comparison across runs.

OUTPUTS
───────
1. models/salary_pipeline.pkl  — trained sklearn Pipeline
2. data/cleaned/processed_data.csv
3. data/predictions_plot.png
4. experiments/results.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# add /src to path so relative imports work when running directly
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_and_clean, get_feature_columns, LOG_TARGET
from evaluate import evaluate_model, plot_predictions, print_observations

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/developers-survey-2025.csv"
PROCESSED_DATA = "data/cleaned/processed_data.csv"
MODEL_OUTPUT_PATH = "models/salary_pipeline.pkl"
EXPERIMENTS_DIR = Path("experiments")
RESULTS_PATH = EXPERIMENTS_DIR / "results.json"

# ── config ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,      # reduces over-fitting on noisy salary data
    "gamma": 0.1,               # min loss-reduction before a split
    "reg_alpha": 0.05,          # L1 regularisation
    "reg_lambda": 1.0,          # L2 regularisation
    "max_delta_step": 1,        # helps with log-transformed target
    "tree_method": "hist",      # fast on large datasets
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


# ── pipeline helpers ───────────────────────────────────────────────────────

def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes and encodes features.

    Numeric pipeline: median impute → StandardScaler.
    Categorical pipeline: most-frequent impute → OneHotEncoder.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    """Combine preprocessor + XGBoost model into one sklearn Pipeline."""
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBRegressor(**XGBOOST_PARAMS)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ── experiment logging ─────────────────────────────────────────────────────

def log_experiment(params: dict, train_metrics: dict, test_metrics: dict) -> None:
    """Append this run's config + results to experiments/results.json."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "xgboost_params": params,
        "train": train_metrics,
        "test": test_metrics,
    }

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            history = json.load(f)
    else:
        history = []

    history.append(record)

    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Experiment logged → {RESULTS_PATH}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end training: load → clean → train → evaluate → save."""
    print("=" * 60)
    print("  Developer Salary Prediction — Training (v2)")
    print("=" * 60)

    # 1. Load and clean data
    df = load_and_clean(RAW_DATA_PATH)

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"\nProcessed data saved to: {PROCESSED_DATA}\n")

    # 2. Feature / target split
    X = df.drop(columns=[LOG_TARGET])
    y = df[LOG_TARGET]          # log1p-transformed salary

    cat_cols, num_cols = get_feature_columns(df)
    print(f"Numeric features  : {num_cols}")
    print(f"Categorical features: {cat_cols}\n")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}\n")

    # 4. Build and train pipeline
    print("Building pipeline ...")
    pipeline = build_pipeline(cat_cols, num_cols)

    print("Training XGBoost model ...")
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # 5. Evaluate — convert log-predictions back to USD for interpretability
    y_pred_log_train = pipeline.predict(X_train)
    y_pred_log_test = pipeline.predict(X_test)

    # Metrics on log scale (what the model optimises)
    train_metrics_log = evaluate_model(
        y_train, y_pred_log_train, title="Training set (log scale)"
    )
    test_metrics_log = evaluate_model(
        y_test, y_pred_log_test, title="Test set (log scale)"
    )

    # Metrics on original USD scale (human-readable)
    y_train_usd = np.expm1(y_train)
    y_test_usd = np.expm1(y_test)
    y_pred_train_usd = np.expm1(y_pred_log_train)
    y_pred_test_usd = np.expm1(y_pred_log_test)

    train_metrics_usd = evaluate_model(
        y_train_usd, y_pred_train_usd, title="Training set (USD)"
    )
    test_metrics_usd = evaluate_model(
        y_test_usd, y_pred_test_usd, title="Test set (USD) ← main metric"
    )

    print_observations(test_metrics_usd)

    # 6. Plot (USD scale is more intuitive)
    plot_predictions(
        y_test_usd.values,
        y_pred_test_usd,
        save_path="data/predictions_plot.png",
    )

    # 7. Log experiment
    log_experiment(
        params=XGBOOST_PARAMS,
        train_metrics=train_metrics_usd,
        test_metrics=test_metrics_usd,
    )

    # 8. Save model pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")

    # 9. Sample prediction (log-scale internally, display in USD)
    print("\nSample prediction:\n")
    sample = pd.DataFrame([{
        "Country": "Ukraine",
        "YearsCode": 10.0,
        "WorkExp": 8.0,
        "EdLevel": "Bachelor's",
        "Employment": "Full-time",
        "DevType": "Full-stack",
        "OrgSize": 3,
        "RemoteWork": "Remote",
        "Industry": "Information Services, IT, Software Development, or other Technology",
        "Age": 2,
        "ICorPM": 0,
        "LanguageCount": 4,
        "DatabaseCount": 3,
        "PlatformCount": 2,
        "ToolCountWork": 5,
    }])

    log_pred = pipeline.predict(sample)[0]
    usd_pred = np.expm1(log_pred)
    mae_usd = test_metrics_usd["mae"]

    print(f"input : {sample.to_dict(orient='records')[0]}")
    print(f"Predicted salary: ${usd_pred:,.0f}  ±  ${mae_usd:,.0f}")
    print("\nTraining script complete.")


if __name__ == "__main__":
    main()
