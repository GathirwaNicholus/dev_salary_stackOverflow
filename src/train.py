"""
End-to-end training script for the developer salary prediction model.

v3 improvements (from v2 baseline R²=0.48):
───────────────────────────────────────────
1. TARGET ENCODING for Country, DevType, Industry
   → replaces 24+ sparse OHE columns with 3 dense, signal-rich features
2. ORDINAL ENCODING for EdLevel & RemoteWork (preserves natural ordering)
3. EMPLOYMENT FILTER — drop students/part-timers/NaN (noisy for salary)
4. INTERACTION FEATURES — YearsCode², WorkExp², experience ratio, tech breadth
5. EARLY STOPPING — fit XGBoost on a train/val split to find optimal tree count
   then retrain full pipeline with that count
6. 5-FOLD CROSS-VALIDATION — proves the R² is reproducible

OUTPUTS
───────
1. models/salary_pipeline.pkl  – trained pipeline
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBRegressor

# add src/ to path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import (
    LOG_TARGET,
    TARGET_ENC_FEATURES,
    add_interaction_features,
    get_feature_columns,
    load_and_clean,
)
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
EARLY_STOP_VAL_SIZE = 0.15  # 15% of training set used for early stopping

XGBOOST_PARAMS: dict = {
    "n_estimators": 2000,       # set high – early stopping picks the best
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "min_child_weight": 8,
    "gamma": 0.15,
    "reg_alpha": 0.08,
    "reg_lambda": 1.5,
    "max_delta_step": 1,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


# ── pipeline helpers ───────────────────────────────────────────────────────

def build_preprocessor(
    target_enc_cols: list[str],
    num_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with two parallel branches:

    1. Numeric → median-impute → standard-scale
    2. Categorical → sklearn TargetEncoder (learns mean-target per category,
       uses internal cross-fitting to prevent data leakage)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # TargetEncoder handles NaN internally (maps to global target mean)
    target_enc = TargetEncoder(
        smooth="auto",
        target_type="continuous",
        random_state=RANDOM_STATE,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("target_enc", target_enc, target_enc_cols),
        ],
        remainder="drop",
    )


def build_pipeline(
    target_enc_cols: list[str],
    num_cols: list[str],
    n_estimators: int | None = None,
) -> Pipeline:
    """Combine preprocessor + XGBoost into one sklearn Pipeline."""
    preprocessor = build_preprocessor(target_enc_cols, num_cols)

    params = XGBOOST_PARAMS.copy()
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    model = XGBRegressor(**params)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def find_best_n_estimators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_enc_cols: list[str],
    num_cols: list[str],
) -> int:
    """
    Use a held-out validation set + early stopping to find the
    optimal number of boosting rounds.  This prevents overfitting
    and closes the train/test R² gap.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=EARLY_STOP_VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    # Fit preprocessor on the training subset
    preprocessor = build_preprocessor(target_enc_cols, num_cols)
    X_tr_processed = preprocessor.fit_transform(X_tr, y_tr)
    X_val_processed = preprocessor.transform(X_val)

    params = XGBOOST_PARAMS.copy()
    params["early_stopping_rounds"] = 50

    model = XGBRegressor(**params)
    model.fit(
        X_tr_processed, y_tr,
        eval_set=[(X_val_processed, y_val)],
        verbose=False,
    )

    best = model.best_iteration + 1   # 0-indexed → count
    print(f"  Early stopping: best iteration = {best} / {params['n_estimators']}")
    return best


# ── experiment logging ─────────────────────────────────────────────────────

def log_experiment(
    params: dict,
    train_metrics: dict,
    test_metrics: dict,
    cv_r2: float | None = None,
    n_features: int | None = None,
) -> None:
    """Append run config + results to experiments/results.json."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "version": "v3",
        "n_features": n_features,
        "xgboost_params": params,
        "train": train_metrics,
        "test": test_metrics,
    }
    if cv_r2 is not None:
        record["cv_r2_mean"] = cv_r2

    history: list = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            history = json.load(f)

    history.append(record)

    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Experiment logged → {RESULTS_PATH}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end: load → clean → early-stop → train → cross-validate → save."""
    print("=" * 60)
    print("  Developer Salary Prediction — Training  (v3)")
    print("=" * 60)

    # ── 1. Load and clean data ────────────────────────────────────────────
    df = load_and_clean(RAW_DATA_PATH)

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"\nProcessed data saved to: {PROCESSED_DATA}\n")

    # ── 2. Feature / target split ─────────────────────────────────────────
    X = df.drop(columns=[LOG_TARGET])
    y = df[LOG_TARGET]

    target_enc_cols, num_cols = get_feature_columns(df)
    print(f"Target-encoded features : {target_enc_cols}")
    print(f"Numeric features ({len(num_cols)}): {num_cols}\n")

    # ── 3. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}\n")

    # ── 4. Find optimal n_estimators via early stopping ───────────────────
    print("Step 1/3  Finding optimal tree count (early stopping) ...")
    best_n = find_best_n_estimators(X_train, y_train, target_enc_cols, num_cols)

    # ── 5. Train final pipeline on full training data ─────────────────────
    print(f"\nStep 2/3  Training final pipeline with {best_n} trees ...")
    pipeline = build_pipeline(target_enc_cols, num_cols, n_estimators=best_n)
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    y_pred_log_train = pipeline.predict(X_train)
    y_pred_log_test = pipeline.predict(X_test)

    # Log scale (what the model optimises)
    evaluate_model(y_train, y_pred_log_train, title="Training set (log scale)")
    evaluate_model(y_test, y_pred_log_test, title="Test set (log scale)")

    # USD scale (human-readable)
    y_train_usd = np.expm1(y_train)
    y_test_usd = np.expm1(y_test)
    y_pred_train_usd = np.expm1(y_pred_log_train)
    y_pred_test_usd = np.expm1(y_pred_log_test)

    train_metrics = evaluate_model(
        y_train_usd, y_pred_train_usd, title="Training set (USD)"
    )
    test_metrics = evaluate_model(
        y_test_usd, y_pred_test_usd, title="Test set (USD) ← main metric"
    )
    print_observations(test_metrics)

    # ── 7. 5-fold cross-validation (proves reproducibility) ───────────────
    print("\nStep 3/3  5-fold cross-validation (this may take a minute) ...")
    cv_pipeline = build_pipeline(target_enc_cols, num_cols, n_estimators=best_n)
    cv_scores = cross_val_score(
        cv_pipeline, X, y,
        cv=5,
        scoring="r2",
    )
    print(f"  CV R² scores : {cv_scores.round(4)}")
    print(f"  Mean CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. Plot (USD scale) ───────────────────────────────────────────────
    plot_predictions(
        y_test_usd.values,
        y_pred_test_usd,
        save_path="data/predictions_plot.png",
    )

    # ── 9. Log experiment ─────────────────────────────────────────────────
    final_params = XGBOOST_PARAMS.copy()
    final_params["n_estimators"] = best_n
    log_experiment(
        params=final_params,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_r2=float(cv_scores.mean()),
        n_features=X.shape[1],
    )

    # ── 10. Save pipeline ─────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")

    # ── 11. Sample prediction ─────────────────────────────────────────────
    print("\nSample prediction:\n")
    sample = pd.DataFrame([{
        "Country": "United States of America",
        "YearsCode": 10.0,
        "WorkExp": 8.0,
        "EdLevel": 4,            # Bachelor's
        "Employment": 1,         # Full-time
        "DevType": "Full-stack",
        "OrgSize": 4,            # 100-499 employees
        "RemoteWork": 2,         # Remote
        "Industry": "Information Services, IT, Software Development, or other Technology",
        "Age": 2,                # 25-34
        "ICorPM": 0,             # Individual contributor
        "LanguageCount": 4,
        "DatabaseCount": 3,
        "PlatformCount": 2,
        "ToolCountWork": 5,
    }])
    sample = add_interaction_features(sample)

    log_pred = pipeline.predict(sample)[0]
    usd_pred = np.expm1(log_pred)
    mae = test_metrics["mae"]

    print(f"  Input : {sample.iloc[0].to_dict()}")
    print(f"  Predicted salary : ${usd_pred:,.0f}  ±  ${mae:,.0f}")
    print("\nTraining script complete. ✓")


if __name__ == "__main__":
    main()
