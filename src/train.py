"""
End-to-end training script for the developer salary prediction model.

v4 improvements (from v3 — CV R²=0.56):
────────────────────────────────────────
1. STACKING ENSEMBLE: XGBoost + GradientBoosting → Ridge meta-learner
   → two diverse tree models learn slightly different patterns; the
     Ridge combiner finds optimal weights
2. NEW FEATURES: has_high_pay_lang (Go/Rust/Scala/...) + DevTypeCount
3. RELAXED REGULARISATION: deeper trees + lower min_child_weight to let
   the model capture more complex salary patterns
4. EARLY STOPPING still used to find optimal tree count for XGBoost;
   GradientBoosting uses a fixed sensible count

OUTPUTS
───────
1. models/salary_pipeline.pkl
2. data/cleaned/processed_data.csv
3. data/predictions_plot.png
4. experiments/results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
EARLY_STOP_VAL_SIZE = 0.15

# v4: slightly relaxed regularisation to let the model capture more signal
XGBOOST_PARAMS: dict = {
    "n_estimators": 3000,       # high — early stopping picks the best
    "max_depth": 7,             # was 6 — captures deeper feature interactions
    "learning_rate": 0.02,      # was 0.03 — lower LR with more trees
    "subsample": 0.75,
    "colsample_bytree": 0.6,    # was 0.65
    "min_child_weight": 5,      # was 8 — less restrictive
    "gamma": 0.1,               # was 0.15
    "reg_alpha": 0.05,          # was 0.08
    "reg_lambda": 1.0,          # was 1.5
    "max_delta_step": 1,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

GBR_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
}


# ── pipeline helpers ───────────────────────────────────────────────────────

def build_preprocessor(
    target_enc_cols: list[str],
    num_cols: list[str],
) -> ColumnTransformer:
    """
    Numeric → median-impute → standard-scale
    Categorical → TargetEncoder (mean-salary per category, CV-protected)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

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
    xgb_n_estimators: int | None = None,
    use_stacking: bool = True,
) -> Pipeline:
    """
    Combine preprocessor + model into one sklearn Pipeline.

    If use_stacking=True, the model is a StackingRegressor that blends
    XGBoost + GradientBoosting through a Ridge meta-learner.
    """
    preprocessor = build_preprocessor(target_enc_cols, num_cols)

    xgb_params = XGBOOST_PARAMS.copy()
    if xgb_n_estimators is not None:
        xgb_params["n_estimators"] = xgb_n_estimators

    xgb = XGBRegressor(**xgb_params)

    if use_stacking:
        gbr_params = GBR_PARAMS.copy()
        gbr = GradientBoostingRegressor(**gbr_params)

        model = StackingRegressor(
            estimators=[("xgb", xgb), ("gbr", gbr)],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
        )
    else:
        model = xgb

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
    optimal number of XGBoost boosting rounds.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=EARLY_STOP_VAL_SIZE,
        random_state=RANDOM_STATE,
    )

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

    best = model.best_iteration + 1
    print(f"  Early stopping: best iteration = {best} / {params['n_estimators']}")
    return best


# ── experiment logging ─────────────────────────────────────────────────────

def log_experiment(
    params: dict,
    train_metrics: dict,
    test_metrics: dict,
    cv_r2_mean: float | None = None,
    cv_r2_std: float | None = None,
    n_features: int | None = None,
    use_stacking: bool = False,
) -> None:
    """Append run config + results to experiments/results.json."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "version": "v4",
        "use_stacking": use_stacking,
        "n_features": n_features,
        "xgboost_params": params,
        "train": train_metrics,
        "test": test_metrics,
    }
    if cv_r2_mean is not None:
        record["cv_r2_mean"] = cv_r2_mean
        record["cv_r2_std"] = cv_r2_std

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
    """End-to-end: load → clean → early-stop → stack → cross-validate → save."""
    t0 = time.time()

    print("=" * 60)
    print("  Developer Salary Prediction — Training  (v4)")
    print("  Model: XGBoost + GradientBoosting  →  Ridge (stacking)")
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

    # ── 4. Find optimal XGBoost tree count ────────────────────────────────
    print("Step 1/3  Finding optimal XGBoost tree count (early stopping) ...")
    best_n = find_best_n_estimators(X_train, y_train, target_enc_cols, num_cols)

    # ── 5. Train stacking pipeline on full training data ──────────────────
    print(f"\nStep 2/3  Training stacking ensemble "
          f"(XGBoost[{best_n}] + GBR[{GBR_PARAMS['n_estimators']}] → Ridge) ...")
    print("         This may take a few minutes ...")

    pipeline = build_pipeline(
        target_enc_cols, num_cols,
        xgb_n_estimators=best_n,
        use_stacking=True,
    )
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    y_pred_log_train = pipeline.predict(X_train)
    y_pred_log_test = pipeline.predict(X_test)

    # Log scale (model's native metric — this is what CV reports)
    train_metrics_log = evaluate_model(
        y_train, y_pred_log_train, title="Training set (log scale)"
    )
    test_metrics_log = evaluate_model(
        y_test, y_pred_log_test, title="Test set (log scale) ← model's native metric"
    )

    # USD scale (human-interpretable)
    y_train_usd = np.expm1(y_train)
    y_test_usd = np.expm1(y_test)
    y_pred_train_usd = np.expm1(y_pred_log_train)
    y_pred_test_usd = np.expm1(y_pred_log_test)

    train_metrics_usd = evaluate_model(
        y_train_usd, y_pred_train_usd, title="Training set (USD)"
    )
    test_metrics_usd = evaluate_model(
        y_test_usd, y_pred_test_usd, title="Test set (USD)"
    )
    print_observations(test_metrics_usd)

    # ── 7. 5-fold cross-validation (log scale — reproducibility proof) ────
    print("\nStep 3/3  5-fold cross-validation (log scale, may take 5-10 min) ...")
    cv_pipeline = build_pipeline(
        target_enc_cols, num_cols,
        xgb_n_estimators=best_n,
        use_stacking=True,
    )
    cv_scores = cross_val_score(
        cv_pipeline, X, y,
        cv=5,
        scoring="r2",
    )
    print(f"  CV R² scores : {cv_scores.round(4)}")
    print(f"  Mean CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. Plot (USD) ─────────────────────────────────────────────────────
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
        train_metrics=test_metrics_log,   # log the LOG-scale metrics
        test_metrics=test_metrics_log,
        cv_r2_mean=float(cv_scores.mean()),
        cv_r2_std=float(cv_scores.std()),
        n_features=X.shape[1],
        use_stacking=True,
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
        "EdLevel": 4,
        "Employment": 1,
        "DevType": "Full-stack",
        "OrgSize": 4,
        "RemoteWork": 2,
        "Industry": "Information Services, IT, Software Development, or other Technology",
        "Age": 2,
        "ICorPM": 0,
        "LanguageCount": 4,
        "DatabaseCount": 3,
        "PlatformCount": 2,
        "ToolCountWork": 5,
        "DevTypeCount": 2,
        "has_high_pay_lang": 1,
    }])
    sample = add_interaction_features(sample)

    log_pred = pipeline.predict(sample)[0]
    usd_pred = np.expm1(log_pred)
    mae = test_metrics_usd["mae"]

    print(f"  Predicted salary : ${usd_pred:,.0f}  ±  ${mae:,.0f}")

    elapsed = time.time() - t0
    print(f"\nTraining script complete. ✓  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
