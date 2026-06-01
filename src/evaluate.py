"""
Evaluation utilities for the salary prediction model.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, title: str = "Model evaluation") -> dict:
    """
    Compute and print regression metrics.

    Returns a dict with keys: mae, rmse, r2, mape.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error — guard against zero denominators
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    print(f"\n{'─' * 45}")
    print(f"  {title}")
    print(f"{'─' * 45}")
    print(f"  MAE   : {mae:>12,.2f}")
    print(f"  RMSE  : {rmse:>12,.2f}")
    print(f"  R²    : {r2:>12.4f}")
    print(f"  MAPE  : {mape:>11.2f}%")
    print(f"{'─' * 45}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def plot_predictions(y_true, y_pred, save_path: str | None = None) -> None:
    """
    Plot actual vs predicted and residual distribution side-by-side.
    Both arrays should already be in the original (USD) scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── plot 1: actual vs predicted ───────────────────────────────────────
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lim_max = max(y_true.max(), y_pred.max())
    axes[0].plot([0, lim_max], [0, lim_max], "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual salary (USD)")
    axes[0].set_ylabel("Predicted salary (USD)")
    axes[0].set_title("Actual vs Predicted Salary")
    axes[0].legend()

    # ── plot 2: residual distribution ────────────────────────────────────
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted) USD")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.suptitle("Model Evaluation", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.close(fig)


def print_observations(metrics: dict) -> None:
    """Print human-readable interpretation of evaluation metrics."""
    mae = metrics["mae"]
    r2 = metrics["r2"]
    mape = metrics.get("mape", None)

    print("\nObservations")
    print(f"  • MAE of ${mae:,.0f} means the average prediction is off by ${mae:,.0f} from the true salary.")

    if mape is not None:
        print(f"  • MAPE of {mape:.1f}% shows the relative error across all salary levels.")

    if r2 > 0.7:
        print(f"  • R² of {r2:.3f} is strong — the model explains {r2 * 100:.1f}% of salary variance.")
    elif r2 > 0.5:
        print(f"  • R² of {r2:.3f} is moderate — there is unexplained variance (expected for salary data).")
    else:
        print(f"  • R² of {r2:.3f} is low — salary data has many unmeasured drivers (role, company, etc.).")
