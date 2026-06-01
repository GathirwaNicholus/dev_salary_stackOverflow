'''
    Evaluation utilities for the salary prediction model
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             r2_score)

def evaluate_model(y_true, y_pred, title: str = "Model evaluation") -> dict:
    """
        Compute and print regression metrics.

        Returns a dict of metric name -> value
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{title}")
    print(f"MAE, Mean Absolute Error: ${mae}")
    print(f"RMSE, Root Mean Squared Error: ${rmse}")
    print(f"R2 score (Co-efficient of det.) {r2}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictions(y_true, y_pred, save_path: str= None):
    """
        Plot actual vs predicted.

        Plot for residuals.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot 1 - actual vs predicted.
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    limit = max(y_true.max(), y_pred.max()) * 105
    axes[0].plot([0, limit], [0, limit], 'r--', linewidth=1.5, label='perfect prediction')

    axes[0].set_xlabel('Actul salary (USD)')
    axes[0].set_ylabel('Predicted salary (USD)')
    axes[0].set_title("Actual vs Predicted salary")
    axes[0].legend()

    # plot 2 - residuals.
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color='coral', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual distribution')

    plt.suptitle('Model evaluation', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

def print_observations(metrics: dict):
    """
        Print observations based on model metrics.
    """
    mae = metrics['mae']
    r2 = metrics['r2']

    print("Observations\n")

    print(f"MAE of ${mae} means our model's average prediction is off by ${mae:,.0f} from the true salary")

    if r2 > 0.7:
        print(f"r2 score of {r2:.3f} is strong - the model explains {r2 * 100}% of the variance in salary")
    elif r2 > 0.5:
        print(f"r2 of {r2:.3f} is moderate - there is still variance the model cannot capture (expe"
              "cted for salary data)")
    else:
        print(f"r2 of {r2:.3f} is relatively low. This is common for salary predictions as many "
              "factors are unmeasured")
        
    

    




