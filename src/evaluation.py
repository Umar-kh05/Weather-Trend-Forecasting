"""
evaluation.py
Comprehensive model evaluation, performance metrics, residual analysis,
and comparative visualizations for Weather Trend Forecasting.
"""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Computes regression & time-series performance metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: avoid divide-by-zero by masking small true values
    nonzero_mask = np.abs(y_true) > 0.5
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    else:
        mape = 0.0

    max_err = float(np.max(np.abs(y_true - y_pred)))

    return {
        "model_name": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE(%)": round(mape, 2),
        "Max_Error": round(max_err, 4)
    }


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Plots a multi-metric comparative bar chart across all evaluated forecasting models.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["MAE", "RMSE", "R2"]
    colors = ["#3B82F6", "#10B981", "#8B5CF6"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        sns.barplot(
            data=results_df, x="model_name", y=metric, ax=ax,
            color=color, alpha=0.85
        )
        ax.set_title(f"Model Comparison: {metric}", weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.3f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=9,
                            xytext=(0, 3), textcoords="offset points")

    plt.suptitle("Weather Trend Forecasting - Model Benchmark Evaluation", weight="bold", fontsize=16, y=1.03)
    fpath = os.path.join(output_dir, "model_metrics_comparison.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Saved: {fpath}")
    return fpath


def plot_residuals(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: str = "outputs/figures") -> str:
    """
    Plots error distributions (residuals = y_true - y_pred) for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    palette = sns.color_palette("muted", n_models)

    for i, (name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        ax = axes[i]
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=ax, color=palette[i], bins=40)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}\nMean Residual: {residuals.mean():.3f}", weight="bold", fontsize=11)
        ax.set_xlabel("Residual (°C)")
        if i == 0:
            ax.set_ylabel("Frequency")

    plt.suptitle("Forecast Residual Distributions (Actual - Predicted)", weight="bold", fontsize=14, y=1.02)
    fpath = os.path.join(output_dir, "model_residuals_distribution.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Saved: {fpath}")
    return fpath


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dates: pd.Series = None,
    output_dir: str = "outputs/figures",
    sample_size: int = 200
) -> str:
    """
    Plots an Actual vs Predicted timeline overlay for a sample of the test period.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    y_t = y_true[:sample_size]
    y_p = y_pred[:sample_size]
    idx = range(len(y_t)) if dates is None else dates[:sample_size]

    ax.plot(idx, y_t, label="Actual Observed Temperature", color="#1F2937", linewidth=2.0)
    ax.plot(idx, y_p, label=f"Forecast ({model_name})", color="#2563EB", linewidth=1.8, linestyle="--")
    ax.fill_between(idx, y_t, y_p, color="#93C5FD", alpha=0.35, label="Prediction Residual Error")

    ax.set_title(f"Weather Forecast Timeline: Actual vs Predicted ({model_name})", weight="bold")
    ax.set_xlabel("Observation Index / Timeline")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")

    filename = f"model_actual_vs_pred_{model_name.lower().replace(' ', '_')}.png"
    fpath = os.path.join(output_dir, filename)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Saved: {fpath}")
    return fpath
