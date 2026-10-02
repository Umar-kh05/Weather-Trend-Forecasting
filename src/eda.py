"""
eda.py
Exploratory Data Analysis (EDA) module for Weather Trend Forecasting.
Generates comprehensive statistical summaries, correlation analyses,
and publication-quality visualizations for temperature, precipitation, and weather patterns.
"""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})


def ensure_output_dirs(output_dir: str = "outputs/figures") -> str:
    """Ensures figure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical profiles including skewness, kurtosis, and quartiles.
    """
    numeric_cols = [
        "temperature_celsius", "feels_like_celsius", "precip_mm", "humidity",
        "wind_kph", "pressure_mb", "cloud", "visibility_km", "uv_index",
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
        "air_quality_Ozone", "air_quality_Nitrogen_dioxide"
    ]
    valid_cols = [c for c in numeric_cols if c in df.columns]
    
    summary = df[valid_cols].describe().T
    summary["skew"] = df[valid_cols].skew()
    summary["kurtosis"] = df[valid_cols].kurt()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary


def plot_temperature_distributions(df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generates temperature distribution and seasonal KDE plots.
    """
    ensure_output_dirs(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Global temperature histogram & KDE
    sns.histplot(df["temperature_celsius"], kde=True, ax=axes[0], color="#2563EB", bins=45)
    mean_temp = df["temperature_celsius"].mean()
    median_temp = df["temperature_celsius"].median()
    axes[0].axvline(mean_temp, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {mean_temp:.1f}°C")
    axes[0].axvline(median_temp, color="#10B981", linestyle="-.", linewidth=1.8, label=f"Median: {median_temp:.1f}°C")
    axes[0].set_title("Global Temperature Distribution (°C)", weight="bold")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # 2. Temperature by Season
    if "season" in df.columns:
        sns.boxplot(
            data=df, x="season", y="temperature_celsius", hue="season", ax=axes[1],
            palette=["#38BDF8", "#34D399", "#F59E0B", "#F87171"], legend=False
        )
        axes[1].set_title("Temperature Variations by Season", weight="bold")
        axes[1].set_xlabel("Season")
        axes[1].set_ylabel("Temperature (°C)")

    filepath = os.path.join(output_dir, "eda_temp_distribution.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved: {filepath}")
    return filepath


def plot_precipitation_patterns(df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Visualizes global precipitation patterns, dry vs wet days, and rainfall intensities.
    """
    ensure_output_dirs(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Dry vs Wet Days proportion
    dry_threshold = 0.1  # mm
    dry_count = (df["precip_mm"] <= dry_threshold).sum()
    wet_count = (df["precip_mm"] > dry_threshold).sum()

    labels = ["Dry Days (<=0.1mm)", "Precipitation (>0.1mm)"]
    sizes = [dry_count, wet_count]
    colors = ["#E0F2FE", "#0284C7"]
    axes[0].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0, 0.08))
    axes[0].set_title("Precipitation Occurrence Worldwide", weight="bold")

    # 2. Distribution of non-zero precipitation (Log Scale)
    wet_days = df[df["precip_mm"] > 0]["precip_mm"]
    sns.histplot(wet_days, bins=35, kde=True, ax=axes[1], color="#0D9488", log_scale=True)
    axes[1].set_title("Rainfall Intensity on Wet Days (Log Scale mm)", weight="bold")
    axes[1].set_xlabel("Precipitation (mm, Log Scale)")
    axes[1].set_ylabel("Frequency")

    filepath = os.path.join(output_dir, "eda_precip_distribution.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved: {filepath}")
    return filepath


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generates a correlation heatmap between primary meteorological and atmospheric variables.
    """
    ensure_output_dirs(output_dir)
    features = [
        "temperature_celsius", "feels_like_celsius", "humidity", "precip_mm",
        "pressure_mb", "wind_kph", "cloud", "visibility_km", "uv_index",
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
        "air_quality_Ozone", "air_quality_Nitrogen_dioxide"
    ]
    valid_cols = [c for c in features if c in df.columns]
    corr = df[valid_cols].corr()

    # Shorten names for cleaner rendering
    rename_dict = {
        "temperature_celsius": "Temp (°C)",
        "feels_like_celsius": "Feels Like (°C)",
        "humidity": "Humidity (%)",
        "precip_mm": "Precip (mm)",
        "pressure_mb": "Pressure (mb)",
        "wind_kph": "Wind (kph)",
        "cloud": "Cloud (%)",
        "visibility_km": "Visibility (km)",
        "uv_index": "UV Index",
        "air_quality_PM2.5": "PM2.5",
        "air_quality_PM10": "PM10",
        "air_quality_Carbon_Monoxide": "CO",
        "air_quality_Ozone": "O3",
        "air_quality_Nitrogen_dioxide": "NO2"
    }
    corr = corr.rename(index=rename_dict, columns=rename_dict)

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Meteorological & Atmospheric Correlation Matrix", weight="bold", pad=15)

    filepath = os.path.join(output_dir, "eda_weather_correlation_matrix.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved: {filepath}")
    return filepath


def plot_temporal_trends(df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Plots the global daily mean temperature timeline with rolling moving average.
    """
    ensure_output_dirs(output_dir)
    if "date" not in df.columns:
        return ""

    daily_trend = df.groupby("date").agg(
        mean_temp=("temperature_celsius", "mean"),
        std_temp=("temperature_celsius", "std"),
        mean_precip=("precip_mm", "mean")
    ).reset_index()
    daily_trend["date"] = pd.to_datetime(daily_trend["date"])
    daily_trend = daily_trend.sort_values("date")
    daily_trend["roll_7"] = daily_trend["mean_temp"].rolling(7, min_periods=1).mean()
    daily_trend["roll_30"] = daily_trend["mean_temp"].rolling(30, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Daily temp line & rolling trends
    ax1.plot(daily_trend["date"], daily_trend["mean_temp"], color="#93C5FD", alpha=0.6, label="Daily Global Mean Temp")
    ax1.plot(daily_trend["date"], daily_trend["roll_7"], color="#2563EB", linewidth=1.8, label="7-Day Rolling Mean")
    ax1.plot(daily_trend["date"], daily_trend["roll_30"], color="#DC2626", linewidth=2.2, label="30-Day Rolling Mean")
    ax1.set_ylabel("Global Mean Temperature (°C)", color="#1E3A8A")
    ax1.set_title("Global Mean Temperature Trend Across 500+ Days", weight="bold")
    ax1.set_xlabel("Timeline")
    ax1.legend(loc="upper left")

    # Secondary axis for precipitation
    ax2 = ax1.twinx()
    ax2.bar(daily_trend["date"], daily_trend["mean_precip"], color="#0D9488", alpha=0.3, width=1.0, label="Daily Mean Precip (mm)")
    ax2.set_ylabel("Global Mean Precipitation (mm)", color="#0D9488")
    ax2.grid(False)

    filepath = os.path.join(output_dir, "eda_temporal_temperature_trend.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved: {filepath}")
    return filepath


def run_full_eda(df: pd.DataFrame, output_dir: str = "outputs/figures") -> Dict:
    """
    Executes the full EDA suite and returns generated artifacts.
    """
    print("[EDA] Executing Exploratory Data Analysis suite...")
    summary = compute_statistical_summary(df)
    
    fig_temp = plot_temperature_distributions(df, output_dir)
    fig_precip = plot_precipitation_patterns(df, output_dir)
    fig_corr = plot_correlation_matrix(df, output_dir)
    fig_trend = plot_temporal_trends(df, output_dir)
    
    return {
        "summary_table": summary,
        "figures": {
            "temperature_dist": fig_temp,
            "precipitation_patterns": fig_precip,
            "correlation_matrix": fig_corr,
            "temporal_trends": fig_trend
        }
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_pipeline
    
    raw = load_raw_data()
    clean_df, _ = preprocess_pipeline(raw)
    eda_results = run_full_eda(clean_df)
    print("[EDA] Statistical Summary Head:\n", eda_results["summary_table"].head())
