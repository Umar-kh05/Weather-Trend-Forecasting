"""
anomaly_detection.py
Advanced Anomaly Detection and Outlier Analysis for Weather Trend Forecasting.
Combines statistical thresholds (IQR, Z-Score), domain extreme events,
and unsupervised Machine Learning (Isolation Forest) across multivariate weather features.
"""

import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def ensure_output_dirs(fig_dir: str = "outputs/figures", metric_dir: str = "outputs/metrics"):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)


def detect_statistical_outliers(df: pd.DataFrame, column: str, z_thresh: float = 3.0) -> pd.Series:
    """
    Identifies outliers using standard Z-score method.
    """
    col_clean = df[column].dropna()
    mean = col_clean.mean()
    std = col_clean.std()
    if std == 0:
        return pd.Series(False, index=df.index)
    z_scores = (df[column] - mean).abs() / std
    return z_scores > z_thresh


def detect_iqr_outliers(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    Identifies outliers using Interquartile Range (IQR) method.
    """
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def fit_isolation_forest(
    df: pd.DataFrame,
    features: list = None,
    contamination: float = 0.02,
    random_state: int = 42
) -> Tuple[pd.DataFrame, IsolationForest]:
    """
    Fits an unsupervised Isolation Forest model to detect multivariate anomalies.
    """
    if features is None:
        features = [
            "temperature_celsius", "humidity", "pressure_mb",
            "wind_kph", "precip_mm", "air_quality_PM2.5"
        ]
    
    valid_features = [f for f in features if f in df.columns]
    df_clean = df.copy()

    # Fill any remaining NaNs with column median
    X = df_clean[valid_features].fillna(df_clean[valid_features].median())

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    
    # -1 for anomaly, 1 for inlier
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X)

    df_clean["is_anomaly_iso"] = (preds == -1).astype(int)
    df_clean["anomaly_score"] = scores
    
    return df_clean, iso_forest


def categorize_meteorological_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes domain-specific meteorological extreme events.
    """
    df = df.copy()
    
    # 1. Extreme Heat (>40°C)
    df["extreme_heat"] = (df["temperature_celsius"] >= 40.0).astype(int)
    
    # 2. Extreme Cold (<-10°C)
    df["extreme_cold"] = (df["temperature_celsius"] <= -10.0).astype(int)
    
    # 3. Severe Precipitation (>99th percentile)
    precip_p99 = df["precip_mm"].quantile(0.99)
    df["extreme_precip"] = (df["precip_mm"] >= max(5.0, precip_p99)).astype(int)
    
    # 4. Severe Wind (>99th percentile)
    wind_p99 = df["wind_kph"].quantile(0.99)
    df["extreme_wind"] = (df["wind_kph"] >= wind_p99).astype(int)
    
    # 5. Hazardous Air Quality (PM2.5 > 150 µg/m³ or EPA index >= 5)
    if "air_quality_PM2.5" in df.columns:
        df["hazardous_air"] = (
            (df["air_quality_PM2.5"] >= 150.0) | 
            (df.get("air_quality_us-epa-index", 0) >= 5)
        ).astype(int)
    else:
        df["hazardous_air"] = 0
        
    df["any_extreme_event"] = (
        df["extreme_heat"] | df["extreme_cold"] | 
        df["extreme_precip"] | df["extreme_wind"] | df["hazardous_air"]
    ).astype(int)

    return df


def plot_anomaly_visualizations(
    df: pd.DataFrame,
    fig_dir: str = "outputs/figures"
) -> Dict[str, str]:
    """
    Generates high-resolution visualizations for detected anomalies.
    """
    ensure_output_dirs(fig_dir=fig_dir)
    plots = {}

    # 1. Multivariate Scatter: Temperature vs Humidity colored by Isolation Forest Anomaly
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = df[df["is_anomaly_iso"] == 0]
    anomalies = df[df["is_anomaly_iso"] == 1]

    ax.scatter(normal["temperature_celsius"], normal["humidity"], c="#3B82F6", alpha=0.3, s=15, label="Normal Observations")
    ax.scatter(anomalies["temperature_celsius"], anomalies["humidity"], c="#EF4444", alpha=0.8, s=40, edgecolors="black", linewidths=0.5, label=f"Isolation Forest Anomalies (n={len(anomalies):,})")
    ax.set_title("Multivariate Weather Anomalies (Isolation Forest)", weight="bold")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.legend(loc="upper left")

    fpath1 = os.path.join(fig_dir, "anomaly_isolation_forest_scatter.png")
    fig.savefig(fpath1, dpi=300)
    plt.close(fig)
    plots["scatter"] = fpath1
    print(f"[Anomaly] Saved: {fpath1}")

    # 2. Timeline of Anomalies across 500+ days
    if "date" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        daily_anoms = df.groupby("date")["is_anomaly_iso"].sum().reset_index()
        daily_anoms["date"] = pd.to_datetime(daily_anoms["date"])
        daily_anoms = daily_anoms.sort_values("date")
        
        ax.plot(daily_anoms["date"], daily_anoms["is_anomaly_iso"], color="#DC2626", linewidth=1.5, marker="o", markersize=3, alpha=0.7)
        ax.fill_between(daily_anoms["date"], daily_anoms["is_anomaly_iso"], color="#FCA5A5", alpha=0.4)
        ax.set_title("Global Frequency of Weather Anomalies Over Time", weight="bold")
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Daily Anomaly Count")

        fpath2 = os.path.join(fig_dir, "anomaly_timeline.png")
        fig.savefig(fpath2, dpi=300)
        plt.close(fig)
        plots["timeline"] = fpath2
        print(f"[Anomaly] Saved: {fpath2}")

    # 3. Top Locations with Most Anomalous Events
    fig, ax = plt.subplots(figsize=(12, 6))
    top_locs = df[df["is_anomaly_iso"] == 1]["location_name"].value_counts().head(15)
    sns.barplot(x=top_locs.values, y=top_locs.index, ax=ax, palette="Reds_r", hue=top_locs.index, legend=False)
    ax.set_title("Top 15 Global Locations with Most Detected Anomalies", weight="bold")
    ax.set_xlabel("Count of Anomalous Weather Days")
    ax.set_ylabel("City / Location")

    fpath3 = os.path.join(fig_dir, "anomaly_top_locations.png")
    fig.savefig(fpath3, dpi=300)
    plt.close(fig)
    plots["top_locations"] = fpath3
    print(f"[Anomaly] Saved: {fpath3}")

    return plots


def run_full_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.02,
    fig_dir: str = "outputs/figures",
    metric_dir: str = "outputs/metrics"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes complete anomaly detection pipeline and exports metrics.
    """
    print("[Anomaly] Initiating Multi-Method Anomaly Detection Pipeline...")
    ensure_output_dirs(fig_dir, metric_dir)

    # 1. Isolation Forest
    df_anom, model = fit_isolation_forest(df, contamination=contamination)

    # 2. Meteorological Extremes
    df_anom = categorize_meteorological_extremes(df_anom)

    # 3. Statistical Z-Score & IQR for temperature
    df_anom["temp_zscore_outlier"] = detect_statistical_outliers(df_anom, "temperature_celsius")
    df_anom["temp_iqr_outlier"] = detect_iqr_outliers(df_anom, "temperature_celsius")

    # 4. Visualizations
    plots = plot_anomaly_visualizations(df_anom, fig_dir=fig_dir)

    # 5. Export summary
    metrics = {
        "total_records": int(len(df_anom)),
        "iso_forest_anomalies": int(df_anom["is_anomaly_iso"].sum()),
        "iso_forest_anomaly_pct": round(float(df_anom["is_anomaly_iso"].mean() * 100), 2),
        "extreme_heat_days": int(df_anom["extreme_heat"].sum()),
        "extreme_cold_days": int(df_anom["extreme_cold"].sum()),
        "extreme_precip_days": int(df_anom["extreme_precip"].sum()),
        "extreme_wind_days": int(df_anom["extreme_wind"].sum()),
        "hazardous_air_days": int(df_anom["hazardous_air"].sum()),
        "any_extreme_event_days": int(df_anom["any_extreme_event"].sum()),
    }

    # Save anomalies dataframe sample
    anom_sample = df_anom[df_anom["is_anomaly_iso"] == 1][
        ["country", "location_name", "last_updated", "temperature_celsius", 
         "humidity", "wind_kph", "precip_mm", "air_quality_PM2.5", "anomaly_score"]
    ].sort_values("anomaly_score").head(100)
    anom_csv_path = os.path.join(metric_dir, "detected_anomalies_top100.csv")
    anom_sample.to_csv(anom_csv_path, index=False)
    print(f"[Anomaly] Exported top anomalies table to: {anom_csv_path}")

    return df_anom, {"metrics": metrics, "plots": plots}


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_pipeline

    raw = load_raw_data()
    clean_df, _ = preprocess_pipeline(raw)
    anom_df, results = run_full_anomaly_detection(clean_df)
    print("\n[Anomaly] Results Summary:")
    for k, v in results["metrics"].items():
        print(f"  - {k}: {v}")
