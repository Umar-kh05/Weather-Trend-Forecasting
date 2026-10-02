"""
advanced_analyses.py
Comprehensive implementation of all 5 Unique Analyses for the Advanced Assessment:
1. Climate Analysis: Long-term patterns, climate zones, seasonal swings.
2. Environmental Impact: Air quality correlations with weather parameters.
3. Feature Importance: Multi-technique feature assessment (Tree MDI, Permutation, Linear).
4. Spatial Analysis: Latitude gradients, hemispheric patterns, geographical clustering.
5. Geographical Patterns: Continental disparities and country-level weather rankings.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def ensure_output_dirs(fig_dir: str = "outputs/figures", metric_dir: str = "outputs/metrics"):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)


# -------------------------------------------------------------
# 1. CLIMATE ANALYSIS
# -------------------------------------------------------------
def run_climate_analysis(df: pd.DataFrame, fig_dir: str = "outputs/figures") -> Dict:
    """
    Studies long-term climate patterns and variations across global climate zones.
    """
    ensure_output_dirs(fig_dir)
    print("[Advanced Analyses] 1/5 Running Climate Analysis...")

    zone_stats = df.groupby("climate_zone").agg(
        mean_temp=("temperature_celsius", "mean"),
        std_temp=("temperature_celsius", "std"),
        min_temp=("temperature_celsius", "min"),
        max_temp=("temperature_celsius", "max"),
        mean_precip=("precip_mm", "mean"),
        mean_humidity=("humidity", "mean"),
        total_obs=("temperature_celsius", "count")
    ).reset_index()

    # Temperature amplitude (seasonal swing)
    zone_stats["temp_amplitude"] = zone_stats["max_temp"] - zone_stats["min_temp"]

    # 1. Visualization: Boxplot across climate zones
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order = ["Tropical", "Subtropical", "Temperate", "Subpolar/Boreal", "Polar"]
    valid_order = [z for z in order if z in df["climate_zone"].unique()]

    sns.boxplot(
        data=df, x="climate_zone", y="temperature_celsius", order=valid_order,
        ax=axes[0], palette="Spectral", hue="climate_zone", legend=False
    )
    axes[0].set_title("Temperature Variations Across Climate Zones", weight="bold")
    axes[0].set_xlabel("Climate Zone")
    axes[0].set_ylabel("Temperature (°C)")

    # 2. Seasonality curves over the year
    df_plot = df.copy()
    if "month" in df_plot.columns:
        monthly_zone = df_plot.groupby(["climate_zone", "month"])["temperature_celsius"].mean().reset_index()
        for zone in valid_order:
            z_data = monthly_zone[monthly_zone["climate_zone"] == zone]
            if not z_data.empty:
                axes[1].plot(z_data["month"], z_data["temperature_celsius"], marker="o", label=zone, linewidth=2)
        axes[1].set_title("Monthly Seasonality Trajectories by Climate Zone", weight="bold")
        axes[1].set_xlabel("Month of Year")
        axes[1].set_ylabel("Mean Temperature (°C)")
        axes[1].set_xticks(range(1, 13))
        axes[1].legend()

    fpath = os.path.join(fig_dir, "climate_zone_patterns.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {fpath}")

    return {"zone_stats": zone_stats, "plot": fpath}


# -------------------------------------------------------------
# 2. ENVIRONMENTAL IMPACT (AIR QUALITY)
# -------------------------------------------------------------
def run_environmental_impact_analysis(df: pd.DataFrame, fig_dir: str = "outputs/figures") -> Dict:
    """
    Analyzes air quality parameters and their correlation with meteorological variables.
    """
    ensure_output_dirs(fig_dir)
    print("[Advanced Analyses] 2/5 Running Environmental Impact Analysis...")

    pollutants = [
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
        "air_quality_Ozone", "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide"
    ]
    weather_vars = ["temperature_celsius", "wind_kph", "humidity", "precip_mm", "pressure_mb"]
    
    valid_pollutants = [p for p in pollutants if p in df.columns]
    valid_weather = [w for w in weather_vars if w in df.columns]

    # Compute correlation between pollutants and weather parameters
    corr_matrix = df[valid_pollutants + valid_weather].corr().loc[valid_pollutants, valid_weather]

    clean_names_p = {p: p.replace("air_quality_", "") for p in valid_pollutants}
    clean_names_w = {
        "temperature_celsius": "Temp", "wind_kph": "Wind Speed",
        "humidity": "Humidity", "precip_mm": "Precipitation", "pressure_mb": "Pressure"
    }
    corr_display = corr_matrix.rename(index=clean_names_p, columns=clean_names_w)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Correlation Heatmap
    sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="vlag", vmin=-0.5, vmax=0.5, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Environmental Impact: Air Quality vs Weather Parameters", weight="bold")
    axes[0].set_ylabel("Air Quality Pollutant")

    # 2. Key interactions: Wind Dispersion Effect on PM2.5
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df, x="wind_kph", y="air_quality_PM2.5",
        hue="temperature_celsius", palette="viridis", alpha=0.6, ax=axes[1]
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Pollutant Dispersion: Wind Speed vs PM2.5 (Log Scale)", weight="bold")
    axes[1].set_xlabel("Wind Speed (kph)")
    axes[1].set_ylabel("PM2.5 Concentration (µg/m³, Log Scale)")

    fpath = os.path.join(fig_dir, "environmental_air_quality_correlations.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {fpath}")

    return {"correlation_matrix": corr_display, "plot": fpath}


# -------------------------------------------------------------
# 3. FEATURE IMPORTANCE ANALYSIS
# -------------------------------------------------------------
def run_feature_importance_analysis(
    rf_model: RandomForestRegressor,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    fig_dir: str = "outputs/figures"
) -> Dict:
    """
    Applies multiple techniques to assess feature importance:
    1. Random Forest Mean Decrease in Impurity (MDI)
    2. Permutation Importance
    """
    ensure_output_dirs(fig_dir)
    print("[Advanced Analyses] 3/5 Running Feature Importance Analysis...")

    # 1. Tree MDI Importance
    mdi_importance = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    # 2. Permutation Importance on a test subset
    sample_idx = np.random.choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    perm_result = permutation_importance(
        rf_model, X_test[sample_idx], y_test[sample_idx],
        n_repeats=5, random_state=42, n_jobs=-1
    )
    perm_importance = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(ascending=False)

    # Combine top 10
    top_features = list(dict.fromkeys(list(mdi_importance.head(10).index) + list(perm_importance.head(10).index)))[:10]

    comp_df = pd.DataFrame({
        "Feature": top_features,
        "Tree MDI Importance": [mdi_importance.get(f, 0) for f in top_features],
        "Permutation Importance": [perm_importance.get(f, 0) for f in top_features]
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # MDI Barplot
    sns.barplot(data=comp_df.sort_values("Tree MDI Importance", ascending=False),
                x="Tree MDI Importance", y="Feature", ax=axes[0], palette="Blues_r", hue="Feature", legend=False)
    axes[0].set_title("Random Forest Feature Importance (MDI / Gini)", weight="bold")
    axes[0].set_xlabel("Relative Importance")

    # Permutation Barplot
    sns.barplot(data=comp_df.sort_values("Permutation Importance", ascending=False),
                x="Permutation Importance", y="Feature", ax=axes[1], palette="Greens_r", hue="Feature", legend=False)
    axes[1].set_title("Permutation Feature Importance (Test Set)", weight="bold")
    axes[1].set_xlabel("Mean Score Decrease (R2 Impact)")
    axes[1].set_ylabel("")

    fpath = os.path.join(fig_dir, "feature_importance_comparison.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {fpath}")

    return {"mdi_importance": mdi_importance, "permutation_importance": perm_importance, "plot": fpath}


# -------------------------------------------------------------
# 4. SPATIAL ANALYSIS
# -------------------------------------------------------------
def run_spatial_analysis(df: pd.DataFrame, fig_dir: str = "outputs/figures") -> Dict:
    """
    Analyzes and visualizes geographical and latitudinal patterns.
    """
    ensure_output_dirs(fig_dir)
    print("[Advanced Analyses] 4/5 Running Spatial Analysis...")

    location_means = df.groupby(["location_name", "country", "latitude", "longitude"]).agg(
        avg_temp=("temperature_celsius", "mean"),
        avg_precip=("precip_mm", "mean"),
        avg_pm25=("air_quality_PM2.5", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # 1. Latitude vs Mean Temperature (Equator-to-Pole Lapse Rate)
    sns.regplot(
        data=location_means, x="latitude", y="avg_temp",
        scatter_kws={"alpha": 0.6, "color": "#0284C7", "s": 40},
        line_kws={"color": "#DC2626", "linewidth": 2.5},
        order=2, ax=axes[0]
    )
    axes[0].axvline(0, color="gray", linestyle=":", label="Equator (0° Lat)")
    axes[0].set_title("Latitudinal Temperature Gradient Worldwide", weight="bold")
    axes[0].set_xlabel("Latitude (°)")
    axes[0].set_ylabel("Mean Temperature (°C)")
    axes[0].legend()

    # 2. Hemispheric Seasonal Inversion
    if "month" in df.columns:
        df["hemisphere"] = np.where(df["latitude"] >= 0, "Northern Hemisphere", "Southern Hemisphere")
        hemi_monthly = df.groupby(["hemisphere", "month"])["temperature_celsius"].mean().reset_index()
        sns.lineplot(
            data=hemi_monthly, x="month", y="temperature_celsius", hue="hemisphere",
            palette=["#2563EB", "#F59E0B"], marker="o", linewidth=2.5, ax=axes[1]
        )
        axes[1].set_title("Hemispheric Seasonal Inversion (Phase Shift)", weight="bold")
        axes[1].set_xlabel("Month of Year")
        axes[1].set_ylabel("Mean Temperature (°C)")
        axes[1].set_xticks(range(1, 13))

    fpath = os.path.join(fig_dir, "spatial_latitude_gradient.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {fpath}")

    return {"location_means": location_means, "plot": fpath}


# -------------------------------------------------------------
# 5. GEOGRAPHICAL PATTERNS (CONTINENTS & COUNTRIES)
# -------------------------------------------------------------
def run_geographical_patterns_analysis(df: pd.DataFrame, fig_dir: str = "outputs/figures") -> Dict:
    """
    Explores how weather conditions differ across countries and continents.
    """
    ensure_output_dirs(fig_dir)
    print("[Advanced Analyses] 5/5 Running Geographical Patterns Analysis...")

    # Continental summary
    continent_stats = df.groupby("continent_region").agg(
        avg_temp=("temperature_celsius", "mean"),
        max_temp=("temperature_celsius", "max"),
        min_temp=("temperature_celsius", "min"),
        avg_precip=("precip_mm", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_wind=("wind_kph", "mean"),
        avg_pm25=("air_quality_PM2.5", "mean")
    ).reset_index().sort_values("avg_temp", ascending=False)

    # Country rankings
    country_stats = df.groupby("country").agg(
        avg_temp=("temperature_celsius", "mean"),
        avg_precip=("precip_mm", "mean"),
        avg_wind=("wind_kph", "mean"),
        avg_pm25=("air_quality_PM2.5", "mean")
    ).reset_index()

    top_hottest = country_stats.sort_values("avg_temp", ascending=False).head(10)
    top_coldest = country_stats.sort_values("avg_temp", ascending=True).head(10)
    top_polluted = country_stats.sort_values("avg_pm25", ascending=False).head(10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. Continental Temperature Profiles
    sns.barplot(
        data=continent_stats, x="continent_region", y="avg_temp",
        ax=axes[0], palette="coolwarm", hue="continent_region", legend=False
    )
    axes[0].set_title("Average Temperature by Continent/Region", weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean Temperature (°C)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")

    # 2. Top 10 Hottest Countries
    sns.barplot(
        data=top_hottest, x="avg_temp", y="country",
        ax=axes[1], palette="YlOrRd_r", hue="country", legend=False
    )
    axes[1].set_title("Top 10 Hottest Countries (Annual Mean)", weight="bold")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("")

    # 3. Top 10 Highest PM2.5 Pollution
    sns.barplot(
        data=top_polluted, x="avg_pm25", y="country",
        ax=axes[2], palette="Purples_r", hue="country", legend=False
    )
    axes[2].set_title("Top 10 Most Polluted Countries (PM2.5)", weight="bold")
    axes[2].set_xlabel("Mean PM2.5 (µg/m³)")
    axes[2].set_ylabel("")

    fpath = os.path.join(fig_dir, "geographical_continental_patterns.png")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {fpath}")

    return {
        "continent_stats": continent_stats,
        "top_hottest": top_hottest,
        "top_coldest": top_coldest,
        "top_polluted": top_polluted,
        "plot": fpath
    }


def run_all_advanced_analyses(
    df: pd.DataFrame,
    rf_model: RandomForestRegressor,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    fig_dir: str = "outputs/figures",
    metric_dir: str = "outputs/metrics"
) -> Dict:
    """
    Orchestrates all 5 advanced analyses and exports metrics.
    """
    ensure_output_dirs(fig_dir, metric_dir)

    r_climate = run_climate_analysis(df, fig_dir)
    r_env = run_environmental_impact_analysis(df, fig_dir)
    r_feat = run_feature_importance_analysis(rf_model, feature_names, X_test, y_test, fig_dir)
    r_spatial = run_spatial_analysis(df, fig_dir)
    r_geo = run_geographical_patterns_analysis(df, fig_dir)

    # Export key tables
    r_climate["zone_stats"].to_csv(os.path.join(metric_dir, "climate_zone_statistics.csv"), index=False)
    r_env["correlation_matrix"].to_csv(os.path.join(metric_dir, "environmental_correlation_matrix.csv"))
    r_geo["continent_stats"].to_csv(os.path.join(metric_dir, "continent_weather_statistics.csv"), index=False)

    return {
        "climate": r_climate,
        "environmental": r_env,
        "feature_importance": r_feat,
        "spatial": r_spatial,
        "geographical": r_geo
    }
