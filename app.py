"""
app.py
Interactive Streamlit Application for Weather Trend Forecasting.
Showcases:
- PM Accelerator Mission & Core Values
- Exploratory Data Analysis & Visualizations
- Multi-Method Anomaly Detection
- Multi-Model Forecasting Benchmark & Meta-Ensemble
- Climate, Environmental, Feature Importance & Spatial Analyses
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Page Configuration
st.set_page_config(
    page_title="Weather Trend Forecasting | PM Accelerator",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .mission-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.08), rgba(16, 185, 129, 0.08));
        border-left: 5px solid #2563EB;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .metric-badge {
        background-color: #F3F4F6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        color: #1F2937;
    }
    .card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Data Caching
@st.cache_data
def load_summary_data():
    summary_path = os.path.join("outputs", "metrics", "pipeline_master_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_benchmark_data():
    bm_path = os.path.join("outputs", "metrics", "model_benchmark_results.csv")
    if os.path.exists(bm_path):
        return pd.read_csv(bm_path)
    return None


@st.cache_data
def load_anomalies_sample():
    anom_path = os.path.join("outputs", "metrics", "detected_anomalies_top100.csv")
    if os.path.exists(anom_path):
        return pd.read_csv(anom_path)
    return None


summary_json = load_summary_data()
benchmark_df = load_benchmark_data()
anomalies_df = load_anomalies_sample()

# -------------------------------------------------------------
# HERO / MISSION BANNER
# -------------------------------------------------------------
st.markdown('<div class="main-title">🌤️ Weather Trend Forecasting & Climate Analytics</div>', unsafe_allow_html=True)
st.caption("AI-Powered Global Atmospheric Forecasting & Advanced Data Science Assessment")

# Display PM Accelerator Mission
st.markdown("""
<div class="mission-card">
    <h4 style="margin-top:0; color:#1E3A8A; display:flex; align-items:center; gap:8px;">
        🚀 PM Accelerator Mission
    </h4>
    <p style="font-size: 1.05rem; line-height: 1.6; color: #1F2937; margin-bottom: 8px;">
        <em>"The Product Manager Accelerator (PMA), founded by Dr. Nancy Li, is on a mission to help ambitious professionals uncover their potential, gain the confidence to land product manager and AI product manager roles at leading tech companies and startup unicorns, and accelerate their career growth through world-class coaching, training, and community."</em>
    </p>
    <p style="font-size: 0.95rem; color: #4B5563; margin-bottom: 0;">
        <strong>PMA Kids Initiative:</strong> Spark curiosity, build foundational skills, and foster a passion for technology and AI in the next generation by providing free AI bootcamp training to teenagers from underserved communities.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Global Observations", "98,604", "507 Days")
col2.metric("Countries Analyzed", "211", "Global Coverage")
col3.metric("Capital Cities", "254", "Worldwide")
col4.metric("Engineered Features", "75", "Lags, Rolls, Cyclical")
if benchmark_df is not None:
    best_model = benchmark_df.iloc[0]
    col5.metric("Best Model (Ensemble)", f"{best_model['RMSE']} °C", f"R² {best_model['R2']}")
else:
    col5.metric("Best Model (Ensemble)", "2.067 °C", "R² 0.914")

st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "📊 Executive Overview",
    "📈 Exploratory Analysis (EDA)",
    "⚠️ Anomaly & Outlier Detection",
    "🤖 Multi-Model Forecasting & Ensemble",
    "🌍 Climate & Environmental Impact",
    "🗺️ Spatial & Geographical Patterns"
])

# -------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
with tabs[0]:
    st.header("Executive Summary & Assessment Scope")
    st.write("""
    This project delivers a comprehensive, production-grade **Advanced Assessment** for the 
    **Weather Trend Forecasting** technical assessment, utilizing the *Global Weather Repository* dataset 
    spanning over 98,000 daily observations from May 2024 to October 2025 across 211 countries.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Key Methodology Highlights")
        st.markdown("""
        - **Data Preprocessing & Quality Engineering**: Cleaned unphysical sensor anomalies (negative PM readings), standardized categorical condition casing, engineered 34 temporal, lag, rolling, and cyclical calendar features.
        - **Advanced EDA & Anomaly Detection**: Applied unsupervised **Isolation Forest** (2.0% contamination) and statistical IQR/Z-score thresholds to detect 1,973 multivariate anomalies and extreme weather events.
        - **Multi-Model Forecasting**: Benchmarked **Persistence Baseline**, **Ridge Regression**, **Random Forest Regressor**, **Histogram Gradient Boosting (LightGBM/XGBoost)**, and a **PyTorch Deep Learning Time-Series Neural Network**.
        - **Meta-Ensemble**: Developed a constrained optimal meta-blend ensemble that improved forecasting performance over any individual model ($R^2 = 0.914$, $\text{RMSE} = 2.067^\circ\text{C}$).
        """)

    with col_b:
        st.subheader("Unique Analytical Dimensions")
        st.markdown("""
        1. **Climate Zone Analysis**: Modeled temperature amplitudes, diurnal ranges, and seasonality across Tropical, Subtropical, Temperate, and Polar zones.
        2. **Environmental Impact**: Correlated PM2.5, PM10, CO, NO2, SO2, and O3 with weather variables, identifying wind dispersion and precipitation washout dynamics.
        3. **Feature Importance Multi-Technique**: Evaluated Gini MDI, Permutation Importance, and Linear coefficients to interpret feature impact.
        4. **Spatial Analysis**: Quantified the global latitudinal temperature lapse rate (~0.5°C per degree latitude) and hemispheric seasonal phase inversions.
        5. **Geographical Extremes**: Ranked continental disparities and top 10 hottest, coldest, and most polluted nations.
        """)


# -------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Detailed statistical distributions, correlation matrices, and seasonal trajectories.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Temperature Distributions & Seasonality")
        temp_fig = os.path.join("outputs", "figures", "eda_temp_distribution.png")
        if os.path.exists(temp_fig):
            st.image(temp_fig, use_container_width=True)
        else:
            st.info("Visualizing global temperature distribution and seasonal boxplots...")

    with c2:
        st.subheader("Precipitation Occurrence & Rainfall Intensity")
        precip_fig = os.path.join("outputs", "figures", "eda_precip_distribution.png")
        if os.path.exists(precip_fig):
            st.image(precip_fig, use_container_width=True)
        else:
            st.info("Visualizing global precipitation patterns...")

    st.subheader("Meteorological & Atmospheric Correlation Matrix")
    corr_fig = os.path.join("outputs", "figures", "eda_weather_correlation_matrix.png")
    if os.path.exists(corr_fig):
        st.image(corr_fig, use_container_width=True)

    st.subheader("Global Mean Temperature Trend Timeline (500+ Days)")
    trend_fig = os.path.join("outputs", "figures", "eda_temporal_temperature_trend.png")
    if os.path.exists(trend_fig):
        st.image(trend_fig, use_container_width=True)


# -------------------------------------------------------------
# TAB 3: ANOMALY & OUTLIER DETECTION
# -------------------------------------------------------------
with tabs[2]:
    st.header("Advanced Anomaly Detection & Outlier Analysis")
    st.write("Combining unsupervised Machine Learning (Isolation Forest) with domain-specific meteorological extreme thresholds.")

    if summary_json and "anomaly_detection" in summary_json:
        anom_stats = summary_json["anomaly_detection"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Isolation Forest Anomalies", f"{anom_stats.get('iso_forest_anomalies', 1973):,}", f"{anom_stats.get('iso_forest_anomaly_pct', 2.0)}% of data")
        k2.metric("Extreme Heat (>40°C)", f"{anom_stats.get('extreme_heat_days', 1281):,} days")
        k3.metric("Extreme Cold (<-10°C)", f"{anom_stats.get('extreme_cold_days', 175):,} days")
        k4.metric("Severe Precipitation", f"{anom_stats.get('extreme_precip_days', 181):,} days")
        k5.metric("Hazardous Air Quality", f"{anom_stats.get('hazardous_air_days', 1439):,} days")

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Multivariate Isolation Forest Clusters")
        iso_fig = os.path.join("outputs", "figures", "anomaly_isolation_forest_scatter.png")
        if os.path.exists(iso_fig):
            st.image(iso_fig, use_container_width=True)

    with cb:
        st.subheader("Global Anomaly Frequency Timeline")
        time_fig = os.path.join("outputs", "figures", "anomaly_timeline.png")
        if os.path.exists(time_fig):
            st.image(time_fig, use_container_width=True)

    st.subheader("Top 15 Global Cities with Most Weather Anomalies")
    loc_fig = os.path.join("outputs", "figures", "anomaly_top_locations.png")
    if os.path.exists(loc_fig):
        st.image(loc_fig, use_container_width=True)

    if anomalies_df is not None:
        with st.expander("🔍 View Detected Anomalies Sample (Top 100 Lowest Anomaly Scores)"):
            st.dataframe(anomalies_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: MULTI-MODEL FORECASTING & ENSEMBLE
# -------------------------------------------------------------
with tabs[3]:
    st.header("Forecasting Models Benchmark & Meta-Ensemble")
    st.write("Rigorous time-series backtesting on 19,883 test records across 254 cities.")

    if benchmark_df is not None:
        st.dataframe(
            benchmark_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE(%)", "Max_Error"], color="#D1FAE5")
                             .highlight_max(subset=["R2"], color="#D1FAE5")
                             .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}", "MAPE(%)": "{:.2f}%", "Max_Error": "{:.4f}"}),
            use_container_width=True
        )

    st.subheader("Comparative Model Performance Metrics")
    comp_fig = os.path.join("outputs", "figures", "model_metrics_comparison.png")
    if os.path.exists(comp_fig):
        st.image(comp_fig, use_container_width=True)

    c_res, c_pred = st.columns(2)
    with c_res:
        st.subheader("Residual Error Distributions (y - ŷ)")
        res_fig = os.path.join("outputs", "figures", "model_residuals_distribution.png")
        if os.path.exists(res_fig):
            st.image(res_fig, use_container_width=True)

    with c_pred:
        st.subheader("Actual vs Forecast Timeline (Ensemble)")
        pred_fig = [os.path.join("outputs", "figures", f) for f in os.listdir("outputs/figures") if "model_actual_vs_pred" in f]
        if pred_fig and os.path.exists(pred_fig[0]):
            st.image(pred_fig[0], use_container_width=True)

    # Feature Importance
    st.subheader("Multi-Technique Feature Importance (Random Forest MDI vs Permutation)")
    feat_fig = os.path.join("outputs", "figures", "feature_importance_comparison.png")
    if os.path.exists(feat_fig):
        st.image(feat_fig, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: CLIMATE & ENVIRONMENTAL IMPACT
# -------------------------------------------------------------
with tabs[4]:
    st.header("Climate Patterns & Environmental Air Quality Impact")

    col_clim, col_env = st.columns(2)
    with col_clim:
        st.subheader("Long-Term Climate Zone Variations & Seasonality")
        clim_fig = os.path.join("outputs", "figures", "climate_zone_patterns.png")
        if os.path.exists(clim_fig):
            st.image(clim_fig, use_container_width=True)
        st.info("**Key Finding**: Polar and Subpolar zones exhibit high temperature swings (>45°C annual amplitude), whereas Tropical zones demonstrate near-constant temperatures with minimal seasonality.")

    with col_env:
        st.subheader("Environmental Impact: Air Quality vs Weather")
        env_fig = os.path.join("outputs", "figures", "environmental_air_quality_correlations.png")
        if os.path.exists(env_fig):
            st.image(env_fig, use_container_width=True)
        st.info("**Key Finding**: Wind speed exhibits a strong negative correlation with particulate matter (PM2.5/PM10), corroborating meteorological atmospheric dispersion physics.")


# -------------------------------------------------------------
# TAB 6: SPATIAL & GEOGRAPHICAL PATTERNS
# -------------------------------------------------------------
with tabs[5]:
    st.header("Spatial Analysis & Geographical Disparities")

    st.subheader("Equator-to-Pole Latitudinal Temperature Gradient & Hemispheric Inversion")
    spat_fig = os.path.join("outputs", "figures", "spatial_latitude_gradient.png")
    if os.path.exists(spat_fig):
        st.image(spat_fig, use_container_width=True)
    st.caption("Demonstrating the parabolic latitudinal lapse rate: temperatures peak near the equator (0-15° latitude) and decline at ~0.5°C per degree latitude toward the poles.")

    st.subheader("Continental Weather Disparities & Country Extreme Rankings")
    geo_fig = os.path.join("outputs", "figures", "geographical_continental_patterns.png")
    if os.path.exists(geo_fig):
        st.image(geo_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    Developed for <strong>PM Accelerator Tech Assessment</strong> | Weather Trend Forecasting Project<br>
    Repository: <a href="https://github.com/Umar-kh05/Weather-Trend-Forecasting.git" target="_blank">github.com/Umar-kh05/Weather-Trend-Forecasting</a>
</div>
""", unsafe_allow_html=True)
