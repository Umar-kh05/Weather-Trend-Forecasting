"""
preprocessing.py
Comprehensive data cleaning, outlier treatment, feature engineering, and normalization
for the Global Weather Repository dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes text columns (e.g., stripping whitespace, title-casing condition_text).
    """
    df = df.copy()
    if "condition_text" in df.columns:
        df["condition_text"] = df["condition_text"].astype(str).str.strip().str.title()
    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
    if "location_name" in df.columns:
        df["location_name"] = df["location_name"].astype(str).str.strip()
    if "wind_direction" in df.columns:
        df["wind_direction"] = df["wind_direction"].astype(str).str.strip().str.upper()
    return df


def parse_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts rich temporal and cyclical calendar features from last_updated.
    """
    df = df.copy()
    if "last_updated" in df.columns:
        df["last_updated_dt"] = pd.to_datetime(df["last_updated"])
        df["date"] = df["last_updated_dt"].dt.date
        df["year"] = df["last_updated_dt"].dt.year
        df["month"] = df["last_updated_dt"].dt.month
        df["day"] = df["last_updated_dt"].dt.day
        df["hour"] = df["last_updated_dt"].dt.hour
        df["dayofweek"] = df["last_updated_dt"].dt.dayofweek
        df["dayofyear"] = df["last_updated_dt"].dt.dayofyear
        df["quarter"] = df["last_updated_dt"].dt.quarter
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical temporal encoding (sine & cosine transforms)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

        # Season classification respecting hemisphere
        def get_season(row):
            lat = row.get("latitude", 0)
            m = row["month"]
            if lat >= 0:  # Northern Hemisphere
                if m in [12, 1, 2]:
                    return "Winter"
                elif m in [3, 4, 5]:
                    return "Spring"
                elif m in [6, 7, 8]:
                    return "Summer"
                else:
                    return "Autumn"
            else:  # Southern Hemisphere
                if m in [12, 1, 2]:
                    return "Summer"
                elif m in [3, 4, 5]:
                    return "Autumn"
                elif m in [6, 7, 8]:
                    return "Winter"
                else:
                    return "Spring"

        df["season"] = df.apply(get_season, axis=1)

    return df


def handle_sensor_anomalies(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans unphysical readings and sensor glitches (e.g. negative air quality PM values).
    """
    df = df.copy()
    cleaned_counts = {}

    # Air quality measures cannot be negative
    aq_cols = [
        "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10"
    ]
    for col in aq_cols:
        if col in df.columns:
            neg_mask = df[col] < 0
            count = int(neg_mask.sum())
            cleaned_counts[col + "_negative_cleaned"] = count
            if count > 0:
                # Replace unphysical negative values with NaN then interpolate by location
                df.loc[neg_mask, col] = np.nan
                df[col] = df.groupby("location_name")[col].transform(
                    lambda g: g.interpolate(method="linear").bfill().ffill()
                )
                # If still any NaN, fill with median
                df[col] = df[col].fillna(df[col].median())

    # Bound physical percentages
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    if "cloud" in df.columns:
        df["cloud"] = df["cloud"].clip(lower=0, upper=100)
    if "precip_mm" in df.columns:
        df["precip_mm"] = df["precip_mm"].clip(lower=0)
    if "wind_kph" in df.columns:
        df["wind_kph"] = df["wind_kph"].clip(lower=0)

    return df, cleaned_counts


def assign_climate_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns macro climate zones based on latitude coordinates.
    """
    df = df.copy()
    if "latitude" in df.columns:
        def categorize_zone(lat):
            abs_lat = abs(lat)
            if abs_lat <= 23.5:
                return "Tropical"
            elif abs_lat <= 35.0:
                return "Subtropical"
            elif abs_lat <= 50.0:
                return "Temperate"
            elif abs_lat <= 66.5:
                return "Subpolar/Boreal"
            else:
                return "Polar"

        df["climate_zone"] = df["latitude"].apply(categorize_zone)

        # Continents / Regional grouping estimation from coordinate boundaries
        def estimate_continent(row):
            lat = row.get("latitude", 0)
            lon = row.get("longitude", 0)
            if lat < -10 and lon > 110 and lon < 180:
                return "Oceania"
            elif lat > 35 and lon > -25 and lon < 45:
                return "Europe"
            elif lat > -35 and lat < 38 and lon > -20 and lon < 55:
                return "Africa"
            elif lat > 5 and lon > 55 and lon < 180:
                return "Asia"
            elif lat > 15 and lon > -170 and lon < -50:
                return "North America"
            elif lat <= 15 and lon > -90 and lon < -30:
                return "South America"
            else:
                return "Other/Global"

        df["continent_region"] = df.apply(estimate_continent, axis=1)

    return df


def engineer_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates historical lag and rolling statistics for time-series forecasting.
    Sorts chronologically per location to avoid lookahead bias.
    """
    df = df.copy()
    if "location_name" in df.columns and "last_updated_dt" in df.columns:
        df = df.sort_values(by=["location_name", "last_updated_dt"]).reset_index(drop=True)

        # Groupby location for lagging
        grouped = df.groupby("location_name")

        # Temperature lags
        df["temp_lag_1"] = grouped["temperature_celsius"].shift(1)
        df["temp_lag_2"] = grouped["temperature_celsius"].shift(2)
        df["temp_lag_3"] = grouped["temperature_celsius"].shift(3)
        df["temp_lag_7"] = grouped["temperature_celsius"].shift(7)

        # Rolling statistics (3-day and 7-day windows using closed='left' to prevent lookahead)
        df["temp_roll_mean_3"] = grouped["temperature_celsius"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["temp_roll_mean_7"] = grouped["temperature_celsius"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        df["temp_roll_std_7"] = grouped["temperature_celsius"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0)
        )
        df["temp_roll_min_7"] = grouped["temperature_celsius"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).min()
        )
        df["temp_roll_max_7"] = grouped["temperature_celsius"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).max()
        )

        # Day-over-day temperature momentum
        df["temp_diff_1"] = df["temp_lag_1"] - df["temp_lag_2"]

        # Weather and environmental lags
        df["humidity_lag_1"] = grouped["humidity"].shift(1)
        df["pressure_lag_1"] = grouped["pressure_mb"].shift(1)
        df["wind_lag_1"] = grouped["wind_kph"].shift(1)
        df["precip_lag_1"] = grouped["precip_mm"].shift(1)
        if "air_quality_PM2.5" in df.columns:
            df["pm25_lag_1"] = grouped["air_quality_PM2.5"].shift(1)

        # Backward fill initial lag values with the earliest available observation per group
        lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_diff_" in c]
        for c in lag_cols:
            df[c] = grouped[c].transform(lambda x: x.bfill().ffill())
            # Fill any single-observation location NaNs with current values or median
            if "temp_" in c:
                df[c] = df[c].fillna(df["temperature_celsius"])
            elif "humidity_" in c:
                df[c] = df[c].fillna(df["humidity"])
            elif "pressure_" in c:
                df[c] = df[c].fillna(df["pressure_mb"])
            elif "wind_" in c:
                df[c] = df[c].fillna(df["wind_kph"])
            elif "precip_" in c:
                df[c] = df[c].fillna(df["precip_mm"])
            elif "pm25_" in c and "air_quality_PM2.5" in df.columns:
                df[c] = df[c].fillna(df["air_quality_PM2.5"])
            else:
                df[c] = df[c].fillna(df[c].median() if not df[c].dropna().empty else 0)

    return df


def preprocess_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the entire end-to-end preprocessing pipeline.
    """
    print("[Preprocessing] Starting data cleaning and feature engineering pipeline...")
    
    # 1. Clean text
    df = clean_text_columns(df)
    
    # 2. Parse temporal features
    df = parse_datetime_features(df)
    
    # 3. Clean sensor anomalies
    df, cleaned_counts = handle_sensor_anomalies(df)
    
    # 4. Assign climate zones & continents
    df = assign_climate_zones(df)
    
    # 5. Generate time-series lags & rolling windows
    df = engineer_time_series_features(df)
    
    print(f"[Preprocessing] Completed pipeline. Total columns now: {df.shape[1]}")
    return df, cleaned_counts


if __name__ == "__main__":
    from data_loader import load_raw_data
    raw = load_raw_data()
    clean_df, report = preprocess_pipeline(raw)
    print("Cleaned anomalies:", report)
    print("New engineered columns:", [c for c in clean_df.columns if c not in raw.columns])
