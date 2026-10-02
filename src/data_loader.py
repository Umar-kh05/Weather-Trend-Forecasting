"""
data_loader.py
Module for ingesting and validating the Global Weather Repository dataset.
Handles file verification, reading with proper encoding, and initial schema summary.
"""

import os
from typing import Optional, Tuple
import pandas as pd


def get_dataset_path(custom_path: Optional[str] = None) -> str:
    """
    Locates the dataset file with fallback searching.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        os.path.join("data", "GlobalWeatherRepository.csv"),
        "GlobalWeatherRepository.csv",
        os.path.join("..", "data", "GlobalWeatherRepository.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "GlobalWeatherRepository.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(
        "Could not find GlobalWeatherRepository.csv in default locations. "
        "Please provide the path directly or place it in data/"
    )


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw CSV data and performs initial validation.
    
    Args:
        filepath: Optional path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    path = get_dataset_path(filepath)
    print(f"[DataLoader] Loading dataset from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[DataLoader] Successfully loaded {len(df):,} records across {df.shape[1]} columns.")
    return df


def get_dataset_metadata(df: pd.DataFrame) -> dict:
    """
    Generates summary metadata for the dataset.
    """
    return {
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "unique_countries": int(df["country"].nunique()) if "country" in df.columns else 0,
        "unique_locations": int(df["location_name"].nunique()) if "location_name" in df.columns else 0,
        "date_min": str(df["last_updated"].min()) if "last_updated" in df.columns else None,
        "date_max": str(df["last_updated"].max()) if "last_updated" in df.columns else None,
        "null_count": int(df.isnull().sum().sum())
    }


if __name__ == "__main__":
    data = load_raw_data()
    meta = get_dataset_metadata(data)
    for k, v in meta.items():
        print(f"  - {k}: {v}")
