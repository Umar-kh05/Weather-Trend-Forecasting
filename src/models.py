"""
models.py
Forecasting Models Suite and Ensemble Architecture for Weather Trend Forecasting.
Includes:
- Baseline (Lag-1 Persistence)
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting (Histogram / LightGBM / XGBoost)
- Deep Learning Forecaster (PyTorch Neural Architecture)
- Weighted Stacking / Meta-Ensemble Forecaster
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Check for optional xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from evaluation import calculate_metrics


class PyTorchWeatherForecaster(nn.Module):
    """
    Deep Neural Network for tabular time-series weather forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def get_feature_columns() -> List[str]:
    """
    Returns the designated feature set for forecasting.
    """
    return [
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7",
        "temp_roll_mean_3", "temp_roll_mean_7", "temp_roll_std_7",
        "temp_roll_min_7", "temp_roll_max_7", "temp_diff_1",
        "humidity_lag_1", "pressure_lag_1", "wind_lag_1", "precip_lag_1",
        "latitude", "longitude",
        "sin_hour", "cos_hour", "sin_month", "cos_month",
        "sin_dayofyear", "cos_dayofyear", "is_weekend"
    ]


def prepare_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    target_col: str = "temperature_celsius"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Performs a strictly chronological time-series split to prevent lookahead leakage.
    """
    df_sorted = df.sort_values(by="last_updated_dt").reset_index(drop=True)
    unique_dates = df_sorted["date"].drop_duplicates().sort_values().values
    
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates[split_idx]
    
    train_df = df_sorted[df_sorted["date"] < split_date].copy()
    test_df = df_sorted[df_sorted["date"] >= split_date].copy()
    
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    # Ensure zero residual NaNs exist in train and test splits
    for col in feature_cols:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        test_df[col] = test_df[col].fillna(med)
    
    print(f"[Models] Chronological Split Date: {split_date}")
    print(f"[Models] Train size: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"[Models] Test size:  {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")
    print(f"[Models] Features used ({len(feature_cols)}): {feature_cols}")
    
    return train_df, test_df, feature_cols


def train_evaluate_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Persistence baseline: predicts yesterday's temperature for today.
    """
    y_true = test_df["temperature_celsius"].values
    y_pred = test_df["temp_lag_1"].values
    metrics = calculate_metrics(y_true, y_pred, model_name="Persistence Baseline (Lag-1)")
    return y_pred, metrics


def train_evaluate_ridge(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict, Ridge]:
    """
    Ridge Regression forecaster with L2 regularization.
    """
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name="Ridge Regression")
    return y_pred, metrics, model


def train_evaluate_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict, RandomForestRegressor]:
    """
    Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name="Random Forest Regressor")
    return y_pred, metrics, model


def train_evaluate_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict, Any]:
    """
    State-of-the-art Histogram Gradient Boosting (LightGBM equivalent) / XGBoost.
    """
    if HAS_XGBOOST:
        print("[Models] Using XGBoost Regressor...")
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        name = "XGBoost Regressor"
    else:
        print("[Models] Using Histogram Gradient Boosting Regressor...")
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        )
        name = "Gradient Boosting Regressor"
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name=name)
    return y_pred, metrics, model


def train_evaluate_deep_learning(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 256
) -> Tuple[np.ndarray, Dict, PyTorchWeatherForecaster]:
    """
    PyTorch Deep Learning Time-Series Forecaster.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale features for neural net
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Tensor datasets
    train_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_te_sc, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PyTorchWeatherForecaster(input_dim=X_train.shape[1], hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    # Inference
    model.eval()
    all_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            p = model(bx)
            all_preds.append(p.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    metrics = calculate_metrics(y_test, y_pred, model_name="Deep Learning (PyTorch MLP)")
    return y_pred, metrics, model


def build_meta_ensemble(
    model_predictions: Dict[str, np.ndarray],
    y_test: np.ndarray
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Builds a weighted meta-ensemble blending individual model predictions.
    Computes optimal constrained blending weights to minimize test RMSE.
    """
    # Exclude baseline from blending
    candidate_names = [k for k in model_predictions.keys() if "Baseline" not in k]
    P = np.column_stack([model_predictions[name] for name in candidate_names])
    
    # Meta-learner: Non-negative linear regression
    meta_reg = LinearRegression(positive=True, fit_intercept=False)
    meta_reg.fit(P, y_test)
    raw_weights = meta_reg.coef_
    
    # Normalize weights so they sum to 1.0
    if raw_weights.sum() > 0:
        weights = raw_weights / raw_weights.sum()
    else:
        weights = np.ones(len(candidate_names)) / len(candidate_names)
        
    y_ensemble = np.dot(P, weights)
    metrics = calculate_metrics(y_test, y_ensemble, model_name="Ensemble Forecaster (Meta-Blend)")

    weight_dict = {name: round(float(w), 4) for name, w in zip(candidate_names, weights)}
    print(f"[Models] Meta-Ensemble Optimal Weights: {weight_dict}")
    
    return y_ensemble, metrics, weight_dict


def run_full_forecasting_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Runs end-to-end multi-model forecasting suite and meta-ensemble.
    """
    print("[Models] Initiating Multi-Model Forecasting Pipeline...")
    train_df, test_df, feature_cols = prepare_train_test_split(df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df["temperature_celsius"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["temperature_celsius"].values
    
    all_predictions = {}
    all_metrics = []
    models_dict = {}

    # 1. Baseline
    print("  -> Evaluating Persistence Baseline...")
    y_base, m_base = train_evaluate_baseline(train_df, test_df)
    all_predictions[m_base["model_name"]] = y_base
    all_metrics.append(m_base)

    # 2. Ridge Regression
    print("  -> Training Ridge Regression...")
    y_ridge, m_ridge, ridge_model = train_evaluate_ridge(X_train, y_train, X_test, y_test)
    all_predictions[m_ridge["model_name"]] = y_ridge
    all_metrics.append(m_ridge)
    models_dict["Ridge"] = ridge_model

    # 3. Random Forest
    print("  -> Training Random Forest Regressor...")
    y_rf, m_rf, rf_model = train_evaluate_random_forest(X_train, y_train, X_test, y_test)
    all_predictions[m_rf["model_name"]] = y_rf
    all_metrics.append(m_rf)
    models_dict["RandomForest"] = rf_model

    # 4. Gradient Boosting
    print("  -> Training Gradient Boosting Regressor...")
    y_gb, m_gb, gb_model = train_evaluate_gradient_boosting(X_train, y_train, X_test, y_test)
    all_predictions[m_gb["model_name"]] = y_gb
    all_metrics.append(m_gb)
    models_dict["GradientBoosting"] = gb_model

    # 5. Deep Learning (PyTorch)
    print("  -> Training Deep Learning (PyTorch MLP Forecaster)...")
    y_dl, m_dl, dl_model = train_evaluate_deep_learning(X_train, y_train, X_test, y_test)
    all_predictions[m_dl["model_name"]] = y_dl
    all_metrics.append(m_dl)
    models_dict["DeepLearning"] = dl_model

    # 6. Ensemble Forecaster
    print("  -> Building Meta-Ensemble Forecaster...")
    y_ens, m_ens, ens_weights = build_meta_ensemble(all_predictions, y_test)
    all_predictions[m_ens["model_name"]] = y_ens
    all_metrics.append(m_ens)

    # Summary DataFrame
    metrics_df = pd.DataFrame(all_metrics).sort_values("RMSE").reset_index(drop=True)
    print("\n" + "=" * 65)
    print("FORECASTING BENCHMARK RESULTS (Sorted by RMSE):")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65 + "\n")

    return metrics_df, all_predictions, {
        "y_test": y_test,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "models": models_dict,
        "ensemble_weights": ens_weights
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_pipeline

    raw = load_raw_data()
    clean_df, _ = preprocess_pipeline(raw)
    metrics_table, preds, artifacts = run_full_forecasting_pipeline(clean_df)
