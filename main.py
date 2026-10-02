"""
main.py
Master execution pipeline for Weather Trend Forecasting (Advanced Assessment).
Integrates:
- Data Ingestion & Preprocessing
- Exploratory Data Analysis (EDA)
- Anomaly Detection (Isolation Forest & Extremes)
- Multi-Model Forecasting (Baseline, Ridge, Random Forest, Gradient Boosting, Deep Learning)
- Meta-Ensemble Architecture
- Unique Advanced Analyses (Climate, Environmental, Feature Importance, Spatial, Geographical)
- PM Accelerator Mission Presentation
"""

import os
import sys
import json
import time

# Ensure src modules are resolvable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from data_loader import load_raw_data, get_dataset_metadata
from preprocessing import preprocess_pipeline
from eda import run_full_eda
from anomaly_detection import run_full_anomaly_detection
from models import run_full_forecasting_pipeline
from evaluation import plot_model_comparison, plot_residuals, plot_actual_vs_predicted
from advanced_analyses import run_all_advanced_analyses


PM_ACCELERATOR_MISSION = (
    "The Product Manager Accelerator (PMA), founded by Dr. Nancy Li, is on a mission to "
    "help ambitious professionals uncover their potential, gain the confidence to land "
    "product manager and AI product manager roles at leading tech companies and startup unicorns, "
    "and accelerate their career growth through world-class coaching, training, and community."
)

PM_ACCELERATOR_KIDS_MISSION = (
    "PMA Kids: Spark curiosity, build foundational skills, and foster a passion for "
    "technology and AI in the next generation by providing free AI bootcamp training "
    "to teenagers from underserved communities."
)


def print_banner():
    banner = f"""
========================================================================================
             TECH ASSESSMENT: WEATHER TREND FORECASTING (ADVANCED PIPELINE)
========================================================================================
PM ACCELERATOR MISSION:
  "{PM_ACCELERATOR_MISSION}"

PMA KIDS PHILANTHROPIC INITIATIVE:
  "{PM_ACCELERATOR_KIDS_MISSION}"

Core Values:
  • Growth Mindset  • Coachability  • Speed & Execution
  • Ownership & Accountability  • Excellence & Service  • Real-World Impact
========================================================================================
"""
    print(banner)


def run_pipeline():
    start_time = time.time()
    print_banner()

    # 1. Ingestion
    print("\n[Step 1/7] Ingesting Raw Dataset...")
    raw_data = load_raw_data()
    metadata = get_dataset_metadata(raw_data)
    print(f"  -> Raw records: {metadata['total_rows']:,}, Features: {metadata['total_columns']}")

    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/7] Preprocessing & Time-Series Feature Engineering...")
    clean_df, sensor_report = preprocess_pipeline(raw_data)
    print(f"  -> Engineered features count: {clean_df.shape[1]}")

    # 3. Exploratory Data Analysis
    print("\n[Step 3/7] Generating Exploratory Data Analysis & Visualizations...")
    eda_results = run_full_eda(clean_df, output_dir="outputs/figures")

    # 4. Anomaly Detection & Outlier Analysis
    print("\n[Step 4/7] Running Multi-Method Anomaly Detection...")
    anom_df, anom_results = run_full_anomaly_detection(clean_df, fig_dir="outputs/figures", metric_dir="outputs/metrics")

    # 5. Multi-Model Forecasting Suite & Meta-Ensemble
    print("\n[Step 5/7] Training & Evaluating Multi-Model Forecasting Suite...")
    metrics_table, predictions, model_artifacts = run_full_forecasting_pipeline(clean_df)

    # Save benchmark table
    benchmark_path = os.path.join("outputs", "metrics", "model_benchmark_results.csv")
    metrics_table.to_csv(benchmark_path, index=False)
    print(f"  -> Saved Model Benchmark Table: {benchmark_path}")

    # 6. Evaluation Visualizations
    print("\n[Step 6/7] Generating Comparative Evaluation Visualizations...")
    plot_model_comparison(metrics_table, output_dir="outputs/figures")

    # Residuals plot
    y_test = model_artifacts["y_test"]
    res_dict = {
        name: (y_test, preds)
        for name, preds in predictions.items()
        if "Baseline" not in name
    }
    plot_residuals(res_dict, output_dir="outputs/figures")

    # Actual vs Predicted for best model (Ensemble)
    best_model_name = metrics_table.iloc[0]["model_name"]
    best_preds = predictions[best_model_name]
    plot_actual_vs_predicted(
        y_test, best_preds, model_name=best_model_name,
        output_dir="outputs/figures", sample_size=250
    )

    # 7. Advanced Analyses (All 5 Unique Analyses)
    print("\n[Step 7/7] Conducting Unique Advanced Analyses (5 Modules)...")
    test_df = model_artifacts["test_df"]
    feature_cols = model_artifacts["feature_cols"]
    rf_model = model_artifacts["models"]["RandomForest"]
    X_test = test_df[feature_cols].values

    adv_results = run_all_advanced_analyses(
        clean_df, rf_model, feature_cols, X_test, y_test,
        fig_dir="outputs/figures", metric_dir="outputs/metrics"
    )

    # 8. Export Master Summary JSON
    summary_data = {
        "pm_accelerator_mission": PM_ACCELERATOR_MISSION,
        "pma_kids_mission": PM_ACCELERATOR_KIDS_MISSION,
        "dataset_metadata": metadata,
        "anomaly_detection": anom_results["metrics"],
        "model_benchmarks": metrics_table.to_dict(orient="records"),
        "meta_ensemble_weights": model_artifacts["ensemble_weights"],
        "execution_time_seconds": round(time.time() - start_time, 2)
    }
    summary_path = os.path.join("outputs", "metrics", "pipeline_master_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[Master Pipeline] Completed in {summary_data['execution_time_seconds']}s!")
    print(f"[Master Pipeline] Master summary exported to: {summary_path}")
    print("=" * 80)
    print("All tasks finished successfully. Ready for dashboard launch and GitHub commit.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
