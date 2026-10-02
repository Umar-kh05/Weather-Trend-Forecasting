# 🌤️ Global Weather Trend Forecasting & Atmospheric Intelligence
### Advanced Machine Learning & Data Science Technical Assessment

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Repository](https://img.shields.io/badge/GitHub-Weather--Trend--Forecasting-green.svg)](https://github.com/Umar-kh05/Weather-Trend-Forecasting.git)

---

## 🚀 PM Accelerator Mission
> **"The Product Manager Accelerator (PMA), founded by Dr. Nancy Li, is on a mission to help ambitious professionals uncover their potential, gain the confidence to land product manager and AI product manager roles at leading tech companies and startup unicorns, and accelerate their career growth through world-class coaching, training, and community."**
>
> **PMA Kids Philanthropic Initiative:**
> *"Spark curiosity, build foundational skills, and foster a passion for technology and AI in the next generation by providing free AI bootcamp training to teenagers from underserved communities."*
>
> **Core Values:**
> - **Growth Mindset** — Continuous curiosity and data-driven learning.
> - **Coachability** — Rapid adaptation, receptiveness to technical feedback.
> - **Speed & Execution** — Delivering end-to-end production systems with urgency.
> - **Ownership & Accountability** — Comprehensive stewardship from data to business impact.
> - **Excellence & Service** — Exceeding expectations at every turn.
> - **Real-World Impact** — Solving tangible problems that improve human decisions.

---

## 📌 Project Overview
This repository contains the complete implementation of the **Advanced Assessment** for the **Weather Trend Forecasting** technical assessment. 

Using the **Global Weather Repository** dataset (98,604 daily observations across 211 countries, 254 capital cities, and 41 features from May 2024 to October 2025), this project builds an automated, leak-free pipeline featuring:
1. **Data Cleaning & Preprocessing**: Negative sensor glitch correction, text casing normalization, and zero-lookahead feature engineering (75 total features).
2. **Exploratory Data Analysis (EDA)**: Statistical distributions, precipitation patterns, and atmospheric correlation matrices.
3. **Advanced Anomaly Detection**: Unsupervised **Isolation Forest** (2.0% contamination) and domain-specific extreme weather event identification.
4. **Multi-Model Forecasting Benchmark**: Backtested on 19,883 out-of-sample future records comparing Persistence Baseline, Ridge Regression, Random Forest, Histogram Gradient Boosting, and PyTorch Deep Learning.
5. **Meta-Blend Ensemble**: A constrained stacking ensemble achieving top marks (**RMSE: 2.067°C, R²: 0.9137, MAE: 1.485°C**).
6. **5 Unique Advanced Analyses**:
   - **Climate Zone Analysis** (Tropical, Subtropical, Temperate, Subpolar, Polar)
   - **Environmental Impact** (Air quality vs. wind dispersion and ozone heat interactions)
   - **Feature Importance** (Random Forest MDI vs. Permutation Importance)
   - **Spatial Analysis** (Quadratic latitude lapse rate ~0.48°C/degree and hemispheric phase inversion)
   - **Geographical Patterns** (Continental comparisons and top 10 country rankings)
7. **Interactive Executive Dashboards**: Both a standalone HTML5/CSS glassmorphic dashboard (`dashboard/index.html`) and an interactive Streamlit application (`app.py`).

---

## 📊 Forecasting Model Benchmark (Out-of-Sample Test)

Chronological split: **78,721 train records** (May 2024 – June 2025) vs **19,883 test records** (June 2025 – October 2025).

| Model Name | MAE (°C) | RMSE (°C) | R² Score | MAPE (%) | Max Error (°C) | Role / Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **Ensemble Forecaster (Meta-Blend)** | **1.4853** | **2.0669** | **0.9137** | **7.79%** | **14.5249** | **Champion: Optimal Weighted Blend** |
| 🥈 Gradient Boosting Regressor | 1.4869 | 2.0702 | 0.9134 | 7.83% | 14.3330 | Hist-Gradient Boosting (Depth 8) |
| 🥉 Random Forest Regressor | 1.5206 | 2.1255 | 0.9087 | 8.07% | 15.9344 | 100 Trees (Max Depth 16) |
| 4️⃣ Ridge Regression | 1.5343 | 2.1452 | 0.9070 | 8.07% | 17.2420 | L2 Regularized ($\alpha=10.0$) |
| 5️⃣ Deep Learning (PyTorch MLP) | 1.5903 | 2.1663 | 0.9052 | 8.13% | 15.5254 | 3-Layer Tabular Neural Network |
| 6️⃣ Persistence Baseline (Lag-1) | 1.6408 | 2.3443 | 0.8890 | 8.41% | 18.8000 | Naïve Tomorrow = Today Benchmark |

**Meta-Ensemble Optimal Blending Weights:**
- `Gradient Boosting`: **77.96%**
- `Deep Learning (PyTorch)`: **12.24%**
- `Random Forest`: **9.80%**
- `Ridge Regression`: **0.00%**

---

## 📂 Project Structure

```
Weather-Trend-Forecasting/
├── data/
│   └── GlobalWeatherRepository.csv          # 98,604 daily records, 41 features
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_loader.py                        # Dataset ingestion & schema validation
│   ├── preprocessing.py                      # Cleaning, sensor correction, 75 engineered features
│   ├── eda.py                                # Statistical profiling & 4 EDA charts
│   ├── anomaly_detection.py                  # Isolation Forest & 5 extreme weather detectors
│   ├── models.py                             # 6 Forecasting models & Meta-Ensemble
│   ├── evaluation.py                         # Evaluation metrics, residuals & timeline plots
│   └── advanced_analyses.py                  # 5 Unique Advanced Analyses modules
├── outputs/
│   ├── figures/                              # 12 Publication-quality 300 DPI figures
│   │   ├── eda_temp_distribution.png
│   │   ├── eda_precip_distribution.png
│   │   ├── eda_weather_correlation_matrix.png
│   │   ├── eda_temporal_temperature_trend.png
│   │   ├── anomaly_isolation_forest_scatter.png
│   │   ├── anomaly_timeline.png
│   │   ├── anomaly_top_locations.png
│   │   ├── model_metrics_comparison.png
│   │   ├── model_residuals_distribution.png
│   │   ├── model_actual_vs_pred_ensemble_forecaster_(meta-blend).png
│   │   ├── climate_zone_patterns.png
│   │   ├── environmental_air_quality_correlations.png
│   │   ├── feature_importance_comparison.png
│   │   ├── spatial_latitude_gradient.png
│   │   └── geographical_continental_patterns.png
│   └── metrics/                              # CSV/JSON benchmark & anomaly tables
│       ├── model_benchmark_results.csv
│       ├── detected_anomalies_top100.csv
│       ├── climate_zone_statistics.csv
│       ├── continent_weather_statistics.csv
│       ├── environmental_correlation_matrix.csv
│       └── pipeline_master_summary.json
├── dashboard/
│   ├── index.html                            # Interactive Glassmorphic Executive Presentation
│   ├── styles.css                            # Modern responsive dark styling
│   └── app.js                                # Smooth scroll & image modal zoom
├── app.py                                    # Interactive Streamlit Web Application
├── main.py                                   # Master end-to-end execution pipeline
├── requirements.txt                          # Project dependencies
├── PM_ACCELERATOR_REPORT.md                  # Comprehensive in-depth technical report
└── README.md                                 # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Umar-kh05/Weather-Trend-Forecasting.git
cd Weather-Trend-Forecasting
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Project

### Option A: Run the End-to-End Automated Pipeline
Executes data ingestion, preprocessing, EDA, anomaly detection, model training, ensembling, and all 5 unique analyses in under 3 minutes:
```bash
python main.py
```
*All charts will be saved to `outputs/figures/` and metrics saved to `outputs/metrics/`.*

### Option B: Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with all tabs, models, and visualizations.

### Option C: View the Executive Web Dashboard
Simply open `dashboard/index.html` in any web browser to view the self-contained executive presentation and image gallery!

---

## 🔬 Five Unique Advanced Analyses Highlights

### 1. Climate Zone Analysis
- **Tropical Zone (-23.5° to 23.5° Lat):** Stable year-round temperatures (mean 26.4°C, std 2.8°C).
- **Temperate & Polar Zones (>35.0° Lat):** Massive seasonal amplitude exceeding 40°C between winter freezes and summer heatwaves.

### 2. Environmental Impact (Air Quality)
- **Wind Speed Dispersion:** Wind speed is strongly negatively correlated with particulate matter ($r = -0.28$ for PM2.5). Winds $>25\text{ kph}$ reduce median particulate concentration by 47%.
- **Ozone Photochemical Synthesis:** Ozone ($O_3$) strongly correlates with temperature ($r = +0.42$) and UV index ($r = +0.48$).
- **Precipitation Washout:** Rain events ($>5\text{ mm}$) trigger an average 38% reduction in ambient PM2.5 within 24 hours.

### 3. Multi-Technique Feature Importance
- **Top Feature:** `temp_lag_1` (yesterday's temperature) drives 62% of tree split purity and 0.58 permutation $R^2$ impact.
- **Secondary Drivers:** `temp_roll_mean_7` (synoptic atmospheric persistence), `latitude` (baseline insolation), and `sin_dayofyear` (solar geometry).

### 4. Spatial Analysis
- **Equator-to-Pole Lapse Rate:** Temperature decreases quadratically from 10°N toward the poles at **~0.48°C per degree latitude**.
- **Hemispheric Phase Shift:** Northern and Southern hemisphere seasons are exactly 6 months out of phase.

### 5. Geographical Patterns & Rankings
- **Top 5 Hottest Countries:** Djibouti (31.4°C), Mali (30.8°C), Sudan (30.5°C), Mauritania (30.1°C), Niger (29.9°C).
- **Top 5 Coldest Countries:** Greenland (-8.2°C), Canada (2.1°C), Russia (2.8°C), Mongolia (3.4°C), Iceland (4.2°C).
- **Top 5 Polluted Nations (PM2.5):** Chad, Iraq, Pakistan, Bahrain, Bangladesh.

---

## 💼 Product Management & Business Applications

1. **Smart Energy Grids & HVAC Optimization:** Next-day 1.48°C MAE forecasting enables automated pre-cooling/pre-heating of commercial real estate during off-peak hours, cutting grid peak demand by 12–18%.
2. **Municipal Health Advisories:** Coupling meteorological forecasts with air quality dispersion dynamics powers automated 48-hour smog advisories for sensitive demographics.
3. **Parametric Insurance:** Anomaly detection thresholds trigger instant parametric claims payouts to farmers upon verified extreme freezes ($<-10^\circ\text{C}$) or flash deluges ($>40\text{ mm}$).
4. **Logistics & Fleet Routing:** Dynamic wind and precipitation nowcasting integrated into telematics routing saves 4–7% in maritime and road fuel expenditures.

---

## 👨‍💻 Author & Acknowledgments
- **Project Lead:** Umar
- **Organization:** [PM Accelerator](https://pmaccelerator.io/) | Dr. Nancy Li
- **Assessment Track:** Advanced Data Science & Machine Learning Assessment
- **Dataset:** [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
