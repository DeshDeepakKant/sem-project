# Streamlit Web App

## Overview

**File:** `project-ppt/ML_Adsorption_Paper1_Method/app.py` (≈ 850 lines)

An interactive web application that lets users:
- Select a biochar material
- Input operating conditions via sliders
- Get live predictions from GPR, SVR, and ANN simultaneously
- Explore model performance dashboards
- View GA-optimised conditions for each biochar

## How to Run

```bash
# From the repository root
streamlit run project-ppt/ML_Adsorption_Paper1_Method/app.py

# Or from inside the ML folder
cd project-ppt/ML_Adsorption_Paper1_Method
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

**Prerequisite:** The pipeline must have been run first (`python run_pipeline.py`) so that
trained models, scalers, and result JSON files exist in `models/` and `results/`.

## App Structure

### Sidebar — Biochar & Input Selection

| Control | Options |
|---------|---------|
| Biochar selector | PAC / PB600 / NaOH-activated SCW |
| pH slider | 2.0 – 14.0 (step 0.5) |
| Temperature slider | 5 – 60 °C (step 1) |
| Contact Time slider | 1 – 1440 min (step 10) |
| Initial Concentration slider | 0.5 – 500 mg/L (step 0.5) |
| Adsorbent Dosage slider | 0.001 – 1.0 g/L (step 0.001) |
| Predict button | Triggers prediction across all three models |

### Main Panel

**Hero banner** — Project title and description.

**Prediction cards (3 columns)** — One card per model (GPR / SVR / ANN):
- Predicted adsorption capacity (mg/g) in large bold font
- Colour-coded by model: GPR = blue, SVR = orange, ANN = green
- GPR additionally shows ±1σ uncertainty interval

**Model Performance Section** — Loads `*_results.json` for the selected biochar and displays:
- R², MAE, RMSE for train and test sets in styled metric cards
- Predicted vs Actual scatter plot for each model (test set)

**GA Optimised Conditions Section** — Reads `ga_results.json` for the selected biochar and shows:
- The fitness model used (SVR or ANN)
- Predicted maximum capacity
- A table of optimised feature values

**Dataset Overview Section** — Reads `dataset_stats.json` and `cleaned_combined.csv`:
- Summary statistics per biochar
- Feature distribution box plots across the three biochars

## Design

The app uses a dark-mode CSS theme (`#0d0d1a` background gradient) with:
- Blue accent header (`#60a5fa`)
- Section headers with left border bar
- Metric cards with `linear-gradient(135deg, #111827, #1e293b)` backgrounds
- Biochar colours: PAC = red (`#e74c3c`), PB600 = green (`#2ecc71`), NaOH = blue (`#3498db`)
- Model colours: GPR = `#2196F3`, SVR = `#FF9800`, ANN = `#4CAF50`

## Key Implementation Notes

- Models are loaded with `@st.cache_resource` to avoid reloading on every interaction.
- GPR uses `gpr.predict(X, return_std=True)` to expose prediction uncertainty.
- TensorFlow logs are suppressed (`TF_CPP_MIN_LOG_LEVEL=3`) to keep the UI output clean.
- The scaler for the selected biochar is applied inside the prediction function so user inputs
  in physical units are correctly standardised before being passed to the models.
- If model files are missing (pipeline not yet run), the app shows a clear warning rather than
  crashing.

## Feature Input Ranges in App

These match the slider bounds defined in `FEATURE_RANGES` in `app.py`:

| Feature | Min | Max | Default | Step |
|---------|-----|-----|---------|------|
| pH | 2.0 | 14.0 | 7.0 | 0.5 |
| Temperature (°C) | 5.0 | 60.0 | 25.0 | 1.0 |
| Contact Time (min) | 1.0 | 1440.0 | 720.0 | 10.0 |
| Initial Concentration (mg/L) | 0.5 | 500.0 | 50.0 | 0.5 |
| Adsorbent Dosage (g/L) | 0.001 | 1.0 | 0.05 | 0.001 |
