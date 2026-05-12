# Phase 1 — Original Paper Model Reproduction

## Goal

Reproduce the methodology from the UNIST/WEIL Group paper using the full 3,757-point dataset
across all adsorbent types. Compare ten ML algorithms to identify the best overall predictor.

## Models Implemented

Each model was implemented in its own script at the repository root level:

| Script | Algorithm | Notes |
|--------|-----------|-------|
| `BA_model.py` | Bagging Regressor | Ensemble of base estimators |
| `CB_model.py` | CatBoost | Gradient boosting with categorical support |
| `DT_model.py` | Decision Tree | Single tree baseline |
| `ET_model.py` | Extra Trees | Randomised tree ensemble |
| `GB_model.py` | Gradient Boosting | Sklearn standard GB |
| `HGB_model.py` | HistGradientBoosting | Faster histogram-based GB |
| `KNN_model.py` | K-Nearest Neighbours | Instance-based learner |
| `LGBM_model.py` | LightGBM | Microsoft fast GB framework |
| `RF_model.py` | Random Forest | Bagged decision trees |
| `XGB_model.py` | XGBoost | Extreme Gradient Boosting |

Visualisation scripts:
- `Correlation_Heatmap.py` — Pearson correlation matrix across all numeric features
- `PDP_categorical.py` / `PDP_non_categorical.py` — Partial Dependence Plots
- `SHAP_CB_model.py` — SHAP explainability for the CatBoost winner

## Best Model — CatBoost

CatBoost achieved the highest test performance across all ten models:

| Metric | Value |
|--------|-------|
| Test R² | **0.9433** |
| Test MAE | **4.95** |

This matches the published paper's reported numbers, confirming the pipeline is correct.

## SHAP Feature Importance (CatBoost)

SHAP (SHapley Additive exPlanations) was applied to the CatBoost model to decompose which
feature groups drive predictions most:

| Feature Group | Contribution |
|--------------|-------------|
| Adsorption experimental conditions | **41%** |
| Adsorbent composition | **35%** |
| Adsorbent characterisation | **20%** |
| Synthesis conditions | **3%** |

This confirms that operational parameters (pH, temperature, time, concentration, dosage) are
the single largest driver of adsorption capacity — more important than even the physical
properties of the biochar itself.

The top individual predictive features identified were:
- N/C ratio (0.017 optimal)
- BET surface area (~1040 m²/g optimal)
- C(%) content (~82.1%)
- Pore volume (~0.46 cm³/g)
- Initial EC concentration (100 mg/L)
- Contaminant type (CAR — carbamazepine)
- Adsorption type (Single)
- Contact time (720 min)

## Partial Dependence Plots

PDP scripts compute and plot the marginal effect of each feature on the predicted adsorption
capacity, holding all other features at their mean values. Separate scripts handle:
- `PDP_categorical.py` — Discrete/category features (adsorbent type, contaminant type, etc.)
- `PDP_non_categorical.py` — Continuous features (pH, temperature, BET surface area, etc.)

## Note on Deleted Files

The original model scripts (`BA_model.py`, `CB_model.py`, etc.) were removed from the working
tree in the current local state (shown as `D` in git status). They are fully recoverable from
git history at commit `4c3db7f` and subsequent commits. The `Raw_data.csv` was added at commit
`25d8188`. The `Raw_data.xlsx` source file was at commit `3fa5995`.
