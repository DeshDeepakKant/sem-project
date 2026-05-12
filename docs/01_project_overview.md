# Project Overview

## Research Context

This project is based on the published paper:

> **"Machine-learning-based prediction and optimization of emerging contaminants' adsorption
> capacity on biochar materials"**
> Zeeshan Haider Jaffari, Heewon Jeong, Jaegwan Shin, Jinwoo Kwak, Changgil Son,
> Yong-Gu Lee, Sangwon Kim, Kangmin Chon, Kyung Hwa Cho
> WEIL Group, UNIST (Ulsan National Institute of Science and Technology), South Korea

Emerging contaminants (ECs) — pharmaceuticals, pesticides, industrial chemicals — are found in
wastewater and surface water at trace concentrations. Biochar-based adsorption is a cost-effective
removal method, but experimentally mapping adsorption capacity across many materials and operating
conditions is expensive. Machine learning allows us to predict adsorption capacity from input
parameters directly, bypassing the need for exhaustive wet-lab screening.

## Problem Statement

Given a set of experimental conditions and adsorbent properties, predict the **adsorption capacity**
(mg of contaminant removed per gram of adsorbent, mg/g) for a given biochar material.

**Input features used in Phase 2 (per-biochar models):**

| Feature | Unit | Role |
|---------|------|------|
| Solution pH | — | Affects surface charge and contaminant speciation |
| Adsorption temperature | °C | Controls kinetics and thermodynamics |
| Contact time | min | Duration of adsorption experiment |
| Initial concentration | mg/L | Starting contaminant load |
| Adsorbent dosage | g/L | Mass of biochar per litre of solution |

**Output:** Adsorption capacity (mg/g)

## Two-Phase Structure

### Phase 1 — Paper Reproduction (10 Models, Full Dataset)

We reproduced the original paper's methodology on the full 3,757-row dataset covering many
adsorbent types. Ten tree-based and ensemble models were trained and compared. The goal was to
validate our preprocessing pipeline against the paper's published results.

Models trained: Bagging (BA), CatBoost (CB), Decision Tree (DT), Extra Trees (ET),
Gradient Boosting (GB), HistGradientBoosting (HGB), KNN, LightGBM (LGBM), Random Forest (RF),
XGBoost (XGB).

SHAP explainability was applied to the top model (CatBoost) to quantify feature importance.

### Phase 2 — Per-Biochar Pipeline (GPR + SVR + ANN)

A new, structured pipeline was developed for three specific biochar materials:
- **PAC** — Powdered Activated Carbon (commercial baseline)
- **PB600** — Biochar produced at 600 °C pyrolysis temperature
- **NaOH-activated SCW** — Sodium hydroxide activated sugarcane-derived biochar

Each biochar gets its own trained models (GPR, SVR, ANN) and its own GA optimization run.
The per-biochar approach avoids cross-adsorbent confounding and produces material-specific
optimized conditions.

A Streamlit app was then built to serve all trained models through an interactive web interface.

## Key Findings Summary

- CatBoost dominated the full-dataset comparison (Phase 1) with R² = 0.9433.
- SHAP analysis revealed that experimental conditions (pH, temperature, contact time, etc.)
  drive 41% of model prediction, while adsorbent composition accounts for 35%.
- In Phase 2, GPR achieved the best results on PAC (R² = 0.772) and SVR tied closely.
  ANN performed best on PB600 (R² = 0.755) and on NaOH-SCW (R² = 0.176 — limited by dataset size).
- GA optimization identified distinct optimal operating windows for each biochar (see
  [06_genetic_algorithm.md](06_genetic_algorithm.md)).
