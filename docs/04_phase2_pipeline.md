# Phase 2 — Per-Biochar ML Pipeline

## Overview

The Phase 2 pipeline lives entirely inside:

```
project-ppt/ML_Adsorption_Paper1_Method/
```

It trains three models (GPR, SVR, ANN) independently for each of the three target biochars,
then runs Genetic Algorithm optimisation per biochar. The entry point is `run_pipeline.py`.

## Pipeline Steps

```
01_data_preprocessing.py
        ↓
02_train_all_models.py      (GPR + SVR + ANN per biochar)
        ↓
03_all_plots.py             (scatter plots, error histograms, violin plots, comparison charts)
        ↓
04_genetic_algorithm.py     (GA optimisation per biochar using best model as fitness function)
        ↓
05_generate_report.py       (Markdown summary report)
```

Run the whole sequence with:

```bash
cd project-ppt/ML_Adsorption_Paper1_Method
python run_pipeline.py
```

---

## Step 01 — Data Preprocessing

**Script:** `code/01_data_preprocessing.py`

- Loads `Raw_data.csv` from the repository root.
- Filters to the three target biochars by matching the `Adsorbent` column.
- Renames columns to clean names.
- Drops NaN rows.
- 80/20 train/test split (`random_state=42`).
- Fits a `StandardScaler` per biochar on the training set.
- Saves `processed_<safe>.pkl`, `scaler_<safe>.pkl`, `cleaned_<safe>.csv`,
  `cleaned_combined.csv`, and `dataset_stats.json` to `data/`.

---

## Step 02 — Train All Models

**Script:** `code/02_train_all_models.py`

Iterates over each of the three biochars and trains all three models, saving results to
`results/<safe>/` and models to `models/<safe>/`.

### GPR (Gaussian Process Regression)

```
Kernel: ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
Optimisation: marginal likelihood maximisation (n_restarts_optimizer=5, random_state=42)
```

GPR provides uncertainty estimates (`y_std`) on test predictions alongside point estimates.
For large datasets (NaOH-SCW has 388 training points), runtime grows as O(n³) — a known
limitation. The current implementation trains on the full training set without subsampling.

### SVR (Support Vector Regression)

```
Kernels tried: RBF, Linear
Hyperparameter search: GridSearchCV (cv=3, scoring='r2', n_jobs=-1)
  C:       [0.1, 1, 10, 100]
  gamma:   ['scale', 'auto', 0.01, 0.1]
  epsilon: [0.01, 0.1, 0.5]
```

The kernel with higher test R² is selected and saved as `svr_model.pkl`.

### ANN (Artificial Neural Network)

```
Framework: TensorFlow / Keras
Architecture: Dense(64, relu) → BN → Dropout(0.2) → Dense(32, relu) → BN → Dropout(0.2)
              → Dense(16, relu) → Dense(1)
Optimiser: Adam (lr=0.001)
Loss: MSE
Epochs: up to 200, EarlyStopping(patience=20, restore_best_weights=True)
Batch size: 16
Validation split: 20% of training data
```

Saved as `ann_model.keras` (Keras native format).

---

## Step 03 — Plots

**Script:** `code/03_all_plots.py`

Generates eight plot types saved to `plots/`:

| Plot | Description | Location |
|------|-------------|----------|
| Predicted vs Actual scatter | Train (blue) + Test (red), per biochar × model | `plots/per_biochar/` |
| Error distribution | Histogram of residuals per biochar × model | `plots/per_biochar/` |
| Violin (observed vs predicted) | Distribution shape comparison | `plots/per_biochar/` |
| ANN training curve | Loss vs epoch | `plots/per_biochar/` |
| GA convergence | Best fitness per generation | `plots/per_biochar/` |
| Model R² comparison | Bar chart across 3 biochars × 3 models | `plots/comparison/` |
| R² line comparison | Line chart across biochars | `plots/comparison/` |
| Violin error comparison | Error spread across models | `plots/comparison/` |
| Feature distribution by biochar | Distribution of each input feature | `plots/comparison/` |
| Combined dashboard | Multi-panel summary | `plots/comparison/` |

---

## Step 04 — Genetic Algorithm

See [06_genetic_algorithm.md](06_genetic_algorithm.md) for full details.

---

## Step 05 — Report Generation

**Script:** `code/05_generate_report.py`

Reads all `*_results.json` and `dataset_stats.json` files and produces a Markdown summary
report in `report/` with:
- Dataset table (samples, train/test split, capacity range per biochar)
- Model performance comparison table per biochar
- GA optimised conditions per biochar

---

## Dependencies

```
streamlit >= 1.32.0
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
tensorflow-cpu >= 2.15.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
openpyxl >= 3.1.0
bayesian-optimization >= 1.4.0
shap >= 0.44.0
```

Install with:

```bash
pip install -r project-ppt/requirements.txt
```
