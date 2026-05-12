"""
Step 3: Train Support Vector Regression (SVR)
- Try RBF and Linear kernels
- GridSearchCV for hyperparameter tuning (C, gamma, epsilon)
- Evaluate on test set
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- Load processed data ---
print("=" * 60)
print("STEP 3: SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"\n[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Test samples: {X_test.shape[0]}")

# ============================================================
# SVR with RBF Kernel + Grid Search
# ============================================================
print("\n" + "-" * 40)
print("Training SVR with RBF Kernel (GridSearchCV)")
print("-" * 40)

param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.5]
}

svr_rbf = SVR(kernel='rbf')
start_time = time.time()

grid_rbf = GridSearchCV(
    svr_rbf, param_grid_rbf,
    cv=5, scoring='r2',
    n_jobs=-1, verbose=1
)
grid_rbf.fit(X_train, y_train)
rbf_train_time = time.time() - start_time

print(f"\n[INFO] Best RBF params: {grid_rbf.best_params_}")
print(f"[INFO] Best RBF CV R²: {grid_rbf.best_score_:.4f}")
print(f"[INFO] Training time: {rbf_train_time:.2f}s")

# Evaluate RBF SVR
svr_rbf_best = grid_rbf.best_estimator_
y_pred_train_rbf = svr_rbf_best.predict(X_train)
y_pred_test_rbf = svr_rbf_best.predict(X_test)

rbf_train_metrics = {
    'R2': float(r2_score(y_train, y_pred_train_rbf)),
    'MAE': float(mean_absolute_error(y_train, y_pred_train_rbf)),
    'RMSE': float(np.sqrt(mean_squared_error(y_train, y_pred_train_rbf)))
}
rbf_test_metrics = {
    'R2': float(r2_score(y_test, y_pred_test_rbf)),
    'MAE': float(mean_absolute_error(y_test, y_pred_test_rbf)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_test_rbf)))
}

print(f"\n--- SVR-RBF Training Metrics ---")
print(f"  R²:   {rbf_train_metrics['R2']:.4f}")
print(f"  MAE:  {rbf_train_metrics['MAE']:.4f}")
print(f"  RMSE: {rbf_train_metrics['RMSE']:.4f}")

print(f"\n--- SVR-RBF Test Metrics ---")
print(f"  R²:   {rbf_test_metrics['R2']:.4f}")
print(f"  MAE:  {rbf_test_metrics['MAE']:.4f}")
print(f"  RMSE: {rbf_test_metrics['RMSE']:.4f}")

# ============================================================
# SVR with Linear Kernel + Grid Search
# ============================================================
print("\n" + "-" * 40)
print("Training SVR with Linear Kernel (GridSearchCV)")
print("-" * 40)

param_grid_lin = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1.0]
}

svr_lin = SVR(kernel='linear')
start_time = time.time()

grid_lin = GridSearchCV(
    svr_lin, param_grid_lin,
    cv=5, scoring='r2',
    n_jobs=-1, verbose=1
)
grid_lin.fit(X_train, y_train)
lin_train_time = time.time() - start_time

print(f"\n[INFO] Best Linear params: {grid_lin.best_params_}")
print(f"[INFO] Best Linear CV R²: {grid_lin.best_score_:.4f}")
print(f"[INFO] Training time: {lin_train_time:.2f}s")

# Evaluate Linear SVR
svr_lin_best = grid_lin.best_estimator_
y_pred_train_lin = svr_lin_best.predict(X_train)
y_pred_test_lin = svr_lin_best.predict(X_test)

lin_train_metrics = {
    'R2': float(r2_score(y_train, y_pred_train_lin)),
    'MAE': float(mean_absolute_error(y_train, y_pred_train_lin)),
    'RMSE': float(np.sqrt(mean_squared_error(y_train, y_pred_train_lin)))
}
lin_test_metrics = {
    'R2': float(r2_score(y_test, y_pred_test_lin)),
    'MAE': float(mean_absolute_error(y_test, y_pred_test_lin)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_test_lin)))
}

print(f"\n--- SVR-Linear Training Metrics ---")
print(f"  R²:   {lin_train_metrics['R2']:.4f}")
print(f"  MAE:  {lin_train_metrics['MAE']:.4f}")
print(f"  RMSE: {lin_train_metrics['RMSE']:.4f}")

print(f"\n--- SVR-Linear Test Metrics ---")
print(f"  R²:   {lin_test_metrics['R2']:.4f}")
print(f"  MAE:  {lin_test_metrics['MAE']:.4f}")
print(f"  RMSE: {lin_test_metrics['RMSE']:.4f}")

# --- Select best SVR model ---
if rbf_test_metrics['R2'] >= lin_test_metrics['R2']:
    best_svr = svr_rbf_best
    best_kernel = 'RBF'
    best_params = grid_rbf.best_params_
    best_train_metrics = rbf_train_metrics
    best_test_metrics = rbf_test_metrics
    best_train_time = rbf_train_time
    y_pred_test_best = y_pred_test_rbf
    y_pred_train_best = y_pred_train_rbf
else:
    best_svr = svr_lin_best
    best_kernel = 'Linear'
    best_params = grid_lin.best_params_
    best_train_metrics = lin_train_metrics
    best_test_metrics = lin_test_metrics
    best_train_time = lin_train_time
    y_pred_test_best = y_pred_test_lin
    y_pred_train_best = y_pred_train_lin

print(f"\n[INFO] Best SVR kernel: {best_kernel}")

# --- Save best SVR model ---
model_path = os.path.join(MODELS_DIR, 'svr_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_svr, f)
print(f"[INFO] Best SVR model saved to: {model_path}")

# --- Save results ---
results = {
    'model': f'Support Vector Regression (SVR-{best_kernel})',
    'best_kernel': best_kernel,
    'best_params': {k: str(v) for k, v in best_params.items()},
    'training_time_seconds': round(best_train_time, 2),
    'train_metrics': best_train_metrics,
    'test_metrics': best_test_metrics,
    'rbf_test_R2': rbf_test_metrics['R2'],
    'linear_test_R2': lin_test_metrics['R2'],
    'comparison': {
        'SVR_RBF': rbf_test_metrics,
        'SVR_Linear': lin_test_metrics
    }
}

results_path = os.path.join(RESULTS_DIR, 'svr_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"[INFO] SVR results saved to: {results_path}")

# --- Save predictions ---
predictions = {
    'y_test': y_test,
    'y_pred_test': y_pred_test_best,
    'y_train': y_train,
    'y_pred_train': y_pred_train_best
}
pred_path = os.path.join(RESULTS_DIR, 'svr_predictions.pkl')
with open(pred_path, 'wb') as f:
    pickle.dump(predictions, f)
print(f"[INFO] SVR predictions saved to: {pred_path}")

print("\n" + "=" * 60)
print("SVR MODEL TRAINING COMPLETE ✓")
print("=" * 60)
