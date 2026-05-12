"""
Step 2: Train Gaussian Process Regression (GPR)
- RBF kernel with WhiteKernel for noise
- Hyperparameter optimization via marginal likelihood
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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- Load processed data ---
print("=" * 60)
print("STEP 2: GAUSSIAN PROCESS REGRESSION (GPR)")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"\n[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Test samples: {X_test.shape[0]}")

# --- GPR with subsampling (GPR scales poorly with large N) ---
# Use a subset for training GPR due to O(n^3) complexity
MAX_GPR_SAMPLES = 1500
if X_train.shape[0] > MAX_GPR_SAMPLES:
    print(f"\n[INFO] GPR is O(n³) — subsampling to {MAX_GPR_SAMPLES} training points for tractability.")
    np.random.seed(42)
    idx = np.random.choice(X_train.shape[0], MAX_GPR_SAMPLES, replace=False)
    X_train_gpr = X_train[idx]
    y_train_gpr = y_train[idx]
else:
    X_train_gpr = X_train
    y_train_gpr = y_train

# --- Define kernel ---
# RBF kernel with automatic length scale + constant kernel + white noise
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
    length_scale=1.0, length_scale_bounds=(1e-3, 1e2)
) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))

print(f"\n[INFO] Kernel: {kernel}")

# --- Train GPR ---
print("\n[INFO] Training GPR model (this may take a few minutes)...")
start_time = time.time()

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=42,
    alpha=1e-2,
    normalize_y=True
)
gpr.fit(X_train_gpr, y_train_gpr)

train_time = time.time() - start_time
print(f"[INFO] Training completed in {train_time:.2f} seconds.")
print(f"[INFO] Optimized kernel: {gpr.kernel_}")
print(f"[INFO] Log-marginal-likelihood: {gpr.log_marginal_likelihood_value_:.4f}")

# --- Predictions ---
y_pred_train = gpr.predict(X_train_gpr)
y_pred_test, y_std_test = gpr.predict(X_test, return_std=True)

# --- Evaluation ---
train_metrics = {
    'R2': float(r2_score(y_train_gpr, y_pred_train)),
    'MAE': float(mean_absolute_error(y_train_gpr, y_pred_train)),
    'RMSE': float(np.sqrt(mean_squared_error(y_train_gpr, y_pred_train)))
}

test_metrics = {
    'R2': float(r2_score(y_test, y_pred_test)),
    'MAE': float(mean_absolute_error(y_test, y_pred_test)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
}

print(f"\n--- GPR Training Metrics ---")
print(f"  R²:   {train_metrics['R2']:.4f}")
print(f"  MAE:  {train_metrics['MAE']:.4f}")
print(f"  RMSE: {train_metrics['RMSE']:.4f}")

print(f"\n--- GPR Test Metrics ---")
print(f"  R²:   {test_metrics['R2']:.4f}")
print(f"  MAE:  {test_metrics['MAE']:.4f}")
print(f"  RMSE: {test_metrics['RMSE']:.4f}")

# --- Save model ---
model_path = os.path.join(MODELS_DIR, 'gpr_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(gpr, f)
print(f"\n[INFO] GPR model saved to: {model_path}")

# --- Save results ---
results = {
    'model': 'Gaussian Process Regression (GPR)',
    'kernel': str(gpr.kernel_),
    'training_samples': int(X_train_gpr.shape[0]),
    'training_time_seconds': round(train_time, 2),
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'log_marginal_likelihood': float(gpr.log_marginal_likelihood_value_)
}

results_path = os.path.join(RESULTS_DIR, 'gpr_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"[INFO] GPR results saved to: {results_path}")

# --- Save predictions for plotting ---
predictions = {
    'y_test': y_test,
    'y_pred_test': y_pred_test,
    'y_std_test': y_std_test,
    'y_train': y_train_gpr,
    'y_pred_train': y_pred_train
}
pred_path = os.path.join(RESULTS_DIR, 'gpr_predictions.pkl')
with open(pred_path, 'wb') as f:
    pickle.dump(predictions, f)
print(f"[INFO] GPR predictions saved to: {pred_path}")

print("\n" + "=" * 60)
print("GPR MODEL TRAINING COMPLETE ✓")
print("=" * 60)
