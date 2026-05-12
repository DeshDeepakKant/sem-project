"""
Step 5: Visualization
- Predicted vs Actual plots for each model
- Error distribution histograms
- Model comparison bar chart
- ANN training curve
- All saved to /plots
"""

import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# --- Style ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

COLORS = {
    'GPR': '#2196F3',   # Blue
    'SVR': '#FF9800',   # Orange
    'ANN': '#4CAF50',   # Green
}

print("=" * 60)
print("STEP 5: VISUALIZATION")
print("=" * 60)

# --- Load predictions ---
models_data = {}
for model_name, pred_file in [('GPR', 'gpr_predictions.pkl'),
                               ('SVR', 'svr_predictions.pkl'),
                               ('ANN', 'ann_predictions.pkl')]:
    pred_path = os.path.join(RESULTS_DIR, pred_file)
    if os.path.exists(pred_path):
        with open(pred_path, 'rb') as f:
            models_data[model_name] = pickle.load(f)
        print(f"[INFO] Loaded {model_name} predictions.")
    else:
        print(f"[WARNING] {pred_file} not found. Skipping {model_name}.")

# --- Load metrics ---
metrics_data = {}
for model_name, result_file in [('GPR', 'gpr_results.json'),
                                 ('SVR', 'svr_results.json'),
                                 ('ANN', 'ann_results.json')]:
    result_path = os.path.join(RESULTS_DIR, result_file)
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            metrics_data[model_name] = json.load(f)

# ============================================================
# PLOT 1: Predicted vs Actual (for each model)
# ============================================================
print("\n[INFO] Generating Predicted vs Actual plots...")

fig, axes = plt.subplots(1, len(models_data), figsize=(6 * len(models_data), 5.5))
if len(models_data) == 1:
    axes = [axes]

for idx, (name, data) in enumerate(models_data.items()):
    ax = axes[idx]
    y_test = data['y_test']
    y_pred = data['y_pred_test']

    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=COLORS.get(name, '#333'),
               edgecolors='none')

    # Perfect prediction line
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - margin, lim_max + margin],
            [lim_min - margin, lim_max + margin],
            'r--', linewidth=1.5, label='Ideal (y=x)')
    ax.set_xlim(lim_min - margin, lim_max + margin)
    ax.set_ylim(lim_min - margin, lim_max + margin)

    # Metrics annotation
    if name in metrics_data:
        m = metrics_data[name]['test_metrics']
        ax.text(0.05, 0.95, f"R² = {m['R2']:.4f}\nMAE = {m['MAE']:.2f}\nRMSE = {m['RMSE']:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('Actual Adsorption Capacity')
    ax.set_ylabel('Predicted Adsorption Capacity')
    ax.set_title(f'{name} — Predicted vs Actual')
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'predicted_vs_actual.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: predicted_vs_actual.png")

# ============================================================
# PLOT 2: Error Distribution Histograms
# ============================================================
print("[INFO] Generating error distribution plots...")

fig, axes = plt.subplots(1, len(models_data), figsize=(6 * len(models_data), 4.5))
if len(models_data) == 1:
    axes = [axes]

for idx, (name, data) in enumerate(models_data.items()):
    ax = axes[idx]
    y_test = data['y_test']
    y_pred = data['y_pred_test']
    errors = y_test - y_pred

    ax.hist(errors, bins=50, color=COLORS.get(name, '#333'), alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=np.mean(errors), color='black', linestyle='-', linewidth=1.2,
               label=f'Mean Error: {np.mean(errors):.2f}')

    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} — Error Distribution')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'error_distribution.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: error_distribution.png")

# ============================================================
# PLOT 3: Model Comparison Bar Chart
# ============================================================
print("[INFO] Generating model comparison bar chart...")

model_names = list(metrics_data.keys())
r2_scores = [metrics_data[m]['test_metrics']['R2'] for m in model_names]
mae_scores = [metrics_data[m]['test_metrics']['MAE'] for m in model_names]
rmse_scores = [metrics_data[m]['test_metrics']['RMSE'] for m in model_names]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² Score
bars = axes[0].bar(model_names, r2_scores,
                   color=[COLORS.get(m, '#333') for m in model_names],
                   edgecolor='black', linewidth=0.8)
axes[0].set_title('R² Score (Test Set)', fontweight='bold')
axes[0].set_ylabel('R²')
axes[0].set_ylim(0, 1.05)
for bar, val in zip(bars, r2_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# MAE
bars = axes[1].bar(model_names, mae_scores,
                   color=[COLORS.get(m, '#333') for m in model_names],
                   edgecolor='black', linewidth=0.8)
axes[1].set_title('MAE (Test Set)', fontweight='bold')
axes[1].set_ylabel('MAE')
for bar, val in zip(bars, mae_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# RMSE
bars = axes[2].bar(model_names, rmse_scores,
                   color=[COLORS.get(m, '#333') for m in model_names],
                   edgecolor='black', linewidth=0.8)
axes[2].set_title('RMSE (Test Set)', fontweight='bold')
axes[2].set_ylabel('RMSE')
for bar, val in zip(bars, rmse_scores):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: model_comparison.png")

# ============================================================
# PLOT 4: ANN Training Curve
# ============================================================
if 'ANN' in metrics_data:
    print("[INFO] Generating ANN training curve...")
    ann_results = metrics_data['ANN']
    if 'training_history' in ann_results:
        loss = ann_results['training_history']['loss']
        val_loss = ann_results['training_history']['val_loss']

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(loss) + 1)
        ax.plot(epochs, loss, 'b-', linewidth=1.5, label='Training Loss')
        ax.plot(epochs, val_loss, 'r-', linewidth=1.5, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('ANN Training & Validation Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ann_training_curve.png'), bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: ann_training_curve.png")

# ============================================================
# PLOT 5: Combined Results Summary
# ============================================================
print("[INFO] Generating combined summary plot...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Top row: Predicted vs Actual for each model
for idx, (name, data) in enumerate(models_data.items()):
    ax = fig.add_subplot(gs[0, idx])
    y_test = data['y_test']
    y_pred = data['y_pred_test']
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color=COLORS.get(name, '#333'), edgecolors='none')
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.2)
    if name in metrics_data:
        m = metrics_data[name]['test_metrics']
        ax.text(0.05, 0.95, f"R²={m['R2']:.4f}", transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal', adjustable='box')

# Bottom left: Comparison bar chart
ax = fig.add_subplot(gs[1, 0:2])
x = np.arange(len(model_names))
width = 0.25
bars1 = ax.bar(x - width, r2_scores, width, label='R²', color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, [m/max(mae_scores) for m in mae_scores], width, label='MAE (norm)', color='#FF9800', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, [r/max(rmse_scores) for r in rmse_scores], width, label='RMSE (norm)', color='#4CAF50', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_title('Normalized Model Comparison', fontweight='bold')
ax.legend()
ax.set_ylabel('Score')

# Bottom right: Performance table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
table_data = [['Model', 'R²', 'MAE', 'RMSE']]
for name in model_names:
    m = metrics_data[name]['test_metrics']
    table_data.append([name, f"{m['R2']:.4f}", f"{m['MAE']:.2f}", f"{m['RMSE']:.2f}"])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
# Style header row
for j in range(4):
    table[0, j].set_facecolor('#37474F')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Test Set Results', fontweight='bold', pad=20)

plt.suptitle('ML Adsorption Capacity Prediction — Results Summary',
             fontsize=16, fontweight='bold')
plt.savefig(os.path.join(PLOTS_DIR, 'results_summary.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: results_summary.png")

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS GENERATED ✓")
print("=" * 60)
