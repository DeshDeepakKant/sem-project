"""
Step 8: Violin Plots
- Input feature distributions
- Target variable distribution
- Model prediction error distributions
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
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
    'figure.dpi': 150,
})

COLORS = {
    'features': ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0'],
    'target': '#37474F',
    'GPR': '#2196F3',
    'SVR': '#FF9800',
    'ANN': '#4CAF50',
}

print("=" * 60)
print("STEP 8: VIOLIN PLOTS")
print("=" * 60)

# --- Load raw data ---
df = pd.read_csv(os.path.join(DATA_DIR, 'Cleaned_data.csv'))
feature_cols = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']
target_col = 'Adsorption capacity'

# --- Load predictions ---
models_pred = {}
for name, pred_file in [('GPR', 'gpr_predictions.pkl'),
                         ('SVR', 'svr_predictions.pkl'),
                         ('ANN', 'ann_predictions.pkl')]:
    path = os.path.join(RESULTS_DIR, pred_file)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models_pred[name] = pickle.load(f)
        print(f"[INFO] Loaded {name} predictions.")

# ============================================================
# PLOT 1: Violin plots for all 5 input features
# ============================================================
print("\n[INFO] Generating input feature violin plots...")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for i, col in enumerate(feature_cols):
    parts = axes[i].violinplot(df[col].dropna().values, positions=[0],
                                showmeans=True, showmedians=True, showextrema=True)

    # Style the violin body
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['features'][i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)

    # Style lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')

    # Add scatter of actual points (jittered)
    jitter = np.random.normal(0, 0.02, size=len(df[col]))
    axes[i].scatter(jitter, df[col].values, alpha=0.03, s=3, color=COLORS['features'][i])

    axes[i].set_title(col, fontweight='bold', fontsize=12)
    axes[i].set_xticks([])
    axes[i].set_ylabel('Value')

    # Stats annotation
    stats_text = f"μ={df[col].mean():.2f}\nσ={df[col].std():.2f}\nmed={df[col].median():.2f}"
    axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Input Feature Distributions (Violin Plots)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'violin_input_features.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: violin_input_features.png")

# ============================================================
# PLOT 2: Violin plot for target variable
# ============================================================
print("[INFO] Generating target variable violin plot...")

fig, ax = plt.subplots(figsize=(6, 7))

parts = ax.violinplot(df[target_col].dropna().values, positions=[0],
                       showmeans=True, showmedians=True, showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor(COLORS['target'])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
    pc.set_linewidth(1.2)

parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2.5)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2.5)
parts['cbars'].set_color('black')
parts['cmins'].set_color('black')
parts['cmaxes'].set_color('black')

# Add jittered points
jitter = np.random.normal(0, 0.03, size=len(df[target_col]))
ax.scatter(jitter, df[target_col].values, alpha=0.05, s=5, color='#FF5722')

ax.set_title('Adsorption Capacity Distribution', fontsize=14, fontweight='bold')
ax.set_xticks([])
ax.set_ylabel('Adsorption Capacity')

# Quartile annotations
q1 = df[target_col].quantile(0.25)
q3 = df[target_col].quantile(0.75)
stats_text = (f"Mean: {df[target_col].mean():.2f}\n"
              f"Median: {df[target_col].median():.2f}\n"
              f"Std: {df[target_col].std():.2f}\n"
              f"Q1: {q1:.2f}\n"
              f"Q3: {q3:.2f}\n"
              f"Min: {df[target_col].min():.2f}\n"
              f"Max: {df[target_col].max():.2f}")
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'violin_target_variable.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: violin_target_variable.png")

# ============================================================
# PLOT 3: Violin plots for model prediction errors
# ============================================================
print("[INFO] Generating prediction error violin plots...")

model_names = list(models_pred.keys())
errors = {}
for name, data in models_pred.items():
    errors[name] = data['y_test'] - data['y_pred_test']

fig, ax = plt.subplots(figsize=(10, 7))

positions = list(range(len(model_names)))
all_parts = []

for i, name in enumerate(model_names):
    parts = ax.violinplot(errors[name], positions=[i],
                           showmeans=True, showmedians=True, showextrema=True)

    for pc in parts['bodies']:
        pc.set_facecolor(COLORS.get(name, '#666'))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)

    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')

    # Jittered scatter
    jitter = np.random.normal(i, 0.04, size=len(errors[name]))
    ax.scatter(jitter, errors[name], alpha=0.08, s=6, color=COLORS.get(name, '#666'))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Error')
ax.set_xticks(positions)
ax.set_xticklabels(model_names, fontsize=13, fontweight='bold')
ax.set_ylabel('Prediction Error (Actual − Predicted)', fontsize=13)
ax.set_title('Model Prediction Error Distributions', fontsize=16, fontweight='bold')
ax.legend(loc='upper right')

# Stats annotation per model
for i, name in enumerate(model_names):
    e = errors[name]
    stats_text = f"μ={np.mean(e):.2f}\nσ={np.std(e):.2f}"
    ax.text(i, ax.get_ylim()[1] * 0.9, stats_text, ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'violin_prediction_errors.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: violin_prediction_errors.png")

# ============================================================
# PLOT 4: Combined violin dashboard (all in one figure)
# ============================================================
print("[INFO] Generating combined violin dashboard...")

fig = plt.figure(figsize=(22, 14))
gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.4)

# Top row: 5 input features
for i, col in enumerate(feature_cols):
    ax = fig.add_subplot(gs[0, i])
    parts = ax.violinplot(df[col].dropna().values, positions=[0],
                           showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['features'][i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')

    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xticks([])
    ax.set_ylabel('Value', fontsize=10)

# Top row, last cell: target
ax = fig.add_subplot(gs[0, 5])
parts = ax.violinplot(df[target_col].dropna().values, positions=[0],
                       showmeans=True, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor(COLORS['target'])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('black')
parts['cmins'].set_color('black')
parts['cmaxes'].set_color('black')
ax.set_title('Adsorption\nCapacity (Target)', fontweight='bold', fontsize=11, color='#D32F2F')
ax.set_xticks([])
ax.set_ylabel('Value', fontsize=10)

# Bottom row: prediction errors (spanning all columns)
ax = fig.add_subplot(gs[1, 1:5])
for i, name in enumerate(model_names):
    parts = ax.violinplot(errors[name], positions=[i],
                           showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS.get(name, '#666'))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')

    jitter = np.random.normal(i, 0.04, size=len(errors[name]))
    ax.scatter(jitter, errors[name], alpha=0.06, s=4, color=COLORS.get(name, '#666'))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=13, fontweight='bold')
ax.set_ylabel('Prediction Error', fontsize=12)
ax.set_title('Model Prediction Error Distributions', fontweight='bold', fontsize=13)

plt.suptitle('Violin Plot Dashboard — Features, Target & Prediction Errors',
             fontsize=17, fontweight='bold', y=1.01)
plt.savefig(os.path.join(PLOTS_DIR, 'violin_dashboard.png'), bbox_inches='tight')
plt.close()
print("[INFO] Saved: violin_dashboard.png")

print("\n" + "=" * 60)
print("ALL VIOLIN PLOTS GENERATED ✓")
print("=" * 60)
