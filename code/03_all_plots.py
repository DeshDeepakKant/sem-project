"""
Step 03: Generate ALL plots — per-biochar + comparison + violin plots.
Matches the style of the original research paper.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(os.path.join(PLOTS_DIR, 'per_biochar'), exist_ok=True)
os.makedirs(os.path.join(PLOTS_DIR, 'comparison'), exist_ok=True)

BIOCHARS = {'PAC': 'PAC', 'PB600': 'PB600', 'NaOH-activated SCW biochars': 'NaOH_activated_SCW'}
COLORS = {'PAC': '#e74c3c', 'PB600': '#2ecc71', 'NaOH_activated_SCW': '#3498db'}
MODEL_NAMES = ['GPR', 'SVR', 'ANN']
MODEL_COLORS = {'GPR': '#2196F3', 'SVR': '#FF9800', 'ANN': '#4CAF50'}

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

def load_predictions(safe, model):
    path = os.path.join(RESULTS_DIR, safe, f'{model.lower()}_predictions.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_results(safe, model):
    path = os.path.join(RESULTS_DIR, safe, f'{model.lower()}_results.json')
    with open(path, 'r') as f:
        return json.load(f)

print("=" * 60)
print("STEP 3: GENERATING ALL PLOTS")
print("=" * 60)

# ================================================================
# PLOT 1: Per-Biochar — Predicted vs Actual (Train + Test colored)
# ================================================================
print("\n[1/8] Per-biochar Predicted vs Actual scatter plots...")
for biochar, safe in BIOCHARS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Predicted vs Actual — {biochar}', fontsize=16, fontweight='bold')
    for i, model in enumerate(MODEL_NAMES):
        pred = load_predictions(safe, model)
        ax = axes[i]
        ax.scatter(pred['y_train'], pred['y_pred_train'], alpha=0.5, s=20, c='#3498db', label='Train', edgecolors='none')
        ax.scatter(pred['y_test'], pred['y_pred_test'], alpha=0.7, s=30, c='#e74c3c', label='Test', edgecolors='none')
        res = load_results(safe, model)
        lims = [min(pred['y_test'].min(), pred['y_pred_test'].min()) - 5,
                max(pred['y_test'].max(), pred['y_pred_test'].max()) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
        ax.set_xlabel('Actual Adsorption Capacity')
        ax.set_ylabel('Predicted Adsorption Capacity')
        ax.set_title(f'{model} (R²={res["test_metrics"]["R2"]:.4f})')
        ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_predicted_vs_actual.png'))
    plt.close()
    print(f"  Saved: {safe}_predicted_vs_actual.png")

# ================================================================
# PLOT 2: Per-Biochar — Error Distribution
# ================================================================
print("[2/8] Per-biochar error distributions...")
for biochar, safe in BIOCHARS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Prediction Error Distribution — {biochar}', fontsize=16, fontweight='bold')
    for i, model in enumerate(MODEL_NAMES):
        pred = load_predictions(safe, model)
        errors = pred['y_test'] - pred['y_pred_test']
        ax = axes[i]
        ax.hist(errors, bins=20, color=MODEL_COLORS[model], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', lw=1)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model} (MAE={np.abs(errors).mean():.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_error_distribution.png'))
    plt.close()
    print(f"  Saved: {safe}_error_distribution.png")

# ================================================================
# PLOT 3: Per-Biochar — ANN Training Curves
# ================================================================
print("[3/8] Per-biochar ANN training curves...")
for biochar, safe in BIOCHARS.items():
    res = load_results(safe, 'ANN')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['history']['loss'], label='Training Loss', color='#3498db')
    ax.plot(res['history']['val_loss'], label='Validation Loss', color='#e74c3c')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'ANN Training Curve — {biochar}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_ann_training_curve.png'))
    plt.close()
    print(f"  Saved: {safe}_ann_training_curve.png")

# ================================================================
# PLOT 4: Comparison — Model Performance Across Biochars (Grouped Bar)
# ================================================================
print("[4/8] Comparison — model performance across biochars...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics_list = ['R2', 'MAE', 'RMSE']
titles = ['R² Score (higher is better)', 'MAE (lower is better)', 'RMSE (lower is better)']

for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
    ax = axes[idx]
    x = np.arange(len(BIOCHARS))
    width = 0.25
    for j, model in enumerate(MODEL_NAMES):
        vals = []
        for safe in BIOCHARS.values():
            res = load_results(safe, model)
            vals.append(res['test_metrics'][metric])
        bars = ax.bar(x + j * width, vals, width, label=model, color=MODEL_COLORS[model], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Biochar')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width)
    short_names = ['PAC', 'PB600', 'NaOH-SCW']
    ax.set_xticklabels(short_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Model Performance Comparison Across Biochars', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'model_comparison_across_biochars.png'))
plt.close()
print("  Saved: model_comparison_across_biochars.png")

# ================================================================
# PLOT 5: Comparison — Best Model Per Biochar
# ================================================================
print("[5/8] Comparison — best model per biochar...")
fig, ax = plt.subplots(figsize=(10, 6))
short_names = ['PAC', 'PB600', 'NaOH-SCW']
for j, model in enumerate(MODEL_NAMES):
    r2_vals = []
    for safe in BIOCHARS.values():
        res = load_results(safe, model)
        r2_vals.append(res['test_metrics']['R2'])
    ax.plot(short_names, r2_vals, 'o-', label=model, color=MODEL_COLORS[model], markersize=10, linewidth=2)
    for i, v in enumerate(r2_vals):
        ax.annotate(f'{v:.3f}', (short_names[i], v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
ax.set_xlabel('Biochar Material', fontsize=13)
ax.set_ylabel('Test R² Score', fontsize=13)
ax.set_title('R² Score Comparison: All Models × All Biochars', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'r2_comparison_line.png'))
plt.close()
print("  Saved: r2_comparison_line.png")

# ================================================================
# PLOT 6: Paired Violin Plots — Observed vs Predicted (Paper Fig 8 style)
# ================================================================
print("[6/8] Paired violin plots — observed vs predicted...")
for biochar, safe in BIOCHARS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Observed vs Predicted Distribution — {biochar}', fontsize=16, fontweight='bold')
    for i, model in enumerate(MODEL_NAMES):
        pred = load_predictions(safe, model)
        violin_data = pd.DataFrame({
            'Value': np.concatenate([pred['y_test'], pred['y_pred_test']]),
            'Type': ['Observed'] * len(pred['y_test']) + ['Predicted'] * len(pred['y_pred_test'])
        })
        ax = axes[i]
        parts = ax.violinplot([pred['y_test'], pred['y_pred_test']], positions=[1, 2], showmeans=True, showmedians=True)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#3498db', MODEL_COLORS[model]][j])
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Observed', 'Predicted'])
        ax.set_ylabel('Adsorption Capacity')
        res = load_results(safe, model)
        ax.set_title(f'{model} (R²={res["test_metrics"]["R2"]:.4f})')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_violin_obs_vs_pred.png'))
    plt.close()
    print(f"  Saved: {safe}_violin_obs_vs_pred.png")

# ================================================================
# PLOT 7: Comparison Violin — All Biochars Side-by-Side
# ================================================================
print("[7/8] Comparison violin — all biochars side-by-side...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
short_map = {'PAC': 'PAC', 'PB600': 'PB600', 'NaOH_activated_SCW': 'NaOH-SCW'}
for i, model in enumerate(MODEL_NAMES):
    ax = axes[i]
    all_data, labels = [], []
    for safe in BIOCHARS.values():
        pred = load_predictions(safe, model)
        errors = pred['y_test'] - pred['y_pred_test']
        all_data.append(errors)
        labels.append(short_map[safe])
    parts = ax.violinplot(all_data, showmeans=True, showmedians=True)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(COLORS.values())[j])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Prediction Error')
    ax.set_title(f'{model} — Error Distribution by Biochar')
    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Prediction Error Comparison Across Biochars', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'violin_error_comparison.png'))
plt.close()
print("  Saved: violin_error_comparison.png")

# ================================================================
# PLOT 8: Input Feature Distributions Per Biochar
# ================================================================
print("[8/8] Input feature distributions per biochar...")
features = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']
df_combined = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_combined.csv'))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(features):
    ax = axes[i]
    for biochar, safe in BIOCHARS.items():
        subset = df_combined[df_combined['Adsorbent'] == biochar][feat]
        ax.hist(subset, bins=20, alpha=0.5, label=short_map[BIOCHARS[biochar]], color=COLORS[BIOCHARS[biochar]])
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {feat}')
    ax.legend()

# Last subplot — target distribution
ax = axes[5]
for biochar, safe in BIOCHARS.items():
    subset = df_combined[df_combined['Adsorbent'] == biochar]['Adsorption capacity']
    ax.hist(subset, bins=20, alpha=0.5, label=short_map[BIOCHARS[biochar]], color=COLORS[BIOCHARS[biochar]])
ax.set_xlabel('Adsorption Capacity')
ax.set_ylabel('Count')
ax.set_title('Distribution of Target Variable')
ax.legend()

fig.suptitle('Feature & Target Distributions by Biochar', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'feature_distributions_by_biochar.png'))
plt.close()
print("  Saved: feature_distributions_by_biochar.png")

# ================================================================
# BONUS: Combined Dashboard
# ================================================================
print("\n[BONUS] Generating combined results dashboard...")
fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

for col, (biochar, safe) in enumerate(BIOCHARS.items()):
    short = short_map[safe]
    # Row 1: Predicted vs Actual (best model)
    best_model = max(MODEL_NAMES, key=lambda m: load_results(safe, m)['test_metrics']['R2'])
    pred = load_predictions(safe, best_model)
    res = load_results(safe, best_model)
    ax1 = fig.add_subplot(gs[0, col])
    ax1.scatter(pred['y_train'], pred['y_pred_train'], alpha=0.4, s=15, c='#3498db', label='Train')
    ax1.scatter(pred['y_test'], pred['y_pred_test'], alpha=0.6, s=25, c='#e74c3c', label='Test')
    lims = [min(pred['y_test'].min(), pred['y_pred_test'].min()) - 5,
            max(pred['y_test'].max(), pred['y_pred_test'].max()) + 5]
    ax1.plot(lims, lims, 'k--', alpha=0.5)
    ax1.set_title(f'{short} — {best_model} (R²={res["test_metrics"]["R2"]:.4f})', fontweight='bold')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.legend(fontsize=8)

    # Row 2: Model comparison bars
    ax2 = fig.add_subplot(gs[1, col])
    r2_vals = [load_results(safe, m)['test_metrics']['R2'] for m in MODEL_NAMES]
    bars = ax2.bar(MODEL_NAMES, r2_vals, color=[MODEL_COLORS[m] for m in MODEL_NAMES], alpha=0.85)
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax2.set_title(f'{short} — R² Comparison')
    ax2.set_ylabel('R² Score')
    ax2.grid(axis='y', alpha=0.3)

    # Row 3: Error violin
    ax3 = fig.add_subplot(gs[2, col])
    error_data = []
    for m in MODEL_NAMES:
        p = load_predictions(safe, m)
        error_data.append(p['y_test'] - p['y_pred_test'])
    parts = ax3.violinplot(error_data, showmeans=True, showmedians=True)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(MODEL_COLORS[MODEL_NAMES[j]])
        pc.set_alpha(0.7)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(MODEL_NAMES)
    ax3.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax3.set_title(f'{short} — Error Distribution')
    ax3.set_ylabel('Prediction Error')

fig.suptitle('ML Adsorption Capacity Prediction — 3-Biochar Dashboard', fontsize=20, fontweight='bold', y=0.98)
plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'combined_dashboard.png'))
plt.close()
print("  Saved: combined_dashboard.png")

print("\n[INFO] All plots generated ✓")
