"""
Step 7: Generate Report
- Reads all results JSON files
- Generates a comprehensive markdown report
"""

import os
import json
from datetime import datetime

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORT_DIR = os.path.join(BASE_DIR, 'report')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

print("=" * 60)
print("STEP 7: REPORT GENERATION")
print("=" * 60)

# --- Load all results ---
def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

dataset_stats = load_json('dataset_stats.json')
gpr_results = load_json('gpr_results.json')
svr_results = load_json('svr_results.json')
ann_results = load_json('ann_results.json')
ga_results = load_json('ga_optimization_results.json')

# --- Build Report ---
report = []
report.append("# ML-Based Adsorption Capacity Prediction — Project Report")
report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append("---\n")

# 1. Objective
report.append("## 1. Objective\n")
report.append("""This project applies machine learning methodologies inspired by Research Paper 1 to predict the adsorption capacity of emerging contaminants (ECs) on biochar materials. Three ML models — Gaussian Process Regression (GPR), Support Vector Regression (SVR), and Artificial Neural Network (ANN) — are trained, tuned, and compared. Additionally, a Genetic Algorithm (GA) is employed to optimize experimental input parameters for maximum adsorption capacity.\n""")

# 2. Dataset Description
report.append("## 2. Dataset Description\n")
if dataset_stats:
    report.append(f"- **Total samples:** {dataset_stats['total_samples']}")
    report.append(f"- **Training samples:** {dataset_stats['train_samples']} (80%)")
    report.append(f"- **Testing samples:** {dataset_stats['test_samples']} (20%)")
    report.append(f"- **Missing values dropped:** {dataset_stats['missing_values_dropped']}")
    report.append(f"- **Target variable:** {dataset_stats['target']}\n")
    report.append("### Feature Statistics\n")
    report.append("| Feature | Mean | Std | Min | Max |")
    report.append("|---------|------|-----|-----|-----|")
    for feat, stats in dataset_stats['feature_stats'].items():
        report.append(f"| {feat} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
    report.append("")
    ts = dataset_stats['target_stats']
    report.append(f"**Target (Adsorption Capacity):** Mean={ts['mean']:.4f}, Std={ts['std']:.4f}, Min={ts['min']:.4f}, Max={ts['max']:.4f}\n")

# 3. Methodology
report.append("## 3. Methodology\n")
report.append("### 3.1 Data Preprocessing\n")
report.append("- Loaded cleaned dataset with 5 input features and 1 target variable")
report.append("- Missing values were removed (if any)")
report.append("- Features standardized using Z-score normalization (StandardScaler)")
report.append("- Data split: 80% training / 20% testing (random_state=42)\n")

report.append("### 3.2 Gaussian Process Regression (GPR)\n")
report.append("- **Kernel:** ConstantKernel × RBF + WhiteKernel (noise)")
report.append("- Hyperparameters optimized via marginal likelihood maximization")
report.append("- 5 random restarts for optimizer robustness")
if gpr_results:
    report.append(f"- Training samples used: {gpr_results.get('training_samples', 'N/A')} (subsampled for O(n³) tractability)")
    report.append(f"- Optimized kernel: `{gpr_results.get('kernel', 'N/A')}`\n")

report.append("### 3.3 Support Vector Regression (SVR)\n")
report.append("- **Kernels tested:** RBF and Linear")
report.append("- **Hyperparameters tuned:** C, gamma, epsilon")
report.append("- **Tuning method:** GridSearchCV with 5-fold cross-validation")
if svr_results:
    report.append(f"- Best kernel: {svr_results.get('best_kernel', 'N/A')}")
    report.append(f"- Best parameters: {svr_results.get('best_params', 'N/A')}\n")

report.append("### 3.4 Artificial Neural Network (ANN)\n")
report.append("- **Framework:** TensorFlow/Keras")
report.append("- **Architecture search:** 3 architectures × 2 learning rates = 6 configurations")
report.append("- Dense layers with ReLU activation, BatchNormalization, and Dropout(0.2)")
report.append("- L2 regularization (1e-4)")
report.append("- Early stopping (patience=20) and ReduceLROnPlateau")
if ann_results:
    bc = ann_results.get('best_config', {})
    report.append(f"- Best config: {bc.get('name', 'N/A')}")
    report.append(f"- Hidden layers: {bc.get('hidden_layers', 'N/A')}")
    report.append(f"- Learning rate: {bc.get('lr', 'N/A')}")
    report.append(f"- Epochs trained: {bc.get('epochs', 'N/A')}\n")

# 4. Results
report.append("## 4. Results Comparison\n")
report.append("### Test Set Performance\n")
report.append("| Model | R² | MAE | RMSE | Training Time (s) |")
report.append("|-------|-----|-----|------|-------------------|")

all_results = [('GPR', gpr_results), ('SVR', svr_results), ('ANN', ann_results)]
for name, res in all_results:
    if res:
        m = res['test_metrics']
        tt = res.get('training_time_seconds', 'N/A')
        report.append(f"| {name} | {m['R2']:.4f} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {tt} |")

# Find best model
valid_results = [(n, r) for n, r in all_results if r]
if valid_results:
    best_name, best_res = max(valid_results, key=lambda x: x[1]['test_metrics']['R2'])
    bm = best_res['test_metrics']
    report.append(f"\n**Best Model: {best_name}** with R²={bm['R2']:.4f}, MAE={bm['MAE']:.4f}, RMSE={bm['RMSE']:.4f}\n")

# Training metrics
report.append("### Training Set Performance\n")
report.append("| Model | R² | MAE | RMSE |")
report.append("|-------|-----|-----|------|")
for name, res in all_results:
    if res:
        m = res['train_metrics']
        report.append(f"| {name} | {m['R2']:.4f} | {m['MAE']:.4f} | {m['RMSE']:.4f} |")
report.append("")

# 5. Genetic Algorithm
if ga_results:
    report.append("## 5. Genetic Algorithm Optimization\n")
    report.append(f"- **Fitness model used:** {ga_results['fitness_model']}")
    report.append(f"- **Population size:** {ga_results['ga_params']['pop_size']}")
    report.append(f"- **Generations:** {ga_results['ga_params']['generations']}")
    report.append(f"- **Runtime:** {ga_results['runtime_seconds']}s\n")
    report.append("### Optimized Experimental Conditions\n")
    report.append("| Parameter | Optimized Value |")
    report.append("|-----------|----------------|")
    for param, val in ga_results['optimized_conditions'].items():
        report.append(f"| {param} | {val:.4f} |")
    report.append(f"\n**Predicted Maximum Adsorption Capacity:** {ga_results['predicted_max_capacity']:.4f}\n")

# 6. Plots
report.append("## 6. Generated Plots\n")
report.append("The following visualizations are saved in the `/plots` directory:\n")
report.append("1. `predicted_vs_actual.png` — Scatter plots of predicted vs actual values for each model")
report.append("2. `error_distribution.png` — Error distribution histograms for each model")
report.append("3. `model_comparison.png` — Bar charts comparing R², MAE, and RMSE across models")
report.append("4. `ann_training_curve.png` — ANN training and validation loss over epochs")
report.append("5. `results_summary.png` — Combined summary dashboard")
report.append("6. `ga_convergence.png` — Genetic algorithm convergence curve\n")

# 7. Key Observations
report.append("## 7. Key Observations\n")
if valid_results:
    r2_values = {n: r['test_metrics']['R2'] for n, r in valid_results}
    sorted_models = sorted(r2_values.items(), key=lambda x: x[1], reverse=True)
    report.append(f"1. **{sorted_models[0][0]}** achieved the highest test R² of {sorted_models[0][1]:.4f}, making it the best performer for this dataset.")
    if len(sorted_models) > 1:
        report.append(f"2. **{sorted_models[1][0]}** ranked second with R² = {sorted_models[1][1]:.4f}.")
    if len(sorted_models) > 2:
        report.append(f"3. **{sorted_models[2][0]}** had the lowest performance with R² = {sorted_models[2][1]:.4f}.")

    # Check overfitting
    for name, res in valid_results:
        train_r2 = res['train_metrics']['R2']
        test_r2 = res['test_metrics']['R2']
        gap = train_r2 - test_r2
        if gap > 0.1:
            report.append(f"4. **Overfitting concern in {name}:** Training R²={train_r2:.4f} vs Test R²={test_r2:.4f} (gap={gap:.4f}).")

if ga_results:
    report.append(f"5. The Genetic Algorithm identified optimal conditions predicting a maximum capacity of **{ga_results['predicted_max_capacity']:.4f}**.")

report.append("\n---")
report.append(f"\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# --- Write report ---
report_content = "\n".join(report)
report_path = os.path.join(REPORT_DIR, 'project_report.md')
with open(report_path, 'w') as f:
    f.write(report_content)
print(f"\n[INFO] Report saved to: {report_path}")
print(f"[INFO] Report length: {len(report_content)} characters, {len(report)} lines")

print("\n" + "=" * 60)
print("REPORT GENERATION COMPLETE ✓")
print("=" * 60)
