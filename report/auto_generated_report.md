# ML-Based Adsorption Capacity Prediction — Project Report

**Generated:** 2026-05-12 22:34:10

---

## 1. Objective

This project applies machine learning methodologies inspired by Research Paper 1 to predict the adsorption capacity of emerging contaminants (ECs) on biochar materials. Three ML models — Gaussian Process Regression (GPR), Support Vector Regression (SVR), and Artificial Neural Network (ANN) — are trained, tuned, and compared. Additionally, a Genetic Algorithm (GA) is employed to optimize experimental input parameters for maximum adsorption capacity.

## 2. Dataset Description

- **Total samples:** 3757
- **Training samples:** 3005 (80%)
- **Testing samples:** 752 (20%)
- **Missing values dropped:** 0
- **Target variable:** Adsorption capacity

### Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| pH | 7.0394 | 1.0208 | 3.0000 | 11.0000 |
| Temperature | 25.2076 | 3.7222 | 15.0000 | 45.0000 |
| Contact time | 996.8560 | 584.8069 | 0.0000 | 1455.0000 |
| Initial concentration | 13.6151 | 13.9492 | 0.5000 | 100.0000 |
| Adsorbent dosage | 0.0801 | 0.0443 | 0.0200 | 0.1600 |

**Target (Adsorption Capacity):** Mean=53.4259, Std=73.0936, Min=-31.7910, Max=536.3388

## 3. Methodology

### 3.1 Data Preprocessing

- Loaded cleaned dataset with 5 input features and 1 target variable
- Missing values were removed (if any)
- Features standardized using Z-score normalization (StandardScaler)
- Data split: 80% training / 20% testing (random_state=42)

### 3.2 Gaussian Process Regression (GPR)

- **Kernel:** ConstantKernel × RBF + WhiteKernel (noise)
- Hyperparameters optimized via marginal likelihood maximization
- 5 random restarts for optimizer robustness
- Training samples used: 1500 (subsampled for O(n³) tractability)
- Optimized kernel: `1.62**2 * RBF(length_scale=3.81) + WhiteKernel(noise_level=0.639)`

### 3.3 Support Vector Regression (SVR)

- **Kernels tested:** RBF and Linear
- **Hyperparameters tuned:** C, gamma, epsilon
- **Tuning method:** GridSearchCV with 5-fold cross-validation
- Best kernel: RBF
- Best parameters: {'C': '100', 'epsilon': '0.5', 'gamma': '0.01'}

### 3.4 Artificial Neural Network (ANN)

- **Framework:** TensorFlow/Keras
- **Architecture search:** 3 architectures × 2 learning rates = 6 configurations
- Dense layers with ReLU activation, BatchNormalization, and Dropout(0.2)
- L2 regularization (1e-4)
- Early stopping (patience=20) and ReduceLROnPlateau
- Best config: ANN_3H_lr0.001
- Hidden layers: [128, 64, 32]
- Learning rate: 0.001
- Epochs trained: 74

## 4. Results Comparison

### Test Set Performance

| Model | R² | MAE | RMSE | Training Time (s) |
|-------|-----|-----|------|-------------------|
| GPR | 0.3461 | 38.3699 | 55.0474 | 53.66 |
| SVR | 0.2079 | 36.8411 | 60.5872 | 14.65 |
| ANN | 0.3512 | 37.3554 | 54.8316 | 13.48 |

**Best Model: ANN** with R²=0.3512, MAE=37.3554, RMSE=54.8316

### Training Set Performance

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| GPR | 0.3613 | 40.7744 | 59.9102 |
| SVR | 0.2054 | 40.0847 | 66.1827 |
| ANN | 0.3628 | 40.1888 | 59.2630 |

## 5. Genetic Algorithm Optimization

- **Fitness model used:** ANN
- **Population size:** 100
- **Generations:** 200
- **Runtime:** 12.9s

### Optimized Experimental Conditions

| Parameter | Optimized Value |
|-----------|----------------|
| pH | 3.0000 |
| Temperature | 21.4771 |
| Contact time | 0.0000 |
| Initial concentration | 100.0000 |
| Adsorbent dosage | 0.0200 |

**Predicted Maximum Adsorption Capacity:** 345.7672

## 6. Generated Plots

The following visualizations are saved in the `/plots` directory:

1. `predicted_vs_actual.png` — Scatter plots of predicted vs actual values for each model
2. `error_distribution.png` — Error distribution histograms for each model
3. `model_comparison.png` — Bar charts comparing R², MAE, and RMSE across models
4. `ann_training_curve.png` — ANN training and validation loss over epochs
5. `results_summary.png` — Combined summary dashboard
6. `ga_convergence.png` — Genetic algorithm convergence curve

## 7. Key Observations

1. **ANN** achieved the highest test R² of 0.3512, making it the best performer for this dataset.
2. **GPR** ranked second with R² = 0.3461.
3. **SVR** had the lowest performance with R² = 0.2079.
5. The Genetic Algorithm identified optimal conditions predicting a maximum capacity of **345.7672**.

---

*Report generated on 2026-05-12 22:34:10*