# Model Results

All metrics are on the held-out test set (20% split, `random_state=42`).
Metrics: R² (higher is better), MAE (mg/g, lower is better), RMSE (mg/g, lower is better).

---

## PAC (Powdered Activated Carbon)

Dataset: 162 total | 129 train | 33 test | Capacity range: 0 – 385.33 mg/g | Mean: 98.42 mg/g

| Model | Test R² | Test MAE | Test RMSE | Training Time |
|-------|---------|----------|-----------|--------------|
| **GPR** | **0.7715** | **27.34** | **34.24** | 0.43 s |
| SVR | 0.7540 | 29.32 | 35.53 | 2.74 s |
| ANN | 0.7149 | 31.29 | 38.25 | 18.13 s |

**Best model: GPR** (marginally ahead of SVR on R² and MAE)

SVR best hyperparameters: `kernel=linear`, `C=10`, `epsilon=0.01`, `gamma=scale`

ANN architecture: Dense(64) → BN → Dropout(0.2) → Dense(32) → BN → Dropout(0.2) → Dense(16) → Dense(1)

---

## PB600 (Biochar at 600 °C)

Dataset: 162 total | 129 train | 33 test | Capacity range: −4.46 – 15.86 mg/g | Mean: 5.02 mg/g

| Model | Test R² | Test MAE | Test RMSE | Training Time |
|-------|---------|----------|-----------|--------------|
| GPR | 0.7357 | 1.15 | 1.49 | 0.40 s |
| SVR | 0.6030 | 1.20 | 1.83 | 0.85 s |
| **ANN** | **0.7551** | **1.08** | **1.43** | 10.99 s |

**Best model: ANN** (R² = 0.755, best MAE and RMSE)

SVR best hyperparameters: `kernel=rbf`, `C=100`, `epsilon=0.01`, `gamma=scale`

Note: The narrow capacity range (≈20 mg/g span) makes absolute MAE/RMSE values small, but the
relative difficulty is comparable — a ±1.4 RMSE on a 0–16 range is meaningful.

---

## NaOH-activated SCW Biochars

Dataset: 486 total | 388 train | 98 test | Capacity range: 0 – 267.76 mg/g | Mean: 105.45 mg/g

| Model | Test R² | Test MAE | Test RMSE | Training Time |
|-------|---------|----------|-----------|--------------|
| GPR | −0.0918 | 47.37 | 55.81 | 3.55 s |
| SVR | 0.0781 | 36.83 | 51.29 | 0.45 s |
| **ANN** | **0.1757** | **43.19** | **48.49** | 10.81 s |

**Best model: ANN** — though all three models perform poorly on this biochar.

### Why NaOH-SCW is hard

- Largest dataset (486 rows vs. 162 for the others), but GPR's O(n³) kernel inversion
  becomes computationally strained and the optimised kernel may not generalise well.
- SVR with a linear kernel (best CV result) also struggles — the adsorption relationship
  for this chemically activated biochar may be strongly non-linear in 5 features alone,
  without the adsorbent characterisation features used in Phase 1.
- ANN achieves a positive R² but still only explains 17.6% of variance on the test set.
  More data or additional adsorbent-specific features are likely needed.

---

## Cross-Biochar Model Comparison

| Biochar | Best Model | Best Test R² |
|---------|-----------|-------------|
| PAC | GPR | 0.7715 |
| PB600 | ANN | 0.7551 |
| NaOH-activated SCW | ANN | 0.1757 |

GPR is excellent for smaller, balanced datasets. ANN edges ahead when the relationship
is more complex and a larger training set is available. SVR provides fast, competitive
results — especially with a linear kernel on PAC — and requires minimal tuning time.

---

## Phase 1 vs Phase 2 Comparison

| Phase | Dataset | Best Model | Best R² |
|-------|---------|-----------|--------|
| Phase 1 (all biochars, full features) | 3,757 rows, ~20+ features | CatBoost | 0.9433 |
| Phase 2 PAC (5 features only) | 162 rows | GPR | 0.7715 |
| Phase 2 PB600 (5 features only) | 162 rows | ANN | 0.7551 |
| Phase 2 NaOH (5 features only) | 486 rows | ANN | 0.1757 |

Phase 1 outperforms Phase 2 because it uses the full feature set (including adsorbent
composition and characterisation). Phase 2 intentionally restricts features to operational
conditions only — making it useful for process optimisation without needing material analysis.
