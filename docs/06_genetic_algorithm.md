# Genetic Algorithm Optimisation

## Purpose

Once the best ML model per biochar is identified, a Genetic Algorithm (GA) is used to find
the combination of operating conditions that **maximises predicted adsorption capacity**.
This replaces exhaustive grid search with an evolutionary search that converges efficiently
in high-dimensional spaces.

## Model Selection for Fitness Function

Before running GA, the script reads the test R² from all three saved result JSON files
and selects the best-performing model for that biochar as the fitness evaluator:

```
PAC:            → SVR  (R² = 0.754)   [slightly preferred over GPR in the GA run's saved result]
PB600:          → ANN  (R² = 0.755)
NaOH-SCW:       → ANN  (R² = 0.176)
```

The selected model's predictions are used as the fitness score — the GA maximises the
model's predicted adsorption capacity.

## GA Design

**Script:** `code/04_genetic_algorithm.py`

| Parameter | Value |
|-----------|-------|
| Population size | 80 |
| Generations | 100 |
| Mutation rate | 0.12 (12%) |
| Crossover type | BLX-α (blend crossover, α = 0.5) |
| Selection | Tournament selection |
| Elitism fraction | 10% of population preserved each generation |
| Mutation type | Gaussian perturbation (10% of feature range) |

**Search space bounds** are derived directly from the training data's min/max for each feature,
keeping the GA inside the data distribution.

## Implementation Details

```python
class GeneticAlgorithm:
    def initialize(self):     # Uniform random within [lower, upper]
    def evaluate(self):       # Scale inputs → predict capacity
    def select(self):         # Tournament selection (pair-wise competition)
    def crossover(self, p1, p2):   # BLX-α blend crossover
    def mutate(self, individual):  # Gaussian noise clipped to bounds
    def run(self):            # Main loop with elitism
```

The scaler fitted during preprocessing is applied inside `evaluate()` before prediction,
ensuring the model always receives properly standardised inputs.

## Optimised Results

### PAC — Fitness model: SVR

| Feature | Optimal Value |
|---------|--------------|
| pH | 3.12 |
| Temperature | 43.4 °C |
| Contact time | 30 min |
| Initial concentration | 97.9 mg/L |
| Adsorbent dosage | 0.05 g/L |
| **Predicted max capacity** | **355.40 mg/g** |

Low pH (acidic) and moderately high temperature favour adsorption. Short contact time reaching
high capacity suggests fast kinetics for PAC. High initial concentration drives the concentration
gradient.

---

### PB600 — Fitness model: ANN

| Feature | Optimal Value |
|---------|--------------|
| pH | 7.0 |
| Temperature | 22.8 °C |
| Contact time | 997 min |
| Initial concentration | 6.9 mg/L |
| Adsorbent dosage | 0.117 g/L |
| **Predicted max capacity** | **11.66 mg/g** |

Neutral pH, near-ambient temperature, and long contact time are optimal for PB600.
Low initial concentration suggests this material operates best under dilute conditions.

---

### NaOH-activated SCW — Fitness model: ANN

| Feature | Optimal Value |
|---------|--------------|
| pH | 8.59 |
| Temperature | 21.6 °C |
| Contact time | 1095 min |
| Initial concentration | 29.6 mg/L |
| Adsorbent dosage | 0.15 g/L |
| **Predicted max capacity** | **125.48 mg/g** |

Slightly alkaline pH and long contact time are optimal. The high maximum capacity compared to
PB600 reflects the chemical activation enhancing surface functional groups. Note: model R² for
this biochar is low (0.176), so these values carry higher uncertainty than for PAC and PB600.

---

## GA Convergence Plots

Saved to `plots/per_biochar/<safe>_ga_convergence.png` for each biochar. Shows best fitness
(predicted capacity) vs. generation number. Convergence is typically achieved within the first
30–50 generations.

## Limitations

- GA optimisation is bounded by the training data range. It cannot recommend conditions outside
  the experimental space covered by the dataset.
- For NaOH-SCW the underlying ANN has low accuracy (R² ≈ 0.18), so the optimised conditions
  should be treated as exploratory rather than reliable recommendations.
- The GA is stochastic — rerunning may produce slightly different results. Setting `random_state`
  in numpy before initialisation would make it deterministic.
