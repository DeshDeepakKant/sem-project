"""
Step 6: Genetic Algorithm Optimization
- Use GA to find optimal input parameters for maximum adsorption capacity
- Optimize: pH, Temperature, Contact time, Initial concentration, Adsorbent dosage
- Uses the best-performing model as the fitness function
"""

import numpy as np
import pickle
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("=" * 60)
print("STEP 6: GENETIC ALGORITHM (GA) OPTIMIZATION")
print("=" * 60)

# --- Load scaler and data ---
with open(os.path.join(DATA_DIR, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

feature_cols = data['feature_cols']
X_train_raw = data['X_train_raw']

# --- Determine best model ---
model_scores = {}
for name, result_file in [('GPR', 'gpr_results.json'), ('SVR', 'svr_results.json'), ('ANN', 'ann_results.json')]:
    path = os.path.join(RESULTS_DIR, result_file)
    if os.path.exists(path):
        with open(path, 'r') as f:
            r = json.load(f)
        model_scores[name] = r['test_metrics']['R2']
        print(f"[INFO] {name} test R²: {r['test_metrics']['R2']:.4f}")

best_model_name = max(model_scores, key=model_scores.get)
print(f"\n[INFO] Best model for GA fitness: {best_model_name} (R²={model_scores[best_model_name]:.4f})")

# Load the best model
if best_model_name == 'GPR':
    with open(os.path.join(MODELS_DIR, 'gpr_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    def predict_fn(X_scaled): return model.predict(X_scaled)
elif best_model_name == 'SVR':
    with open(os.path.join(MODELS_DIR, 'svr_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    def predict_fn(X_scaled): return model.predict(X_scaled)
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'ann_model.keras'))
    def predict_fn(X_scaled): return model.predict(X_scaled, verbose=0).flatten()

# --- Define search bounds from training data ---
bounds = {}
for i, col in enumerate(feature_cols):
    col_min = float(X_train_raw[:, i].min())
    col_max = float(X_train_raw[:, i].max())
    bounds[col] = (col_min, col_max)
    print(f"[INFO] {col}: [{col_min:.4f}, {col_max:.4f}]")

# ============================================================
# GENETIC ALGORITHM IMPLEMENTATION
# ============================================================

class GeneticAlgorithm:
    def __init__(self, fitness_fn, bounds, pop_size=100, generations=200,
                 mutation_rate=0.15, crossover_rate=0.8, elite_frac=0.1):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.n_vars = len(bounds)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = max(2, int(pop_size * elite_frac))

        self.lower = np.array([b[0] for b in bounds.values()])
        self.upper = np.array([b[1] for b in bounds.values()])

    def initialize(self):
        """Random initialization within bounds."""
        return np.random.uniform(self.lower, self.upper, (self.pop_size, self.n_vars))

    def evaluate(self, population):
        """Evaluate fitness for entire population."""
        X_scaled = scaler.transform(population)
        return self.fitness_fn(X_scaled)

    def select(self, population, fitness):
        """Tournament selection."""
        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(population), 2, replace=False)
            winner = i if fitness[i] > fitness[j] else j
            selected.append(population[winner])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """BLX-alpha crossover."""
        alpha = 0.5
        if np.random.rand() < self.crossover_rate:
            gamma = (1 + 2 * alpha) * np.random.rand(self.n_vars) - alpha
            child1 = parent1 + gamma * (parent2 - parent1)
            child2 = parent2 + gamma * (parent1 - parent2)
            return np.clip(child1, self.lower, self.upper), np.clip(child2, self.lower, self.upper)
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Gaussian mutation."""
        if np.random.rand() < self.mutation_rate:
            mutation_strength = 0.1 * (self.upper - self.lower)
            noise = np.random.randn(self.n_vars) * mutation_strength
            individual = np.clip(individual + noise, self.lower, self.upper)
        return individual

    def run(self):
        """Run the GA."""
        population = self.initialize()
        best_fitness_history = []

        for gen in range(self.generations):
            fitness = self.evaluate(population)

            # Track best
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx].copy()
            best_fitness_history.append(float(best_fitness))

            if (gen + 1) % 50 == 0 or gen == 0:
                print(f"  Generation {gen+1:>3d} | Best Fitness (Capacity): {best_fitness:.4f} | "
                      f"Mean: {np.mean(fitness):.4f}")

            # Elitism
            elite_idx = np.argsort(fitness)[-self.elite_size:]
            elites = population[elite_idx].copy()

            # Selection
            selected = self.select(population, fitness)

            # Crossover + Mutation
            new_pop = list(elites)
            for i in range(0, self.pop_size - self.elite_size, 2):
                p1 = selected[np.random.randint(len(selected))]
                p2 = selected[np.random.randint(len(selected))]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = np.array(new_pop[:self.pop_size])

        return best_individual, best_fitness, best_fitness_history

# --- Run GA ---
print("\n[INFO] Running Genetic Algorithm...")
start_time = time.time()

ga = GeneticAlgorithm(
    fitness_fn=predict_fn,
    bounds=bounds,
    pop_size=100,
    generations=200,
    mutation_rate=0.15,
    crossover_rate=0.8
)
best_params, best_capacity, fitness_history = ga.run()

ga_time = time.time() - start_time
print(f"\n[INFO] GA completed in {ga_time:.2f} seconds.")

# --- Display optimized conditions ---
print("\n" + "=" * 50)
print("OPTIMIZED CONDITIONS FOR MAXIMUM ADSORPTION CAPACITY")
print("=" * 50)
for col, val in zip(feature_cols, best_params):
    print(f"  {col:>25s}: {val:.4f}")
print(f"  {'Predicted Capacity':>25s}: {best_capacity:.4f}")

# --- Save GA results ---
ga_results = {
    'optimizer': 'Genetic Algorithm',
    'fitness_model': best_model_name,
    'optimized_conditions': {col: float(val) for col, val in zip(feature_cols, best_params)},
    'predicted_max_capacity': float(best_capacity),
    'ga_params': {
        'pop_size': 100,
        'generations': 200,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8
    },
    'runtime_seconds': round(ga_time, 2),
    'fitness_history': fitness_history
}

results_path = os.path.join(RESULTS_DIR, 'ga_optimization_results.json')
with open(results_path, 'w') as f:
    json.dump(ga_results, f, indent=2)
print(f"\n[INFO] GA results saved to: {results_path}")

# --- GA convergence plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', linewidth=1.5)
ax.set_xlabel('Generation')
ax.set_ylabel('Best Fitness (Predicted Capacity)')
ax.set_title('Genetic Algorithm Convergence')
ax.grid(True, alpha=0.3)

PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ga_convergence.png'), bbox_inches='tight')
plt.close()
print(f"[INFO] GA convergence plot saved to: plots/ga_convergence.png")

print("\n" + "=" * 60)
print("GENETIC ALGORITHM OPTIMIZATION COMPLETE ✓")
print("=" * 60)
