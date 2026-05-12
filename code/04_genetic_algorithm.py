"""
Step 04: Genetic Algorithm optimization — find optimal conditions per biochar.
"""
import numpy as np
import pickle, os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

import matplotlib.pyplot as plt

BIOCHARS = {'PAC': 'PAC', 'PB600': 'PB600', 'NaOH-activated SCW biochars': 'NaOH_activated_SCW'}
FEATURES = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']

print("=" * 60)
print("STEP 4: GENETIC ALGORITHM OPTIMIZATION")
print("=" * 60)

def genetic_algorithm(fitness_fn, bounds, pop_size=100, generations=150, mutation_rate=0.1):
    n_vars = len(bounds)
    pop = np.random.rand(pop_size, n_vars)
    for i in range(n_vars):
        pop[:, i] = pop[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    best_history = []
    for gen in range(generations):
        fitness = np.array([fitness_fn(ind) for ind in pop])
        best_idx = np.argmax(fitness)
        best_history.append(float(fitness[best_idx]))

        # Selection (tournament)
        selected = []
        for _ in range(pop_size):
            i, j = np.random.randint(0, pop_size, 2)
            selected.append(pop[i] if fitness[i] > fitness[j] else pop[j])
        selected = np.array(selected)

        # Crossover
        children = []
        for k in range(0, pop_size, 2):
            p1, p2 = selected[k], selected[min(k+1, pop_size-1)]
            alpha = np.random.rand()
            children.append(alpha * p1 + (1 - alpha) * p2)
            children.append((1 - alpha) * p1 + alpha * p2)
        pop = np.array(children[:pop_size])

        # Mutation
        for k in range(pop_size):
            if np.random.rand() < mutation_rate:
                gene = np.random.randint(0, n_vars)
                pop[k, gene] = np.random.uniform(bounds[gene][0], bounds[gene][1])

        # Elitism
        pop[0] = selected[best_idx]

    best_fitness = np.array([fitness_fn(ind) for ind in pop])
    best = pop[np.argmax(best_fitness)]
    return best, max(best_fitness), best_history

for biochar, safe in BIOCHARS.items():
    print(f"\n--- GA Optimization: {biochar} ---")

    with open(os.path.join(DATA_DIR, f'processed_{safe}.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(DATA_DIR, f'scaler_{safe}.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Find best model
    best_model_name, best_r2 = None, -999
    for model_name in ['GPR', 'SVR', 'ANN']:
        res_path = os.path.join(RESULTS_DIR, safe, f'{model_name.lower()}_results.json')
        with open(res_path) as f:
            r2 = json.load(f)['test_metrics']['R2']
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name

    print(f"  Using best model: {best_model_name} (R²={best_r2:.4f})")

    if best_model_name == 'ANN':
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, safe, 'ann_model.keras'))
        def predict_fn(x):
            x_s = scaler.transform(x.reshape(1, -1))
            return float(model.predict(x_s, verbose=0).flatten()[0])
    else:
        with open(os.path.join(MODELS_DIR, safe, f'{best_model_name.lower()}_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        def predict_fn(x):
            x_s = scaler.transform(x.reshape(1, -1))
            return float(model.predict(x_s)[0] if best_model_name == 'SVR' else model.predict(x_s))

    # Bounds from data
    X = data['X_train']
    bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

    best_params, best_capacity, history = genetic_algorithm(predict_fn, bounds)

    result = {
        'biochar': biochar,
        'fitness_model': best_model_name,
        'predicted_max_capacity': round(float(best_capacity), 4),
        'optimized_conditions': {FEATURES[i]: round(float(best_params[i]), 4) for i in range(len(FEATURES))},
        'ga_params': {'population': 100, 'generations': 150, 'mutation_rate': 0.1}
    }
    with open(os.path.join(RESULTS_DIR, safe, 'ga_results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Max Capacity: {best_capacity:.2f}")
    print(f"  Optimal: {result['optimized_conditions']}")

    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history, color='#e74c3c', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (Predicted Capacity)')
    ax.set_title(f'GA Convergence — {biochar}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_ga_convergence.png'))
    plt.close()

print("\n[INFO] GA optimization complete ✓")
