"""
Step 02: Train GPR, SVR, ANN for each of the 3 biochars separately.
"""
import numpy as np
import pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

BIOCHARS = {'PAC': 'PAC', 'PB600': 'PB600', 'NaOH-activated SCW biochars': 'NaOH_activated_SCW'}

def calc_metrics(y_true, y_pred):
    return {
        'R2': round(float(r2_score(y_true, y_pred)), 4),
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    }

def train_gpr(X_train, y_train, X_test, y_test):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train)
    y_pred_train = gpr.predict(X_train)
    y_pred_test, y_std_test = gpr.predict(X_test, return_std=True)
    return gpr, y_pred_train, y_pred_test, y_std_test

def train_svr(X_train, y_train, X_test, y_test):
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.5]
    }
    grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    svr = grid.best_estimator_
    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)
    return svr, y_pred_train, y_pred_test, grid.best_params_

def train_ann(X_train, y_train, X_test, y_test):
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0,
                        validation_split=0.2,
                        callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    return model, y_pred_train, y_pred_test, history.history

# === Main ===
print("=" * 60)
print("STEP 2: TRAIN ALL MODELS (Per-Biochar)")
print("=" * 60)

for biochar, safe in BIOCHARS.items():
    print(f"\n{'='*60}")
    print(f"  BIOCHAR: {biochar}")
    print(f"{'='*60}")

    with open(os.path.join(DATA_DIR, f'processed_{safe}.pkl'), 'rb') as f:
        data = pickle.load(f)

    X_tr, X_te = data['X_train_scaled'], data['X_test_scaled']
    y_tr, y_te = data['y_train'], data['y_test']
    os.makedirs(os.path.join(MODELS_DIR, safe), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, safe), exist_ok=True)

    # --- GPR ---
    print(f"\n  [GPR] Training...")
    t0 = time.time()
    gpr, gpr_train, gpr_test, gpr_std = train_gpr(X_tr, y_tr, X_te, y_te)
    gpr_time = time.time() - t0
    with open(os.path.join(MODELS_DIR, safe, 'gpr_model.pkl'), 'wb') as f:
        pickle.dump(gpr, f)
    gpr_res = {'train_metrics': calc_metrics(y_tr, gpr_train), 'test_metrics': calc_metrics(y_te, gpr_test),
               'training_time_seconds': round(gpr_time, 2)}
    with open(os.path.join(RESULTS_DIR, safe, 'gpr_results.json'), 'w') as f:
        json.dump(gpr_res, f, indent=2)
    pickle.dump({'y_test': y_te, 'y_pred_test': gpr_test, 'y_pred_train': gpr_train, 'y_train': y_tr,
                 'y_std_test': gpr_std},
                open(os.path.join(RESULTS_DIR, safe, 'gpr_predictions.pkl'), 'wb'))
    print(f"  [GPR] R²={gpr_res['test_metrics']['R2']}, MAE={gpr_res['test_metrics']['MAE']} ({gpr_time:.1f}s)")

    # --- SVR ---
    print(f"  [SVR] Training...")
    t0 = time.time()
    svr, svr_train, svr_test, svr_params = train_svr(X_tr, y_tr, X_te, y_te)
    svr_time = time.time() - t0
    with open(os.path.join(MODELS_DIR, safe, 'svr_model.pkl'), 'wb') as f:
        pickle.dump(svr, f)
    svr_res = {'train_metrics': calc_metrics(y_tr, svr_train), 'test_metrics': calc_metrics(y_te, svr_test),
               'best_params': {k: str(v) for k, v in svr_params.items()}, 'training_time_seconds': round(svr_time, 2)}
    with open(os.path.join(RESULTS_DIR, safe, 'svr_results.json'), 'w') as f:
        json.dump(svr_res, f, indent=2)
    pickle.dump({'y_test': y_te, 'y_pred_test': svr_test, 'y_pred_train': svr_train, 'y_train': y_tr},
                open(os.path.join(RESULTS_DIR, safe, 'svr_predictions.pkl'), 'wb'))
    print(f"  [SVR] R²={svr_res['test_metrics']['R2']}, MAE={svr_res['test_metrics']['MAE']} ({svr_time:.1f}s) Best: {svr_params}")

    # --- ANN ---
    print(f"  [ANN] Training...")
    t0 = time.time()
    ann, ann_train, ann_test, ann_hist = train_ann(X_tr, y_tr, X_te, y_te)
    ann_time = time.time() - t0
    ann.save(os.path.join(MODELS_DIR, safe, 'ann_model.keras'))
    ann_res = {'train_metrics': calc_metrics(y_tr, ann_train), 'test_metrics': calc_metrics(y_te, ann_test),
               'training_time_seconds': round(ann_time, 2),
               'history': {k: [float(v) for v in vals] for k, vals in ann_hist.items()}}
    with open(os.path.join(RESULTS_DIR, safe, 'ann_results.json'), 'w') as f:
        json.dump(ann_res, f, indent=2)
    pickle.dump({'y_test': y_te, 'y_pred_test': ann_test, 'y_pred_train': ann_train, 'y_train': y_tr},
                open(os.path.join(RESULTS_DIR, safe, 'ann_predictions.pkl'), 'wb'))
    print(f"  [ANN] R²={ann_res['test_metrics']['R2']}, MAE={ann_res['test_metrics']['MAE']} ({ann_time:.1f}s)")

print("\n[INFO] All models trained ✓")
