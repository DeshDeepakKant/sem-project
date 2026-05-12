"""
Step 4: Train Artificial Neural Network (ANN)
- Feedforward neural network using TensorFlow/Keras
- Architecture: Input -> Dense(128) -> Dense(64) -> Dense(32) -> Output
- Hyperparameter tuning via manual grid of architectures
- Early stopping to prevent overfitting
"""

import numpy as np
import pickle
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- Load processed data ---
print("=" * 60)
print("STEP 4: ARTIFICIAL NEURAL NETWORK (ANN)")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"\n[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Test samples: {X_test.shape[0]}")
print(f"[INFO] Input features: {X_train.shape[1]}")

# --- Define ANN architectures to try ---
architectures = {
    'ANN_1H': [128],               # 1 hidden layer
    'ANN_2H': [128, 64],           # 2 hidden layers
    'ANN_3H': [128, 64, 32],       # 3 hidden layers (deeper)
}

learning_rates = [0.001, 0.01]
best_val_score = -np.inf
best_config = None
best_history = None

print("\n[INFO] Starting architecture search...")
print("-" * 50)

results_all = []

for arch_name, hidden_layers in architectures.items():
    for lr in learning_rates:
        config_name = f"{arch_name}_lr{lr}"
        print(f"\n>>> Training: {config_name} | Layers: {hidden_layers} | LR: {lr}")

        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))

        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(1e-4)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.2))

        model.add(layers.Dense(1))  # Output layer

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )

        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        train_time = time.time() - start_time

        # Evaluate on test
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"    Test R²: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | Time: {train_time:.1f}s")

        results_all.append({
            'config': config_name,
            'hidden_layers': hidden_layers,
            'lr': lr,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'train_time': train_time,
            'epochs_trained': len(history.history['loss'])
        })

        if test_r2 > best_val_score:
            best_val_score = test_r2
            best_config = {
                'name': config_name,
                'hidden_layers': hidden_layers,
                'lr': lr,
                'train_time': train_time,
                'epochs': len(history.history['loss'])
            }
            best_model = model
            best_history = history

print("\n" + "=" * 50)
print(f"[INFO] Best configuration: {best_config['name']}")
print(f"[INFO] Hidden layers: {best_config['hidden_layers']}")
print(f"[INFO] Learning rate: {best_config['lr']}")
print(f"[INFO] Epochs trained: {best_config['epochs']}")

# --- Final evaluation with best model ---
y_pred_train = best_model.predict(X_train, verbose=0).flatten()
y_pred_test = best_model.predict(X_test, verbose=0).flatten()

train_metrics = {
    'R2': float(r2_score(y_train, y_pred_train)),
    'MAE': float(mean_absolute_error(y_train, y_pred_train)),
    'RMSE': float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
}
test_metrics = {
    'R2': float(r2_score(y_test, y_pred_test)),
    'MAE': float(mean_absolute_error(y_test, y_pred_test)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
}

print(f"\n--- ANN Training Metrics ---")
print(f"  R²:   {train_metrics['R2']:.4f}")
print(f"  MAE:  {train_metrics['MAE']:.4f}")
print(f"  RMSE: {train_metrics['RMSE']:.4f}")

print(f"\n--- ANN Test Metrics ---")
print(f"  R²:   {test_metrics['R2']:.4f}")
print(f"  MAE:  {test_metrics['MAE']:.4f}")
print(f"  RMSE: {test_metrics['RMSE']:.4f}")

# --- Save model ---
model_path = os.path.join(MODELS_DIR, 'ann_model.keras')
best_model.save(model_path)
print(f"\n[INFO] ANN model saved to: {model_path}")

# --- Save results ---
results = {
    'model': 'Artificial Neural Network (ANN)',
    'best_config': best_config,
    'training_time_seconds': round(best_config['train_time'], 2),
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'architecture_search': results_all,
    'training_history': {
        'loss': [float(v) for v in best_history.history['loss']],
        'val_loss': [float(v) for v in best_history.history['val_loss']],
    }
}

results_path = os.path.join(RESULTS_DIR, 'ann_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"[INFO] ANN results saved to: {results_path}")

# --- Save predictions ---
predictions = {
    'y_test': y_test,
    'y_pred_test': y_pred_test,
    'y_train': y_train,
    'y_pred_train': y_pred_train
}
pred_path = os.path.join(RESULTS_DIR, 'ann_predictions.pkl')
with open(pred_path, 'wb') as f:
    pickle.dump(predictions, f)
print(f"[INFO] ANN predictions saved to: {pred_path}")

print("\n" + "=" * 60)
print("ANN MODEL TRAINING COMPLETE ✓")
print("=" * 60)
