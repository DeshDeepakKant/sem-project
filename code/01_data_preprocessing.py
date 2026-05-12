"""
Step 01: Data Preprocessing — Filter to 3 biochars, prepare per-biochar datasets.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, os, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# === Configuration ===
BIOCHARS = ['PAC', 'PB600', 'NaOH-activated SCW biochars']
SAFE_NAMES = {'PAC': 'PAC', 'PB600': 'PB600', 'NaOH-activated SCW biochars': 'NaOH_activated_SCW'}
RAW_FEATURES = ['Solution pH', 'Adsorption temperature', 'Adsorption time',
                'Initial concentration', 'Adsorbent dosage']
CLEAN_FEATURES = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']
TARGET_RAW = 'Capacity'
TARGET_CLEAN = 'Adsorption capacity'

print("=" * 60)
print("STEP 1: DATA PREPROCESSING (3-Biochar Pipeline)")
print("=" * 60)

# Load raw data
raw_path = os.path.join(BASE_DIR, '..', '..', 'Raw_data.csv')
if not os.path.exists(raw_path):
    raw_path = '/home/anya/vjta-project/Adsorption-capacity-prediction-for-ECs/Raw_data.csv'
df_raw = pd.read_csv(raw_path)
print(f"[INFO] Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"[INFO] All adsorbents: {df_raw['Adsorbent'].nunique()} types")

# Filter to selected biochars
df_filtered = df_raw[df_raw['Adsorbent'].isin(BIOCHARS)].copy()
print(f"[INFO] Filtered to 3 biochars: {df_filtered.shape[0]} total rows")

all_stats = {}

for biochar in BIOCHARS:
    safe = SAFE_NAMES[biochar]
    print(f"\n--- {biochar} ({safe}) ---")
    df_bio = df_filtered[df_filtered['Adsorbent'] == biochar].copy()

    # Select and rename columns
    df_clean = df_bio[RAW_FEATURES + [TARGET_RAW]].copy()
    df_clean.columns = CLEAN_FEATURES + [TARGET_CLEAN]

    # Drop missing
    before = len(df_clean)
    df_clean.dropna(inplace=True)
    after = len(df_clean)
    if before != after:
        print(f"  [WARN] Dropped {before - after} rows with NaN")

    # Save cleaned CSV
    df_clean.to_csv(os.path.join(DATA_DIR, f'cleaned_{safe}.csv'), index=False)

    # Train/test split
    X = df_clean[CLEAN_FEATURES].values
    y = df_clean[TARGET_CLEAN].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    data = {
        'X_train': X_train, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': CLEAN_FEATURES, 'biochar_name': biochar
    }
    with open(os.path.join(DATA_DIR, f'processed_{safe}.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(DATA_DIR, f'scaler_{safe}.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"  Total: {after} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Capacity range: [{y.min():.2f}, {y.max():.2f}], Mean: {y.mean():.2f}")

    all_stats[biochar] = {
        'safe_name': safe, 'total': after,
        'train': int(len(X_train)), 'test': int(len(X_test)),
        'target_mean': round(float(y.mean()), 4),
        'target_std': round(float(y.std()), 4),
        'target_min': round(float(y.min()), 4),
        'target_max': round(float(y.max()), 4)
    }

# Save combined dataset with adsorbent label
df_combined = df_filtered[RAW_FEATURES + [TARGET_RAW, 'Adsorbent']].copy()
df_combined.columns = CLEAN_FEATURES + [TARGET_CLEAN, 'Adsorbent']
df_combined.dropna(inplace=True)
df_combined.to_csv(os.path.join(DATA_DIR, 'cleaned_combined.csv'), index=False)

with open(os.path.join(DATA_DIR, 'dataset_stats.json'), 'w') as f:
    json.dump(all_stats, f, indent=2)

print(f"\n[INFO] Combined dataset: {len(df_combined)} rows")
print("[INFO] Data preprocessing complete ✓")
