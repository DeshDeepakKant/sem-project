# Dataset

## Source

**File:** `Raw_data.csv` (root directory) — 3,757 rows, converted from the original `Raw_data.xlsx`.

This dataset was compiled by the WEIL Group at UNIST from literature-reported adsorption
experiments across a wide variety of biochar adsorbents and emerging contaminants.

## Raw Columns Used (Phase 2 Pipeline)

| Raw Column Name | Renamed To | Type |
|----------------|-----------|------|
| Solution pH | pH | Numeric |
| Adsorption temperature | Temperature | Numeric |
| Adsorption time | Contact time | Numeric |
| Initial concentration | Initial concentration | Numeric |
| Adsorbent dosage | Adsorbent dosage | Numeric |
| Capacity | Adsorption capacity | Numeric (target) |
| Adsorbent | Adsorbent | Categorical (filter key) |

The full dataset also contains adsorbent composition features (BET surface area, pore volume,
N/C ratio, C% content, etc.) and synthesis condition features which were used in Phase 1 only.

## Three Biochars Selected for Phase 2

Only rows where `Adsorbent` matches one of the three values below are used:

| Adsorbent | Safe name | Total rows | Train (80%) | Test (20%) |
|-----------|-----------|-----------|-------------|-----------|
| PAC | PAC | 162 | 129 | 33 |
| PB600 | PB600 | 162 | 129 | 33 |
| NaOH-activated SCW biochars | NaOH_activated_SCW | 486 | 388 | 98 |

## Adsorption Capacity Statistics (mg/g)

| Biochar | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| PAC | 0.00 | 385.33 | 98.42 | 71.32 |
| PB600 | −4.46 | 15.86 | 5.02 | 2.71 |
| NaOH-activated SCW | 0.00 | 267.76 | 105.45 | 54.75 |

Note: PB600 shows a small number of slightly negative capacity values — likely experimental
measurement artefacts at very low contaminant loads.

## Preprocessing Steps

Implemented in `code/01_data_preprocessing.py`:

1. **Filter** — Keep only rows for the three target biochars.
2. **Column rename** — Map raw column names to clean names.
3. **Drop NaN** — Remove any row with missing values in the selected columns.
4. **Train/test split** — 80/20 stratified random split (`random_state=42`).
5. **Standardisation** — `StandardScaler` fitted on training data, applied to both splits.
   Scaler is saved per biochar as `scaler_<safe_name>.pkl`.
6. **Save** — Processed arrays saved to `processed_<safe_name>.pkl`.
   A combined CSV `cleaned_combined.csv` is also saved for cross-biochar plots.

## Output Files

| File | Location | Contents |
|------|----------|---------|
| `cleaned_<safe>.csv` | `data/` | Cleaned per-biochar dataframe |
| `cleaned_combined.csv` | `data/` | All three biochars in one file with `Adsorbent` label |
| `processed_<safe>.pkl` | `data/` | Dict with X_train, X_test, y_train, y_test (raw + scaled) |
| `scaler_<safe>.pkl` | `data/` | Fitted StandardScaler for the biochar |
| `dataset_stats.json` | `data/` | Summary statistics for all three biochars |
