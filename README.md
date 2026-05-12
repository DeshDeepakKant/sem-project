# Adsorption Capacity Prediction for Emerging Contaminants (ECs)

## Project Documentation Index

| File | Contents |
|------|----------|
| [docs/01_project_overview.md](docs/01_project_overview.md) | Research background, goals, and two-phase structure |
| [docs/02_dataset.md](docs/02_dataset.md) | Raw data description, features, biochars, and statistics |
| [docs/03_phase1_original_models.md](docs/03_phase1_original_models.md) | Original 10-model paper reproduction (CatBoost winner, SHAP) |
| [docs/04_phase2_pipeline.md](docs/04_phase2_pipeline.md) | New GPR/SVR/ANN per-biochar pipeline — code walkthrough |
| [docs/05_model_results.md](docs/05_model_results.md) | All performance metrics and model comparisons |
| [docs/06_genetic_algorithm.md](docs/06_genetic_algorithm.md) | GA optimization — method, parameters, and optimal conditions |
| [docs/07_streamlit_app.md](docs/07_streamlit_app.md) | Streamlit web app — features and how to run |
| [docs/08_project_structure.md](docs/08_project_structure.md) | Full directory tree and file descriptions |
| [report/Project_Report.md](report/Project_Report.md) | Full project report with figures |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full Phase 2 pipeline (preprocessing → training → plots → GA → report)
python run_pipeline.py

# Launch the Streamlit app
streamlit run app.py
```

## What We Built

**Phase 1** — Reproduced a 10-model comparison from the UNIST/WEIL group paper on EC adsorption.
CatBoost was the top performer (R² = 0.9433, MAE = 4.95) across the full 3,757-point dataset.

**Phase 2** — Built a new per-biochar ML pipeline for three specific materials (PAC, PB600, NaOH-activated SCW)
using GPR, SVR, and ANN, then ran Genetic Algorithm optimization to find conditions that maximize
adsorption capacity for each biochar.

**Streamlit App** — Interactive web interface for live predictions across all three biochars and models,
with model comparison dashboards and optimized condition displays.
