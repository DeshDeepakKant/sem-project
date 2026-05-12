"""
Premium Streamlit App — ML Adsorption Capacity Predictor
Per-biochar prediction using GPR, SVR, and ANN models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR= os.path.join(BASE_DIR, 'results')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')

# ── Biochar config ─────────────────────────────────────────────────────────────
BIOCHAR_OPTIONS = ['PAC', 'PB600', 'NaOH-activated SCW']
SAFE_NAMES = {
    'PAC': 'PAC',
    'PB600': 'PB600',
    'NaOH-activated SCW': 'NaOH_activated_SCW',
}
# Actual adsorbent names stored in the combined CSV
DATASET_NAMES = {
    'PAC': 'PAC',
    'PB600': 'PB600',
    'NaOH-activated SCW': 'NaOH-activated SCW biochars',
}
BIOCHAR_COLORS = {
    'PAC': '#e74c3c',
    'PB600': '#2ecc71',
    'NaOH-activated SCW': '#3498db',
}
MODEL_COLORS = {'GPR': '#2196F3', 'SVR': '#FF9800', 'ANN': '#4CAF50'}
FEATURES = ['pH', 'Temperature (°C)', 'Contact Time (min)',
            'Initial Concentration (mg/L)', 'Adsorbent Dosage (g/L)']
FEATURE_KEYS = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']
FEATURE_RANGES = {
    'pH':                   (2.0,  14.0,  7.0,  0.5),
    'Temperature (°C)':     (5.0,  60.0,  25.0, 1.0),
    'Contact Time (min)':   (1.0,  1440.0, 720.0, 10.0),
    'Initial Concentration (mg/L)': (0.5, 500.0, 50.0, 0.5),
    'Adsorbent Dosage (g/L)':       (0.001, 1.0,  0.05, 0.001),
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adsorption Capacity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1f3d 50%, #0a2a5e 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    text-align: center;
}
.hero h1 { color: #60a5fa; font-size: 2.2rem; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
.hero p  { color: #94a3b8; font-size: 1.05rem; margin: 0; }

/* ── Section header ── */
.section-title {
    font-size: 1.35rem; font-weight: 700;
    color: #e2e8f0; margin: 1.5rem 0 0.8rem;
    border-left: 4px solid #3b82f6;
    padding-left: 0.7rem;
}

/* ── Metric card ── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1rem;
    text-align: center;
    color: white;
    height: 100%;
}
.metric-card .model-label { font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.3rem; }
.metric-card .metric-val  { font-size: 2.4rem; font-weight: 800; line-height: 1; margin: 0.2rem 0; }
.metric-card .metric-sub  { font-size: 0.85rem; color: #64748b; }
.metric-card hr { border-color: #334155; margin: 0.7rem 0; }
.metric-card .detail-row  { display: flex; justify-content: space-between; font-size: 0.82rem; }
.metric-card .detail-row span.lbl { color: #94a3b8; }
.metric-card .detail-row span.val { color: #e2e8f0; font-weight: 600; }

/* ── Prediction result box ── */
.pred-box {
    border-radius: 14px;
    padding: 1.6rem 1rem;
    text-align: center;
    color: white;
    border: 2px solid;
    margin-top: 0.5rem;
}
.pred-box .pred-label { font-size: 1rem; margin-bottom: 0.4rem; }
.pred-box .pred-value { font-size: 2.8rem; font-weight: 900; line-height: 1; }
.pred-box .pred-unit  { font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }
.pred-box .pred-unc   { font-size: 0.9rem; color: #cbd5e1; margin-top: 0.2rem; }

/* ── Biochar badge ── */
.bc-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

/* ── GA result card ── */
.ga-hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 2px solid #38bdf8;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
}
.ga-hero h2 { color: #38bdf8; font-size: 3rem; margin: 0.5rem 0; }
.ga-hero p  { color: #94a3b8; margin: 0; }

/* ── Table styling ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    color: #e2e8f0;
    font-size: 0.92rem;
}
.styled-table th {
    background: #1e293b;
    padding: 0.6rem 1rem;
    text-align: left;
    color: #94a3b8;
    font-weight: 600;
    border-bottom: 1px solid #334155;
}
.styled-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #1e293b;
}
.styled-table tr:hover td { background: #1e293b; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117; }
section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; font-size: 0.92rem; }
section[data-testid="stSidebar"] h2 { color: #60a5fa; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b; color: #94a3b8;
    border-radius: 8px; padding: 8px 18px;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important; color: white !important;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models_for(safe: str):
    models = {}
    gpr_path = os.path.join(MODELS_DIR, safe, 'gpr_model.pkl')
    svr_path = os.path.join(MODELS_DIR, safe, 'svr_model.pkl')
    ann_path = os.path.join(MODELS_DIR, safe, 'ann_model.keras')

    if os.path.exists(gpr_path):
        with open(gpr_path, 'rb') as f:
            models['GPR'] = pickle.load(f)
    if os.path.exists(svr_path):
        with open(svr_path, 'rb') as f:
            models['SVR'] = pickle.load(f)
    if os.path.exists(ann_path):
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        models['ANN'] = tf.keras.models.load_model(ann_path)
    return models


@st.cache_resource
def load_scaler_for(safe: str):
    path = os.path.join(DATA_DIR, f'scaler_{safe}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_results_for(safe: str):
    results = {}
    for name in ['GPR', 'SVR', 'ANN']:
        path = os.path.join(RESULTS_DIR, safe, f'{name.lower()}_results.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[name] = json.load(f)
    return results


@st.cache_data
def load_ga_results_for(safe: str):
    path = os.path.join(RESULTS_DIR, safe, 'ga_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_combined_data():
    path = os.path.join(DATA_DIR, 'cleaned_combined.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def predict_single(models, scaler, inputs, model_name):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    if model_name == 'ANN':
        return float(models['ANN'].predict(X_scaled, verbose=0).flatten()[0]), None
    elif model_name == 'GPR':
        pred, std = models['GPR'].predict(X_scaled, return_std=True)
        return float(pred[0]), float(std[0])
    else:
        return float(models['SVR'].predict(X_scaled)[0]), None


def plot_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(PLOTS_DIR, filename))


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧪 Adsorption ML")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    biochar = st.selectbox(
        "**Select Biochar Material**",
        BIOCHAR_OPTIONS,
        help="Choose which biochar adsorbent to analyse",
    )
    safe = SAFE_NAMES[biochar]
    bc_color = BIOCHAR_COLORS[biochar]

    st.markdown(f"""
    <div class="bc-badge" style="background:{bc_color}22; color:{bc_color}; border:1px solid {bc_color}55;">
        ● {biochar}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "**Navigation**",
        [
            "🔮 Live Prediction",
            "📊 Model Performance",
            "📈 Visualizations",
            "🧬 GA Optimization",
            "📁 Batch Prediction",
            "📋 Dataset Explorer",
        ],
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Built with Streamlit · scikit-learn · TensorFlow<br>"
        "Based on Jaffari et al. (WEIL Group, UNIST)</small>",
        unsafe_allow_html=True,
    )

# ── Load per-biochar assets ────────────────────────────────────────────────────
models  = load_models_for(safe)
scaler  = load_scaler_for(safe)
results = load_results_for(safe)

# ══════════════════════════════════════════════════════════════════════════════
# Hero
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🧪 Adsorption Capacity Predictor</h1>
    <p>ML-powered prediction of emerging contaminant removal by biochar — GPR · SVR · ANN</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Live Prediction":
    st.markdown(f'<div class="section-title">🔮 Live Prediction — {biochar}</div>', unsafe_allow_html=True)
    st.caption("Adjust experimental conditions and get instant predictions from all three ML models.")

    if scaler is None or not models:
        st.error("Models or scaler not found for this biochar. Please run the pipeline first.")
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Experimental Parameters**")
        ph      = st.slider("🧫 Solution pH",                  2.0,  14.0,  7.0,  0.5)
        temp    = st.slider("🌡️ Temperature (°C)",             5.0,  60.0,  25.0, 1.0)
        contact = st.slider("⏱️ Contact Time (min)",           1.0,  1440.0, 720.0, 10.0)
        conc    = st.slider("💧 Initial Concentration (mg/L)", 0.5,  500.0, 50.0, 0.5)
        dosage  = st.slider("⚗️ Adsorbent Dosage (g/L)",       0.001, 1.0,  0.05, 0.001,
                            format="%.3f")
        inputs  = [ph, temp, contact, conc, dosage]

    with col_right:
        st.markdown("**Prediction Results**")
        model_cfg = {
            'GPR': ('#2196F3', '🔵'),
            'SVR': ('#FF9800', '🟠'),
            'ANN': ('#4CAF50', '🟢'),
        }
        for name, (color, icon) in model_cfg.items():
            if name not in models:
                st.warning(f"{name} model not loaded.")
                continue
            pred, unc = predict_single(models, scaler, inputs, name)
            unc_html = f'<div class="pred-unc">± {unc:.2f} uncertainty</div>' if unc is not None else ""
            st.markdown(f"""
            <div class="pred-box" style="background:linear-gradient(135deg,{color}18,{color}08);
                 border-color:{color}88;">
                <div class="pred-label">{icon} {name}</div>
                <div class="pred-value" style="color:{color};">{pred:.2f}</div>
                <div class="pred-unit">mg/g — Adsorption Capacity</div>
                {unc_html}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Input Summary**")
    summary_df = pd.DataFrame(
        [[ph, temp, contact, conc, dosage]],
        columns=['pH', 'Temp (°C)', 'Contact Time (min)', 'Conc (mg/L)', 'Dosage (g/L)'],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # All-model comparison bar
    if len(models) > 1:
        st.markdown('<div class="section-title">Model Comparison at These Conditions</div>',
                    unsafe_allow_html=True)
        comparison = {}
        for name in models:
            val, _ = predict_single(models, scaler, inputs, name)
            comparison[name] = val
        comp_df = pd.DataFrame(list(comparison.items()), columns=['Model', 'Predicted Capacity (mg/g)'])
        st.bar_chart(comp_df.set_index('Model'))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown(f'<div class="section-title">📊 Model Performance — {biochar}</div>',
                unsafe_allow_html=True)

    if not results:
        st.error("No results found. Please run the training pipeline first.")
        st.stop()

    # Metric cards
    cols = st.columns(len(results), gap="medium")
    for i, (name, res) in enumerate(results.items()):
        m  = res['test_metrics']
        mt = res['train_metrics']
        c  = MODEL_COLORS[name]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="model-label" style="color:{c};font-weight:700;font-size:1.1rem;">{name}</div>
                <div class="metric-val" style="color:{c};">{m['R2']:.4f}</div>
                <div class="metric-sub">Test R² Score</div>
                <hr>
                <div class="detail-row">
                    <span class="lbl">Test MAE</span>
                    <span class="val">{m['MAE']:.2f}</span>
                </div>
                <div class="detail-row">
                    <span class="lbl">Test RMSE</span>
                    <span class="val">{m['RMSE']:.2f}</span>
                </div>
                <div class="detail-row">
                    <span class="lbl">Train R²</span>
                    <span class="val">{mt['R2']:.4f}</span>
                </div>
                <div class="detail-row">
                    <span class="lbl">Train MAE</span>
                    <span class="val">{mt['MAE']:.2f}</span>
                </div>
                <div class="detail-row">
                    <span class="lbl">Train Time</span>
                    <span class="val">{res.get('training_time_seconds','—')}s</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Full table
    st.markdown('<div class="section-title">Detailed Metrics Table</div>', unsafe_allow_html=True)
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'Test R²': res['test_metrics']['R2'],
            'Test MAE': res['test_metrics']['MAE'],
            'Test RMSE': res['test_metrics']['RMSE'],
            'Train R²': res['train_metrics']['R2'],
            'Train MAE': res['train_metrics']['MAE'],
            'Training Time (s)': res.get('training_time_seconds', '—'),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Comparison charts
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
    metric_choice = st.selectbox("Select metric", ['R2', 'MAE', 'RMSE'], key='metric_sel')
    chart_df = pd.DataFrame({
        'Model': list(results.keys()),
        metric_choice: [results[m]['test_metrics'][metric_choice] for m in results],
    })
    st.bar_chart(chart_df.set_index('Model'))

    # Across all biochars comparison
    st.markdown('<div class="section-title">R² Across All Biochars</div>', unsafe_allow_html=True)
    cross_rows = []
    for bc_display, bc_safe in SAFE_NAMES.items():
        bc_res = load_results_for(bc_safe)
        for model_name, res in bc_res.items():
            cross_rows.append({
                'Biochar': bc_display,
                'Model': model_name,
                'Test R²': res['test_metrics']['R2'],
                'Test MAE': res['test_metrics']['MAE'],
            })
    if cross_rows:
        cross_df = pd.DataFrame(cross_rows)
        pivot = cross_df.pivot(index='Biochar', columns='Model', values='Test R²')
        st.dataframe(pivot.style.background_gradient(cmap='Blues', axis=None),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown(f'<div class="section-title">📈 Visualizations — {biochar}</div>',
                unsafe_allow_html=True)

    tabs = st.tabs([
        "Predicted vs Actual",
        "Error Distribution",
        "ANN Training Curve",
        "Violin: Obs vs Pred",
        "Cross-Biochar Plots",
    ])

    per = 'per_biochar'
    comp = 'comparison'

    def show_plot(rel_path, caption=""):
        full = os.path.join(PLOTS_DIR, rel_path)
        if os.path.exists(full):
            st.image(full, caption=caption, use_container_width=True)
        else:
            st.info(f"Plot not yet generated: `{rel_path}`")

    with tabs[0]:
        show_plot(f'{per}/{safe}_predicted_vs_actual.png',
                  f'Predicted vs Actual — {biochar}')

    with tabs[1]:
        show_plot(f'{per}/{safe}_error_distribution.png',
                  f'Prediction Error Distribution — {biochar}')

    with tabs[2]:
        show_plot(f'{per}/{safe}_ann_training_curve.png',
                  f'ANN Training Curve — {biochar}')

    with tabs[3]:
        show_plot(f'{per}/{safe}_violin_obs_vs_pred.png',
                  f'Observed vs Predicted Distribution — {biochar}')

    with tabs[4]:
        sub = st.selectbox("Select comparison plot", [
            "Model Comparison (All Biochars)",
            "R² Line Comparison",
            "Violin Error Comparison",
            "Feature Distributions",
            "Combined Dashboard",
        ])
        plot_map = {
            "Model Comparison (All Biochars)": f'{comp}/model_comparison_across_biochars.png',
            "R² Line Comparison":              f'{comp}/r2_comparison_line.png',
            "Violin Error Comparison":         f'{comp}/violin_error_comparison.png',
            "Feature Distributions":           f'{comp}/feature_distributions_by_biochar.png',
            "Combined Dashboard":              f'{comp}/combined_dashboard.png',
        }
        show_plot(plot_map[sub], sub)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GA Optimization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 GA Optimization":
    st.markdown(f'<div class="section-title">🧬 Genetic Algorithm Optimization — {biochar}</div>',
                unsafe_allow_html=True)

    ga = load_ga_results_for(safe)

    if ga is None:
        st.warning(
            f"GA results not yet generated for **{biochar}**. "
            "Please run `python3 code/04_genetic_algorithm.py` from the project folder."
        )
    else:
        fitness_model = ga.get('fitness_model', 'N/A')
        max_cap       = ga.get('predicted_max_capacity', 0.0)
        opt_cond      = ga.get('optimized_conditions', {})
        ga_params     = ga.get('ga_params', {})

        st.markdown(f"""
        <div class="ga-hero">
            <p style="color:#94a3b8; font-size:1rem; margin-bottom:0.2rem;">
                🏆 Maximum Predicted Adsorption Capacity
            </p>
            <h2>{max_cap:.2f} <span style="font-size:1.2rem;color:#94a3b8;">mg/g</span></h2>
            <p>Optimised using <b style="color:#38bdf8;">{fitness_model}</b> model
               over <b style="color:#38bdf8;">{ga_params.get('generations','—')}</b> generations
               (population {ga_params.get('population','—')})</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown('<div class="section-title">Optimised Conditions</div>',
                        unsafe_allow_html=True)
            label_map = {
                'pH': 'Solution pH',
                'Temperature': 'Temperature (°C)',
                'Contact time': 'Contact Time (min)',
                'Initial concentration': 'Initial Concentration (mg/L)',
                'Adsorbent dosage': 'Adsorbent Dosage (g/L)',
            }
            rows_html = ""
            for k, v in opt_cond.items():
                label = label_map.get(k, k)
                rows_html += f"""
                <tr>
                    <td><span class="lbl">{label}</span></td>
                    <td style="text-align:right;font-weight:700;color:#38bdf8;">{v:.4f}</td>
                </tr>"""
            st.markdown(f"""
            <table class="styled-table">
                <thead><tr><th>Parameter</th><th style="text-align:right;">Optimal Value</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">GA Parameters</div>', unsafe_allow_html=True)
            for k, v in ga_params.items():
                st.metric(k.replace('_', ' ').title(), v)

        # Convergence plot
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        conv_path = os.path.join(PLOTS_DIR, 'per_biochar', f'{safe}_ga_convergence.png')
        if os.path.exists(conv_path):
            st.markdown('<div class="section-title">Convergence Curve</div>', unsafe_allow_html=True)
            st.image(conv_path, use_container_width=True)
        else:
            st.info("GA convergence plot not yet generated for this biochar.")

    # Cross-biochar GA comparison
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GA Results — All Biochars</div>', unsafe_allow_html=True)
    ga_rows = []
    for bc_display, bc_safe in SAFE_NAMES.items():
        bc_ga = load_ga_results_for(bc_safe)
        if bc_ga:
            ga_rows.append({
                'Biochar': bc_display,
                'Best Model': bc_ga.get('fitness_model', '—'),
                'Max Capacity (mg/g)': bc_ga.get('predicted_max_capacity', '—'),
                'Optimal pH': bc_ga.get('optimized_conditions', {}).get('pH', '—'),
                'Optimal Temp (°C)': bc_ga.get('optimized_conditions', {}).get('Temperature', '—'),
                'Optimal Time (min)': bc_ga.get('optimized_conditions', {}).get('Contact time', '—'),
            })
    if ga_rows:
        st.dataframe(pd.DataFrame(ga_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No GA results available yet. Run step 04 of the pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.markdown(f'<div class="section-title">📁 Batch Prediction — {biochar}</div>',
                unsafe_allow_html=True)
    st.caption("Upload a CSV with columns: `pH`, `Temperature`, `Contact time`, "
               "`Initial concentration`, `Adsorbent dosage`")

    if scaler is None or not models:
        st.error("Models or scaler not found for this biochar.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
    with c2:
        model_choice = st.selectbox("Model for prediction", list(models.keys()))

    required_cols = ['pH', 'Temperature', 'Contact time', 'Initial concentration', 'Adsorbent dosage']

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(input_df.head(10), use_container_width=True)
        st.caption(f"{len(input_df)} rows × {len(input_df.columns)} columns")

        missing = [c for c in required_cols if c not in input_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                X_new    = input_df[required_cols].values
                X_scaled = scaler.transform(X_new)

                if model_choice == 'ANN':
                    preds = models['ANN'].predict(X_scaled, verbose=0).flatten()
                elif model_choice == 'GPR':
                    preds, _ = models['GPR'].predict(X_scaled, return_std=True)
                else:
                    preds = models['SVR'].predict(X_scaled)

                result_df = input_df.copy()
                result_df['Predicted Capacity (mg/g)'] = np.round(preds, 4)

                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Capacity", f"{preds.mean():.2f} mg/g")
                m2.metric("Max Capacity",  f"{preds.max():.2f} mg/g")
                m3.metric("Min Capacity",  f"{preds.min():.2f} mg/g")

                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv_out,
                    f"batch_predictions_{safe}_{model_choice}.csv",
                    "text/csv",
                    use_container_width=True,
                )
    else:
        # Example template
        st.markdown("**Example CSV format:**")
        example = pd.DataFrame({
            'pH': [7.0, 6.5, 8.0],
            'Temperature': [25, 30, 20],
            'Contact time': [720, 480, 1440],
            'Initial concentration': [50, 100, 25],
            'Adsorbent dosage': [0.05, 0.08, 0.03],
        })
        st.dataframe(example, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Template CSV",
            example.to_csv(index=False),
            "template.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.markdown(f'<div class="section-title">📋 Dataset Explorer — {biochar}</div>',
                unsafe_allow_html=True)

    df_combined = load_combined_data()
    if df_combined is None:
        st.error("Combined dataset not found. Run the preprocessing pipeline first.")
        st.stop()

    df_bc = df_combined[df_combined['Adsorbent'] == DATASET_NAMES[biochar]].copy()

    st.markdown(f"**{len(df_bc)} records** for {biochar} | "
                f"**{len(df_combined)} total** across all biochars")

    tabs = st.tabs(["This Biochar", "All Biochars", "Statistics"])

    with tabs[0]:
        st.dataframe(df_bc, use_container_width=True)
        feat = st.selectbox("Distribution of:", df_bc.columns.tolist(),
                            index=df_bc.columns.tolist().index('Adsorption capacity')
                            if 'Adsorption capacity' in df_bc.columns else 0)
        st.bar_chart(df_bc[feat].value_counts().sort_index().head(60))

    with tabs[1]:
        st.dataframe(df_combined, use_container_width=True)
        all_feat = st.selectbox("Distribution of (all):", df_combined.columns.tolist(),
                                index=df_combined.columns.tolist().index('Adsorption capacity')
                                if 'Adsorption capacity' in df_combined.columns else 0,
                                key='all_feat')
        for bc_d in BIOCHAR_OPTIONS:
            subset = df_combined[df_combined['Adsorbent'] == DATASET_NAMES[bc_d]]
            st.write(f"**{bc_d}** — {len(subset)} rows")

    with tabs[2]:
        st.markdown("**Statistics for selected biochar**")
        st.dataframe(df_bc.describe().T.style.background_gradient(cmap='Blues', axis=1),
                     use_container_width=True)
        st.markdown("**Statistics for all biochars**")
        st.dataframe(df_combined.describe().T, use_container_width=True)
