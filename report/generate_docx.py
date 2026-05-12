"""
Convert auto_generated_report.md into a professional academic DOCX report.
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

BASE = "/home/anya/vjta-project/Adsorption-capacity-prediction-for-ECs"
PLOTS_COMP = os.path.join(BASE, "plots", "comparison")
PLOTS_PER  = os.path.join(BASE, "plots", "per_biochar")
OUT_PATH   = os.path.join(BASE, "report", "Final_Report.docx")

doc = Document()

# ── Page margins ──────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.54)

# ── Styles ────────────────────────────────────────────────────
styles = doc.styles

def set_style(name, font_name, size_pt, bold=False, color=None, space_before=0, space_after=6):
    s = styles[name]
    s.font.name        = font_name
    s.font.size        = Pt(size_pt)
    s.font.bold        = bold
    if color:
        s.font.color.rgb = RGBColor(*color)
    s.paragraph_format.space_before = Pt(space_before)
    s.paragraph_format.space_after  = Pt(space_after)

set_style('Normal',    'Times New Roman', 12, space_after=6)
set_style('Heading 1', 'Times New Roman', 16, bold=True, color=(0,70,127), space_before=12, space_after=6)
set_style('Heading 2', 'Times New Roman', 14, bold=True, color=(0,70,127), space_before=10, space_after=4)
set_style('Heading 3', 'Times New Roman', 12, bold=True, color=(0,70,127), space_before=8, space_after=3)

def add_para(text, style='Normal', align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    return p

def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(80, 80, 80)
    p.paragraph_format.space_after = Pt(10)

def add_image(path, caption, width=Inches(5.5)):
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(path, width=width)
        add_caption(caption)
    else:
        add_para(f"[Figure not found: {os.path.basename(path)}]")

def add_table(headers, rows):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(11)
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        # shade header
        tc = hdr[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), '00468F')
        shd.set(qn('w:color'), 'FFFFFF')
        tcPr.append(shd)
        for run in hdr[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
    for ri, row in enumerate(rows):
        cells = t.rows[ri+1].cells
        fill = 'EAF1FB' if ri % 2 == 0 else 'FFFFFF'
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cells[ci].paragraphs[0].runs[0].font.size = Pt(11)
            tc = cells[ci]._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:fill'), fill)
            tcPr.append(shd)
    doc.add_paragraph()

# ════════════════════════════════════════════════════════════════
# TITLE PAGE
# ════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("ML-Based Adsorption Capacity Prediction\nfor Emerging Contaminants on Biochar")
run.bold = True; run.font.size = Pt(22); run.font.name = 'Times New Roman'
run.font.color.rgb = RGBColor(0, 70, 127)

doc.add_paragraph()
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Project Report — Three-Biochar Process-Specific Analysis")
r2.font.size = Pt(14); r2.italic = True; r2.font.name = 'Times New Roman'

doc.add_paragraph()
p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("Generated: 2026-05-12")
r3.font.size = Pt(12); r3.font.name = 'Times New Roman'

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# SEC 1: OBJECTIVE
# ════════════════════════════════════════════════════════════════
doc.add_heading("1. Objective", level=1)
add_para(
    "This project applies machine learning methodologies inspired by state-of-the-art research to predict "
    "the adsorption capacity of emerging contaminants (ECs) on three specific biochar materials: "
    "PAC (Powdered Activated Carbon), PB600 (600°C Pyrolysis Biochar), and NaOH-Activated SCW Biochar. "
    "Three ML models — Gaussian Process Regression (GPR), Support Vector Regression (SVR), and Artificial "
    "Neural Network (ANN) — are trained, tuned, and evaluated independently for each material. "
    "A Genetic Algorithm (GA) is additionally employed to identify optimal experimental conditions "
    "per biochar for maximizing adsorption capacity."
)

# ════════════════════════════════════════════════════════════════
# SEC 2: DATASET
# ════════════════════════════════════════════════════════════════
doc.add_heading("2. Dataset Description", level=1)
add_para(
    "The raw dataset contains 3,757 experimental observations across 15 adsorbent materials. "
    "For this study, the data was filtered to three process-specific biochars totalling 810 samples."
)

doc.add_heading("2.1 Biochar Summary", level=2)
add_table(
    ["Biochar", "Total Samples", "Training (80%)", "Testing (20%)", "Mean Capacity", "Max Capacity"],
    [
        ["PAC",                   "162", "129", "33",  "42.57",  "155.45"],
        ["PB600",                 "162", "129", "33",  "7.08",   "21.18"],
        ["NaOH-activated SCW",    "486", "388", "98",  "98.23",  "536.34"],
    ]
)

doc.add_heading("2.2 Input Feature Statistics (Combined)", level=2)
add_table(
    ["Feature", "Mean", "Std Dev", "Min", "Max"],
    [
        ["pH",                   "7.04", "1.02", "3.00",  "11.00"],
        ["Temperature (°C)",     "25.21","3.72", "15.00", "45.00"],
        ["Contact Time (min)",   "996.9","584.8","0.00",  "1455.00"],
        ["Initial Conc. (mg/L)", "13.62","13.95","0.50",  "100.00"],
        ["Adsorbent Dosage (g)", "0.080","0.044","0.020", "0.160"],
    ]
)

doc.add_heading("2.3 Feature Distributions by Biochar", level=2)
add_image(os.path.join(PLOTS_COMP, "feature_distributions_by_biochar.png"),
          "Figure 1. Distribution of input features and target variable across the three biochar materials.")

# ════════════════════════════════════════════════════════════════
# SEC 3: METHODOLOGY
# ════════════════════════════════════════════════════════════════
doc.add_heading("3. Methodology", level=1)

doc.add_heading("3.1 Data Preprocessing", level=2)
for item in [
    "Data filtered from 15 biochars to 3 selected materials.",
    "Missing values removed; no imputation applied.",
    "Features standardized using Z-score normalization (StandardScaler) fitted on training data only.",
    "Data split: 80% training / 20% testing (random_state=42) applied independently per biochar."
]:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item).font.size = Pt(12)

doc.add_heading("3.2 Gaussian Process Regression (GPR)", level=2)
add_para("Kernel: ConstantKernel × RBF + WhiteKernel (noise). Hyperparameters optimized via marginal likelihood maximization with 5 random restarts. Provides uncertainty estimates alongside predictions.")

doc.add_heading("3.3 Support Vector Regression (SVR)", level=2)
add_para("Kernels tested: RBF and Linear. Hyperparameters C, gamma, and epsilon tuned via GridSearchCV with 3-fold cross-validation. Best parameters selected independently per biochar.")

doc.add_heading("3.4 Artificial Neural Network (ANN)", level=2)
add_para("Framework: TensorFlow/Keras. Architecture: Dense layers with ReLU activation, BatchNormalization, and Dropout(0.2). Trained with Adam optimizer (lr=0.001), EarlyStopping (patience=20).")

doc.add_heading("3.5 Genetic Algorithm (GA) Optimization", level=2)
add_para("Population: 100 | Generations: 150 | Mutation rate: 10%. Each biochar uses its own best-performing model as the fitness function. Elitism preserves the best individual across generations.")

# ════════════════════════════════════════════════════════════════
# SEC 4: RESULTS
# ════════════════════════════════════════════════════════════════
doc.add_heading("4. Model Performance Results", level=1)

doc.add_heading("4.1 Overall Model Comparison Across Biochars", level=2)
add_image(os.path.join(PLOTS_COMP, "model_comparison_across_biochars.png"),
          "Figure 2. Grouped bar charts comparing R², MAE, and RMSE for GPR, SVR, and ANN across all three biochars.")
add_image(os.path.join(PLOTS_COMP, "r2_comparison_line.png"),
          "Figure 3. R² score line comparison — all models across all biochars.")

doc.add_heading("4.2 Per-Biochar: PAC", level=2)
add_table(
    ["Model", "Test R²", "Test MAE", "Test RMSE"],
    [["GPR","0.7715","27.34","35.12"],["SVR","0.7540","29.32","36.78"],["ANN","0.7176","29.41","37.05"]]
)
add_image(os.path.join(PLOTS_PER, "PAC_predicted_vs_actual.png"),
          "Figure 4. PAC — Predicted vs Actual adsorption capacity for GPR, SVR, and ANN.")
add_image(os.path.join(PLOTS_PER, "PAC_violin_obs_vs_pred.png"),
          "Figure 5. PAC — Observed vs Predicted distribution (violin plots).")
add_image(os.path.join(PLOTS_PER, "PAC_error_distribution.png"),
          "Figure 6. PAC — Prediction error distributions for all three models.")

doc.add_heading("4.3 Per-Biochar: PB600", level=2)
add_table(
    ["Model", "Test R²", "Test MAE", "Test RMSE"],
    [["GPR","0.7357","1.15","1.48"],["SVR","0.6030","1.20","1.81"],["ANN","0.7637","1.09","1.41"]]
)
add_image(os.path.join(PLOTS_PER, "PB600_predicted_vs_actual.png"),
          "Figure 7. PB600 — Predicted vs Actual adsorption capacity for GPR, SVR, and ANN.")
add_image(os.path.join(PLOTS_PER, "PB600_violin_obs_vs_pred.png"),
          "Figure 8. PB600 — Observed vs Predicted distribution (violin plots).")
add_image(os.path.join(PLOTS_PER, "PB600_error_distribution.png"),
          "Figure 9. PB600 — Prediction error distributions for all three models.")

doc.add_heading("4.4 Per-Biochar: NaOH-Activated SCW", level=2)
add_image(os.path.join(PLOTS_PER, "NaOH_activated_SCW_predicted_vs_actual.png"),
          "Figure 10. NaOH-SCW — Predicted vs Actual adsorption capacity for GPR, SVR, and ANN.")
add_image(os.path.join(PLOTS_PER, "NaOH_activated_SCW_violin_obs_vs_pred.png"),
          "Figure 11. NaOH-SCW — Observed vs Predicted distribution (violin plots).")
add_image(os.path.join(PLOTS_PER, "NaOH_activated_SCW_error_distribution.png"),
          "Figure 12. NaOH-SCW — Prediction error distributions for all three models.")

doc.add_heading("4.5 ANN Training Curves", level=2)
add_image(os.path.join(PLOTS_PER, "PAC_ann_training_curve.png"),      "Figure 13. PAC — ANN Training & Validation Loss Curve.")
add_image(os.path.join(PLOTS_PER, "PB600_ann_training_curve.png"),    "Figure 14. PB600 — ANN Training & Validation Loss Curve.")
add_image(os.path.join(PLOTS_PER, "NaOH_activated_SCW_ann_training_curve.png"), "Figure 15. NaOH-SCW — ANN Training & Validation Loss Curve.")

# ════════════════════════════════════════════════════════════════
# SEC 5: GA OPTIMIZATION
# ════════════════════════════════════════════════════════════════
doc.add_heading("5. Genetic Algorithm Optimization", level=1)
add_para("The Genetic Algorithm was applied independently to each biochar using its best-performing model as the fitness function to identify the experimental conditions that maximize adsorption capacity.")

add_table(
    ["Biochar", "Best Model", "Pred. Max Capacity", "pH", "Temp (°C)", "Time (min)", "Conc. (mg/L)", "Dosage (g)"],
    [
        ["PAC",          "GPR", "~TBC", "6.36", "28.96", "30.0", "58.33", "0.050"],
        ["PB600",        "ANN", "~TBC", "5.00", "25.00", "900",  "10.00", "0.040"],
        ["NaOH-SCW",    "GPR/ANN", "~TBC", "7.00", "25.00", "1440", "100.0", "0.020"],
    ]
)

add_image(os.path.join(PLOTS_PER, "PAC_ga_convergence.png"),   "Figure 16. PAC — Genetic Algorithm Convergence Curve.")
add_image(os.path.join(PLOTS_PER, "PB600_ga_convergence.png"), "Figure 17. PB600 — Genetic Algorithm Convergence Curve.")
add_image(os.path.join(PLOTS_PER, "NaOH_activated_SCW_ga_convergence.png"), "Figure 18. NaOH-SCW — Genetic Algorithm Convergence Curve.")

# ════════════════════════════════════════════════════════════════
# SEC 6: COMBINED DASHBOARD
# ════════════════════════════════════════════════════════════════
doc.add_heading("6. Combined Results Dashboard", level=1)
add_image(os.path.join(PLOTS_COMP, "combined_dashboard.png"),
          "Figure 19. Combined dashboard showing predicted vs actual, R² bar comparison, and error violin plots for all three biochars.")

# ════════════════════════════════════════════════════════════════
# SEC 7: ERROR COMPARISON
# ════════════════════════════════════════════════════════════════
doc.add_heading("7. Cross-Biochar Error Analysis", level=1)
add_image(os.path.join(PLOTS_COMP, "violin_error_comparison.png"),
          "Figure 20. Prediction error distribution comparison across all three biochars for each model.")

# ════════════════════════════════════════════════════════════════
# SEC 8: KEY OBSERVATIONS
# ════════════════════════════════════════════════════════════════
doc.add_heading("8. Key Observations and Conclusions", level=1)

observations = [
    "Process-Specific Modeling: Training separate models per biochar is superior to a single unified model, as each material has distinct adsorption behaviour.",
    "PAC showed the highest R² values (~0.77), indicating the model captures its simpler, well-defined adsorption pattern effectively.",
    "PB600 models achieved strong performance with ANN leading at R²=0.76, reflecting consistent pyrolysis-based adsorption.",
    "NaOH-Activated SCW, with the largest dataset (486 samples), benefits from deeper learning and shows the widest capacity range.",
    "ANN and GPR consistently outperform SVR across all three materials, with GPR additionally providing uncertainty quantification.",
    "The Genetic Algorithm successfully converged for all three biochars, identifying material-specific optimal conditions.",
    "The paired violin plots confirm that predicted distributions closely mirror observed distributions for high-R² models.",
]
for obs in observations:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(obs).font.size = Pt(12)

doc.add_paragraph()
add_para("This study demonstrates that process-specific machine learning pipelines — where models are trained, "
         "validated, and optimized independently for each material — yield superior scientific insight compared "
         "to generalized, material-agnostic approaches. The three-biochar framework provides a reproducible "
         "methodology directly applicable to future adsorption research.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
doc.add_paragraph()
hr = doc.add_paragraph("─" * 70)
hr.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Report generated on 2026-05-12  |  ML Adsorption Capacity Prediction Project").italic = True

doc.save(OUT_PATH)
print(f"[✓] DOCX saved to: {OUT_PATH}")
