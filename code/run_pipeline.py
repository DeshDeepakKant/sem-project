"""
Master Pipeline Runner
Executes all steps in sequence:
  01 -> Data Preprocessing
  02 -> Train GPR
  03 -> Train SVR
  04 -> Train ANN
  05 -> Visualization
  06 -> Genetic Algorithm Optimization
  07 -> Report Generation
"""

import subprocess
import sys
import os
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

steps = [
    ('01_data_preprocessing.py', 'Data Preprocessing'),
    ('02_train_gpr.py', 'Gaussian Process Regression'),
    ('03_train_svr.py', 'Support Vector Regression'),
    ('04_train_ann.py', 'Artificial Neural Network'),
    ('05_visualization.py', 'Visualization'),
    ('06_genetic_algorithm.py', 'Genetic Algorithm Optimization'),
    ('07_generate_report.py', 'Report Generation'),
]

print("=" * 70)
print("  ML ADSORPTION CAPACITY PREDICTION — FULL PIPELINE")
print("=" * 70)

total_start = time.time()

for i, (script, description) in enumerate(steps, 1):
    script_path = os.path.join(CODE_DIR, script)
    print(f"\n{'#' * 70}")
    print(f"# STEP {i}/{len(steps)}: {description}")
    print(f"# Script: {script}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Step {i} ({script}) failed with return code {result.returncode}")
        print("Aborting pipeline.")
        sys.exit(1)
    else:
        print(f"\n[DONE] Step {i} completed in {elapsed:.2f}s")

total_time = time.time() - total_start
print(f"\n{'=' * 70}")
print(f"  FULL PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s ✓")
print(f"{'=' * 70}")
