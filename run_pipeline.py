"""
Main Pipeline Orchestrator — Executes all steps for the 3-biochar analysis.
"""
import subprocess
import os
import time

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')

scripts = [
    '01_data_preprocessing.py',
    '02_train_all_models.py',
    '03_all_plots.py',
    '04_genetic_algorithm.py',
    '05_generate_report.py'
]

def run_script(script_name):
    script_path = os.path.join(CODE_DIR, script_name)
    print(f"\n\n>>> STARTING: {script_name}")
    start_time = time.time()
    try:
        # Using sys.executable to ensure we use the same environment
        import sys
        result = subprocess.run([sys.executable, script_path], check=True, text=True)
        elapsed = time.time() - start_time
        print(f">>> FINISHED: {script_name} in {elapsed:.1f} seconds ✓")
        return True
    except subprocess.CalledProcessError as e:
        print(f">>> FAILED: {script_name}")
        print(e)
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING COMPLETE ADSORPTION ML PIPELINE (3-BIOCHAR)")
    print("=" * 60)
    
    total_start = time.time()
    success_count = 0
    
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print("\n[CRITICAL] Pipeline halted due to error.")
            break
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED: {success_count}/{len(scripts)} scripts succeeded.")
    print(f"Total Time: {total_elapsed/60:.2f} minutes")
    print("=" * 60)
