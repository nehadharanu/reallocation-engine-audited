"""
Run the full reallocation-engine pipeline end to end, in order.
Each script is run as a subprocess so a failure in one script is caught
and reported without crashing the whole pipeline.
"""
import subprocess
import sys
import time

PIPELINE = [
    ("Sample data", "src/sample_data.py"),
    ("GIGO gate", "src/gigo_gate.py"),
    ("Engine (train + score)", "src/engine.py"),
    ("Bias audit", "src/bias_audit.py"),
    ("Explainability (SHAP)", "src/explainability.py"),
    ("Causal analysis", "src/causal_analysis.py"),
    ("Adversarial test", "src/adversarial_test.py"),
    ("Delegation gate", "src/delegation_gate.py"),
    ("Uncertainty communication", "src/uncertainty_communication.py"),
]

def main():
    results = []
    for name, path in PIPELINE:
        print("=" * 70)
        print(f"RUNNING: {name} ({path})")
        print("=" * 70)
        start = time.time()
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
        elapsed = time.time() - start
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"!!! FAILED: {name} (exit code {proc.returncode}) !!!")
            print(proc.stderr)
            results.append((name, "FAILED", elapsed, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""))
        else:
            results.append((name, "OK", elapsed, ""))
        print()

    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for name, status, elapsed, err in results:
        line = f"[{status}] {name} ({elapsed:.1f}s)"
        if err:
            line += f" - {err}"
        print(line)

    n_failed = sum(1 for _, status, _, _ in results if status == "FAILED")
    print()
    if n_failed == 0:
        print("All scripts completed successfully.")
    else:
        print(f"{n_failed} script(s) failed. See output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
