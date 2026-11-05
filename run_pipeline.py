"""Single entry point for Bree's reproducible submission.

Examples (from any working directory):
    python run_pipeline.py                 # Verify environment, EDA, tests, validation
    python run_pipeline.py --finalize-test  # Finalize once, if not already finalized
    python run_pipeline.py --reproduce-final # Deliberate exact re-evaluation

The original dataset is committed. It is generated only if missing, or if
--regenerate is explicitly provided; Bree's unmodified generator remains intact.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data-analysis" / "loan_applications.csv"
FINAL_RESULT = ROOT / "modelling-experiments" / "outputs" / "test_metrics.csv"


def execute(*args: str) -> None:
    print("\n$", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-test", action="store_true",
                        help="Evaluate the held-out test if not yet finalized")
    parser.add_argument("--reproduce-final", action="store_true",
                        help="Deliberately rerun final evaluation with unchanged locked settings")
    parser.add_argument("--regenerate", action="store_true",
                        help="Explicitly recreate Bree's original CSV using its seed")
    args = parser.parse_args()
    if args.finalize_test and args.reproduce_final:
        parser.error("Choose either --finalize-test or --reproduce-final.")
    if not DATASET.exists() or args.regenerate:
        execute("data-analysis/run_generation.py")
    else:
        print(f"Using committed dataset: {DATASET}", flush=True)
    execute("data-analysis/analysis.py")
    execute("-m", "unittest", "discover", "-s", "modelling-experiments/tests", "-v")
    model_cmd = ["modelling-experiments/run_experiments.py"]
    if args.reproduce_final:
        model_cmd.extend(["--finalize-test", "--overwrite-test"])
    elif args.finalize_test and not FINAL_RESULT.exists():
        model_cmd.append("--finalize-test")
    elif args.finalize_test:
        print("Final test outputs already exist. Preserving them. Use --reproduce-final for a verified rerun.", flush=True)
    execute(*model_cmd)
    print("\nPASS: generation/EDA/test suite/model evaluation pipeline completed.")
    print("The submitted test results are in:", FINAL_RESULT)


if __name__ == "__main__":
    main()
