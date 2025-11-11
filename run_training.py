"""Runner to execute the end-to-end pipeline:
- load data
- clean data
- detect target (or use last column)
- train RandomForest
- save model and report

Run from project root. Outputs are written to `results/`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from src import train_pipeline


def main():
    data_path = DATA_DIR / "Updated Warehouse Data.csv"
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(2)

    print("Running full pipeline (load -> clean -> detect target -> train -> save)...")
    model, metrics = train_pipeline(str(data_path), results_dir=str(RESULTS_DIR))

    print("Training complete.")
    print(f"Model saved to: {RESULTS_DIR / 'warehouse_model.pkl'}")
    print(f"Report saved to: {RESULTS_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
