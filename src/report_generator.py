"""Simple report generation utilities.

Currently writes a plain-text report. This is intentionally minimal — you can
replace or extend it later to produce Word/PDF/HTML as needed.
"""
from typing import Dict, Any


def save_text_report(metrics: Dict[str, Any], path: str) -> None:
    """Write a simple text report to `path`.

    `metrics` can include keys like 'accuracy', 'report', 'model_path', etc.
    """
    lines = []
    lines.append("Model evaluation report")
    lines.append("=" * 60)
    if "accuracy" in metrics:
        lines.append(f"Accuracy: {metrics['accuracy']}")
    if "report" in metrics:
        lines.append("\nClassification report:\n")
        lines.append(str(metrics["report"]))
    if "model_path" in metrics:
        lines.append(f"\nSaved model: {metrics['model_path']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
