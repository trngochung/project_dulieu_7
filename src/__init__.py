"""src package for my_project_py

This package contains small, focused modules for:
- data cleaning (``data_cleaning.py``)
- model training (``model_training.py``)
- evaluation (``evaluation.py``)
- report generation (``report_generator.py``)

This module exposes a small, stable surface so callers can do::

    from src import load_data, train_random_forest

The implementation imports are intentionally lightweight.
"""

from .data_cleaning import load_data, clean_df, detect_target, save_cleaned
from .model_training import train_random_forest, train_pipeline
from .evaluation import evaluate_model
from .report_generator import save_text_report

__all__ = [
    "load_data",
    "clean_df",
    "detect_target",
    "save_cleaned",
    "train_random_forest",
    "train_pipeline",
    "evaluate_model",
    "save_text_report",
]

__version__ = "0.1.0"
