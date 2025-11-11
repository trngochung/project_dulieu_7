"""Model training helpers.

Provides a simple RandomForest training function appropriate for quick
experiments. The function returns the trained model and basic evaluation
information; the caller is responsible for serializing the model if desired.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                        random_state: int = 42, n_estimators: int = 100,
                        save_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Train a RandomForestClassifier and return model + metrics.

    Returns
    -------
    model: trained estimator
    metrics: dict with keys: accuracy, report, X_test, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=(y if len(np.unique(y))>1 else None)
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    if save_path:
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    return model, metrics


def train_pipeline(data_path: str,
                   results_dir: Optional[str] = None,
                   target: Optional[str] = None,
                   drop_thresh: float = 0.5,
                   random_state: int = 42,
                   n_estimators: int = 100) -> Tuple[Any, Dict[str, Any]]:
    """End-to-end: load -> clean -> detect target -> train -> save outputs.

    Parameters
    ----------
    data_path: path to input data (CSV)
    results_dir: directory where cleaned file, model and report will be saved. If
        None, defaults to the folder of data_path.
    target: optional name of target column. If None, function will attempt to
        detect common names and fall back to the last column.
    drop_thresh: threshold for dropping columns with too many missing values.

    Returns (model, metrics)
    """
    # Local imports to avoid circular import at module import time
    from pathlib import Path
    from .data_cleaning import load_data, clean_df, detect_target, save_cleaned
    from .report_generator import save_text_report

    results_dir = Path(results_dir) if results_dir else Path(data_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_data(data_path)

    # Clean
    df_clean = clean_df(df, drop_thresh=drop_thresh)
    cleaned_path = results_dir / "warehouse_cleaned.csv"
    save_cleaned(df_clean, str(cleaned_path))

    # Detect target if not provided
    if target is None:
        target = detect_target(df_clean)
        if target is None:
            target = df_clean.columns[-1]

    if target not in df_clean.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    # Train
    model_path = results_dir / "warehouse_model.pkl"
    model, metrics = train_random_forest(X, y, random_state=random_state,
                                         n_estimators=n_estimators,
                                         save_path=str(model_path))

    # Persist report
    report_metrics = {
        "accuracy": metrics.get("accuracy"),
        "report": metrics.get("report"),
        "model_path": str(model_path),
    }
    report_path = results_dir / "report.txt"
    save_text_report(report_metrics, str(report_path))

    # Return for programmatic use
    return model, metrics
