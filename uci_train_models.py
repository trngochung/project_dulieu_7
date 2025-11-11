"""UCI-standard model comparison runner.

Trains multiple classification algorithms on cleaned warehouse data:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Gradient Boosting
- K-Nearest Neighbors (KNN)

Evaluates each using accuracy, precision, recall, F1 and produces a comparison report.
Run from project root: python uci_train_models.py
"""

from pathlib import Path
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from src import train_pipeline


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    return metrics, y_pred


def train_uci_models(data_path, results_dir=None, target=None, test_size=0.2, random_state=42):
    """Train multiple algorithms and compare them."""
    
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("UCI-STANDARD MODEL COMPARISON")
    print("="*70)
    
    # Step 1: Run the pipeline to get clean data
    print("\n[1/3] Loading and cleaning data...")
    model_temp, metrics_temp = train_pipeline(data_path, results_dir=str(results_dir), target=target)
    
    # Load cleaned data for fresh splits
    from src import load_data, clean_df, detect_target
    df = load_data(data_path)
    df_clean = clean_df(df)
    
    if target is None:
        target = detect_target(df_clean)
        if target is None:
            target = df_clean.columns[-1]
    
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    print(f"  ✓ Data shape: {X.shape}, target: {target} ({y.nunique()} classes)")
    
    # Step 2: Split data (train/test)
    print("\n[2/3] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Scale features for SVM and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Train models
    print("\n[3/3] Training models...")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"  → Training {model_name}...", end=" ")
        
        # Use scaled data for SVM and KNN
        if "SVM" in model_name or "KNN" in model_name:
            model.fit(X_train_scaled, y_train)
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test, model_name)
        else:
            model.fit(X_train, y_train)
            metrics, y_pred = evaluate_model(model, X_test, y_test, model_name)
        
        results.append(metrics)
        predictions[model_name] = y_pred
        
        print(f"✓ Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # Save individual model
        model_path = results_dir / f"model_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS (sorted by F1 score)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save detailed report
    report_path = results_dir / "uci_comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("UCI-STANDARD MODEL COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Data: {data_path}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
        f.write(f"Classes: {y.nunique()}\n\n")
        
        f.write("RESULTS SUMMARY (sorted by F1 score):\n")
        f.write("-"*70 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        # Best model
        best_model = results_df.iloc[0]
        f.write("BEST MODEL:\n")
        f.write(f"  Name: {best_model['model']}\n")
        f.write(f"  Accuracy: {best_model['accuracy']:.4f}\n")
        f.write(f"  Precision: {best_model['precision']:.4f}\n")
        f.write(f"  Recall: {best_model['recall']:.4f}\n")
        f.write(f"  F1 Score: {best_model['f1']:.4f}\n\n")
        
        # Detailed classification reports
        f.write("DETAILED CLASSIFICATION REPORTS:\n")
        f.write("="*70 + "\n\n")
        
        for model_name, y_pred in predictions.items():
            f.write(f"--- {model_name} ---\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))
            f.write("\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save comparison CSV
    csv_path = results_dir / "uci_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    return results_df, predictions


if __name__ == "__main__":
    data_path = DATA_DIR / "Updated Warehouse Data.csv"
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)
    
    train_uci_models(str(data_path), results_dir=str(RESULTS_DIR))
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
