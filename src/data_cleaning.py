"""Data cleaning utilities.

Functions provided:
- load_data(file_path) -> pd.DataFrame
- clean_df(df, drop_thresh=0.5) -> pd.DataFrame
- detect_target(df, candidates=None) -> Optional[str]
- save_cleaned(df, path)

Designed to be small and non-interactive (no input() calls) so it can be imported
by other scripts. If automatic detection fails, functions return None and the
caller may decide how to proceed (ask user, choose a column, skip training).
"""
from typing import Optional, Sequence
import pandas as pd
import numpy as np


def load_data(file_path: str, **read_kwargs) -> pd.DataFrame:
    """Read CSV (or other) into a DataFrame.

    Parameters
    ----------
    file_path: str
        Path to the data file (CSV by default).
    read_kwargs: passed to pd.read_csv
    """
    return pd.read_csv(file_path, **read_kwargs)


def clean_df(df: pd.DataFrame, drop_thresh: float = 0.5) -> pd.DataFrame:
    """Basic cleaning pipeline:
    - drop columns with > drop_thresh missing rate
    - fill numeric missing with median
    - fill categorical missing with mode
    - drop duplicates
    - try converting object columns to numeric when possible
    - simple LabelEncoding for remaining object columns
    - remove outliers (IQR) for numeric columns

    Returns cleaned DataFrame (copy).
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > drop_thresh].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            # if empty column (all nan), mode() may fail; guard it
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except Exception:
                df[col] = df[col].fillna("")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Try converting object columns to numeric where possible
    for col in df.select_dtypes(include=[object]).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Simple label encoding for remaining object columns
    from sklearn.preprocessing import LabelEncoder

    cat_cols = df.select_dtypes(include=[object]).columns.tolist()
    if cat_cols:
        le = LabelEncoder()
        for col in cat_cols:
            # convert to string first to avoid errors on mixed types
            df[col] = df[col].astype(str)
            try:
                df[col] = le.fit_transform(df[col])
            except Exception:
                # if encoding fails, leave as-is
                pass

    # Remove outliers using IQR on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Reset index after filtering
    df = df.reset_index(drop=True)
    return df


def detect_target(df: pd.DataFrame, candidates: Optional[Sequence[str]] = None) -> Optional[str]:
    """Try to detect a target column using common names.

    Returns the column name if found, otherwise None.
    """
    if candidates is None:
        candidates = ["target", "label", "class", "output", "y", "result", "prediction", "status"]

    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def save_cleaned(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
