from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_brain_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a brain CSV and normalize its basic schema."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "subject_id" in df.columns and "subject" not in df.columns:
        df = df.rename(columns={"subject_id": "subject"})

    if "emotion" not in df.columns:
        raise ValueError(f"{path} is missing the required 'emotion' column.")

    df["emotion"] = df["emotion"].astype(str).str.lower()

    numeric_cols = [
        column
        for column in df.columns
        if column not in {"subject", "emotion"} and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_cols:
        raise ValueError(f"{path} does not contain any numeric brain-feature columns.")

    X = df[numeric_cols].to_numpy()
    labels = np.array(sorted(df["emotion"].dropna().unique()))
    return X, labels, df
