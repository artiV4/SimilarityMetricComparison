from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"subject", "sessionIndex", "rep"}


@dataclass(frozen=True)
class Dataset:
    """In-memory representation of the DSL strong-password dataset."""

    df: pd.DataFrame
    feature_columns: list[str]


def load_dsl_strong_password_csv(csv_path: str | Path) -> Dataset:
    """Load the DSL-StrongPasswordData.csv dataset.

    Assumptions (based on common DSL Strong Password dataset schema):
    - Rows are samples.
    - `subject` is the user id.
    - `(sessionIndex, rep)` defines the temporal ordering of samples within a subject.

    Returns a Dataset with numeric feature columns inferred from the CSV.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Ensure ordering columns are numeric where possible.
    for col in ["sessionIndex", "rep"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Infer features: everything except id/order columns.
    non_feature = {"subject", "sessionIndex", "rep"}
    feature_columns = [c for c in df.columns if c not in non_feature]

    # Coerce features to numeric (keystroke timings should be floats).
    for c in feature_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return Dataset(df=df, feature_columns=feature_columns)


def iter_subject_samples(dataset: Dataset) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield (subject, subject_df) ordered by time for each subject."""

    df = dataset.df

    # Stable deterministic user order.
    for subject in sorted(df["subject"].unique().tolist()):
        sdf = df[df["subject"] == subject].copy()
        sdf = sdf.sort_values(["sessionIndex", "rep"], kind="mergesort")
        yield subject, sdf


def to_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Convert a subject dataframe slice to an (n_samples, n_features) numpy array."""

    return df[feature_columns].to_numpy(dtype=float, copy=True)
