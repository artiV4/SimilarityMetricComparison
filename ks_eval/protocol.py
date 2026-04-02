from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubjectSplit:
    subject: str
    baseline_df: pd.DataFrame
    probes_df: pd.DataFrame


def split_baseline_and_probes(
    subject_df: pd.DataFrame,
    *,
    baseline_fraction: float = 0.10,
    min_baseline: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single subject's ordered samples into baseline and sequential probes.

    Baseline is the first `baseline_fraction` of rows (at least `min_baseline`).
    Remaining rows are probes, preserving their order.
    """

    if not (0.0 < baseline_fraction < 1.0):
        raise ValueError("baseline_fraction must be in (0, 1)")

    n = len(subject_df)
    if n == 0:
        return subject_df.iloc[0:0], subject_df.iloc[0:0]

    n_base = int(np.floor(n * baseline_fraction))
    n_base = max(min_baseline, n_base)
    n_base = min(n_base, n)  # safety

    baseline = subject_df.iloc[:n_base].copy()
    probes = subject_df.iloc[n_base:].copy()
    return baseline, probes
