from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ks_eval.preprocess import PreprocessConfig


def _preprocess_key(cfg: PreprocessConfig) -> str:
    # Keep this stable and human-readable.
    d = asdict(cfg)
    # Normalize tuples/lists to strings for CSV storage.
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = ",".join(str(x) for x in v)
    # Order keys for stability.
    return "|".join(f"{k}={d[k]}" for k in sorted(d.keys()))


def load_threshold(
    *,
    csv_path: str | Path,
    metric: str,
    fpr: float,
    baseline_fraction: float,
    min_baseline: int,
    impostor_per_subject: int,
    seed: int,
    preprocess: PreprocessConfig,
) -> float | None:
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if len(df) == 0:
        return None

    pre_key = _preprocess_key(preprocess)
    filt = (
        (df["metric"] == metric)
        & (df["fpr"] == float(fpr))
        & (df["baseline_fraction"] == float(baseline_fraction))
        & (df["min_baseline"] == int(min_baseline))
        & (df["impostor_per_subject"] == int(impostor_per_subject))
        & (df["seed"] == int(seed))
        & (df["preprocess_key"] == pre_key)
    )
    hit = df.loc[filt]
    if len(hit) == 0:
        return None
    # If duplicates exist, take the last one.
    return float(hit.iloc[-1]["threshold"])


def upsert_threshold(
    *,
    csv_path: str | Path,
    metric: str,
    fpr: float,
    baseline_fraction: float,
    min_baseline: int,
    impostor_per_subject: int,
    seed: int,
    preprocess: PreprocessConfig,
    threshold: float,
) -> None:
    p = Path(csv_path)
    pre_key = _preprocess_key(preprocess)

    row = {
        "metric": metric,
        "fpr": float(fpr),
        "baseline_fraction": float(baseline_fraction),
        "min_baseline": int(min_baseline),
        "impostor_per_subject": int(impostor_per_subject),
        "seed": int(seed),
        "preprocess_key": pre_key,
        "threshold": float(threshold),
    }

    if p.exists():
        df = pd.read_csv(p)
    else:
        df = pd.DataFrame(
            columns=[
                "metric",
                "fpr",
                "baseline_fraction",
                "min_baseline",
                "impostor_per_subject",
                "seed",
                "preprocess_key",
                "threshold",
            ]
        )

    # Drop any existing matching row (so we keep one per key).
    filt = (
        (df["metric"] == row["metric"])
        & (df["fpr"] == row["fpr"])
        & (df["baseline_fraction"] == row["baseline_fraction"])
        & (df["min_baseline"] == row["min_baseline"])
        & (df["impostor_per_subject"] == row["impostor_per_subject"])
        & (df["seed"] == row["seed"])
        & (df["preprocess_key"] == row["preprocess_key"])
    )
    if len(df) > 0:
        df = df.loc[~filt]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)
