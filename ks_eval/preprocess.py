from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    """Configurable preprocessing switches.

    All steps are optional. The pipeline is intentionally simple and transparent.
    Fit operations are done on a per-subject *baseline* set to avoid probe leakage.
    """

    # Missing values
    impute: str = "none"  # none|mean|median

    # Scaling
    scaler: str = "none"  # none|standard|robust

    # Outlier handling (applied after scaling if scaling is enabled)
    clip: float | None = None  # e.g. 5.0 => clip to [-5, 5]

    # Convenience guardrails
    abs_values: bool = False  # make all features non-negative (not usually needed)


def preprocess_fit_transform(
    baseline_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    feature_columns: list[str],
    cfg: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit preprocessing on baseline, transform both baseline and probes.

    Returns: (baseline_out, probes_out, artifacts)
    where artifacts contains fitted parameters (imputer stats / scaler) for debugging.
    """

    baseline_out = baseline_df.copy()
    probes_out = probes_df.copy()

    artifacts: dict = {"config": cfg}

    # --- Imputation ---
    if cfg.impute not in {"none", "mean", "median"}:
        raise ValueError("cfg.impute must be one of: none, mean, median")
    if cfg.impute != "none":
        if cfg.impute == "mean":
            fill = baseline_out[feature_columns].mean(numeric_only=True)
        else:
            fill = baseline_out[feature_columns].median(numeric_only=True)

        artifacts["impute_values"] = fill
        baseline_out.loc[:, feature_columns] = baseline_out[feature_columns].fillna(fill)
        probes_out.loc[:, feature_columns] = probes_out[feature_columns].fillna(fill)

    # --- Scaling ---
    if cfg.scaler not in {"none", "standard", "robust"}:
        raise ValueError("cfg.scaler must be one of: none, standard, robust")

    scaler = None
    if cfg.scaler == "standard":
        scaler = StandardScaler()
    elif cfg.scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True)

    if scaler is not None:
        scaler.fit(baseline_out[feature_columns].to_numpy(dtype=float))
        artifacts["scaler"] = scaler
        baseline_out.loc[:, feature_columns] = scaler.transform(
            baseline_out[feature_columns].to_numpy(dtype=float)
        )
        probes_out.loc[:, feature_columns] = scaler.transform(
            probes_out[feature_columns].to_numpy(dtype=float)
        )

    # --- Clipping ---
    if cfg.clip is not None:
        clip = float(cfg.clip)
        baseline_out.loc[:, feature_columns] = np.clip(
            baseline_out[feature_columns].to_numpy(dtype=float), -clip, clip
        )
        probes_out.loc[:, feature_columns] = np.clip(
            probes_out[feature_columns].to_numpy(dtype=float), -clip, clip
        )

    # --- Absolute value ---
    if cfg.abs_values:
        baseline_out.loc[:, feature_columns] = np.abs(
            baseline_out[feature_columns].to_numpy(dtype=float)
        )
        probes_out.loc[:, feature_columns] = np.abs(
            probes_out[feature_columns].to_numpy(dtype=float)
        )

    return baseline_out, probes_out, artifacts
