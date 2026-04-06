from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler


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

    # Power transform (distribution shaping)
    power_transform: str = "none"  # none|yeo-johnson

    # Outlier handling (applied after scaling if scaling is enabled)
    clip: float | None = None  # e.g. 5.0 => clip to [-5, 5]

    # Winsorization (clamp to per-feature quantiles fitted on baseline)
    winsorize: tuple[float, float] | None = None  # e.g. (0.01, 0.99)

    # Feature filtering (fitted on baseline)
    min_variance: float | None = None  # drop features with variance below threshold

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

    # --- Feature filtering (baseline variance) ---
    selected_features = list(feature_columns)
    if cfg.min_variance is not None:
        variances = baseline_out[selected_features].var(axis=0, ddof=0, numeric_only=True)
        keep = variances[variances >= float(cfg.min_variance)].index.tolist()
        # Ensure we don't accidentally drop everything.
        if len(keep) == 0:
            keep = selected_features
        selected_features = keep
        artifacts["selected_features"] = selected_features

    # --- Power transform ---
    if cfg.power_transform not in {"none", "yeo-johnson"}:
        raise ValueError("cfg.power_transform must be one of: none, yeo-johnson")
    pt = None
    if cfg.power_transform == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)

    if pt is not None:
        pt.fit(baseline_out[selected_features].to_numpy(dtype=float))
        artifacts["power_transformer"] = pt
        baseline_out.loc[:, selected_features] = pt.transform(
            baseline_out[selected_features].to_numpy(dtype=float)
        )
        probes_out.loc[:, selected_features] = pt.transform(
            probes_out[selected_features].to_numpy(dtype=float)
        )

    # --- Scaling ---
    if cfg.scaler not in {"none", "standard", "robust"}:
        raise ValueError("cfg.scaler must be one of: none, standard, robust")

    scaler = None
    if cfg.scaler == "standard":
        scaler = StandardScaler()
    elif cfg.scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True)

    if scaler is not None:
        scaler.fit(baseline_out[selected_features].to_numpy(dtype=float))
        artifacts["scaler"] = scaler
        baseline_out.loc[:, selected_features] = scaler.transform(
            baseline_out[selected_features].to_numpy(dtype=float)
        )
        probes_out.loc[:, selected_features] = scaler.transform(
            probes_out[selected_features].to_numpy(dtype=float)
        )

    # --- Winsorization ---
    if cfg.winsorize is not None:
        lo_q, hi_q = cfg.winsorize
        lo_q = float(lo_q)
        hi_q = float(hi_q)
        if not (0.0 <= lo_q < hi_q <= 1.0):
            raise ValueError("cfg.winsorize must be a (low, high) quantile in [0,1]")

        lo = baseline_out[selected_features].quantile(lo_q, numeric_only=True)
        hi = baseline_out[selected_features].quantile(hi_q, numeric_only=True)
        artifacts["winsorize_low"] = lo
        artifacts["winsorize_high"] = hi
        baseline_out.loc[:, selected_features] = baseline_out[selected_features].clip(lo, hi, axis=1)
        probes_out.loc[:, selected_features] = probes_out[selected_features].clip(lo, hi, axis=1)

    # --- Clipping ---
    if cfg.clip is not None:
        clip = float(cfg.clip)
        baseline_out.loc[:, selected_features] = np.clip(
            baseline_out[selected_features].to_numpy(dtype=float), -clip, clip
        )
        probes_out.loc[:, selected_features] = np.clip(
            probes_out[selected_features].to_numpy(dtype=float), -clip, clip
        )

    # --- Absolute value ---
    if cfg.abs_values:
        baseline_out.loc[:, selected_features] = np.abs(
            baseline_out[selected_features].to_numpy(dtype=float)
        )
        probes_out.loc[:, selected_features] = np.abs(
            probes_out[selected_features].to_numpy(dtype=float)
        )

    artifacts["feature_columns_used"] = selected_features
    return baseline_out, probes_out, artifacts
