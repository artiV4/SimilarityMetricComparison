from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _as_vector(row: pd.Series, feature_columns: list[str]) -> np.ndarray:
    return row[feature_columns].to_numpy(dtype=float)


def _baseline_matrix(baseline_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return baseline_df[feature_columns].to_numpy(dtype=float)


class Scorer(Protocol):
    """Interface for a pluggable similarity/distance scorer.

    Contract:
    - fit() is called once per subject with that subject's baseline enrollment set.
    - score() is called for each probe row.

    Returned score should follow the scorer's convention:
    - similarity: higher is more similar
    - distance: lower is more similar
    """

    name: str
    greater_is_better: bool

    def fit(self, baseline_df: pd.DataFrame, feature_columns: list[str]) -> "Scorer":
        ...

    def score(self, probe_row: pd.Series, feature_columns: list[str]) -> float:
        ...


@dataclass
class EuclideanDistanceScorer:
    name: str = "euclidean"
    greater_is_better: bool = False

    mu_: np.ndarray | None = None

    def fit(self, baseline_df: pd.DataFrame, feature_columns: list[str]) -> "EuclideanDistanceScorer":
        X = _baseline_matrix(baseline_df, feature_columns)
        self.mu_ = np.nanmean(X, axis=0)
        return self

    def score(self, probe_row: pd.Series, feature_columns: list[str]) -> float:
        if self.mu_ is None:
            raise RuntimeError("Scorer not fit")
        x = _as_vector(probe_row, feature_columns)
        d = x - self.mu_
        return float(np.sqrt(np.dot(d, d)))


@dataclass
class CosineSimilarityScorer:
    name: str = "cosine"
    greater_is_better: bool = True

    mu_: np.ndarray | None = None
    mu_norm_: float | None = None

    def fit(self, baseline_df: pd.DataFrame, feature_columns: list[str]) -> "CosineSimilarityScorer":
        X = _baseline_matrix(baseline_df, feature_columns)
        mu = np.nanmean(X, axis=0)
        self.mu_ = mu
        self.mu_norm_ = float(np.linalg.norm(mu))
        return self

    def score(self, probe_row: pd.Series, feature_columns: list[str]) -> float:
        if self.mu_ is None or self.mu_norm_ is None:
            raise RuntimeError("Scorer not fit")
        x = _as_vector(probe_row, feature_columns)
        x_norm = float(np.linalg.norm(x))
        if x_norm == 0.0 or self.mu_norm_ == 0.0:
            return 0.0
        return float(np.dot(x, self.mu_) / (x_norm * self.mu_norm_))


@dataclass
class MahalanobisDistanceScorer:
    """Mahalanobis distance using Ledoit-Wolf shrinkage covariance (baseline-fitted)."""

    name: str = "mahalanobis"
    greater_is_better: bool = False

    mu_: np.ndarray | None = None
    inv_cov_: np.ndarray | None = None

    def fit(
        self, baseline_df: pd.DataFrame, feature_columns: list[str]
    ) -> "MahalanobisDistanceScorer":
        X = _baseline_matrix(baseline_df, feature_columns)
        self.mu_ = np.nanmean(X, axis=0)

        # LedoitWolf can't handle NaNs; assume preprocessing handled them.
        cov = LedoitWolf().fit(X).covariance_
        self.inv_cov_ = np.linalg.pinv(cov)
        return self

    def score(self, probe_row: pd.Series, feature_columns: list[str]) -> float:
        if self.mu_ is None or self.inv_cov_ is None:
            raise RuntimeError("Scorer not fit")
        x = _as_vector(probe_row, feature_columns)
        d = x - self.mu_
        return float(np.sqrt(d.T @ self.inv_cov_ @ d))


@dataclass
class GaussianLogLikelihoodScorer:
    """Multivariate Gaussian log-likelihood under baseline-fitted mean/cov.

    Uses Ledoit-Wolf covariance shrinkage for stability.
    Higher score => more likely => more similar.
    """

    name: str = "gaussian_ll"
    greater_is_better: bool = True

    mu_: np.ndarray | None = None
    inv_cov_: np.ndarray | None = None
    log_det_cov_: float | None = None
    d_: int | None = None

    def fit(
        self, baseline_df: pd.DataFrame, feature_columns: list[str]
    ) -> "GaussianLogLikelihoodScorer":
        X = _baseline_matrix(baseline_df, feature_columns)
        self.mu_ = np.nanmean(X, axis=0)
        self.d_ = int(X.shape[1])

        cov = LedoitWolf().fit(X).covariance_
        inv_cov = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        # Due to numerical issues, sign could be <=0 for near-singular matrices.
        if sign <= 0:
            # Fall back to pseudo-det via inv-cov trace-ish approximation isn't great;
            # this keeps LL finite and comparable.
            logdet = float(np.log(np.maximum(np.finfo(float).eps, np.linalg.det(cov))))

        self.inv_cov_ = inv_cov
        self.log_det_cov_ = float(logdet)
        return self

    def score(self, probe_row: pd.Series, feature_columns: list[str]) -> float:
        if (
            self.mu_ is None
            or self.inv_cov_ is None
            or self.log_det_cov_ is None
            or self.d_ is None
        ):
            raise RuntimeError("Scorer not fit")
        x = _as_vector(probe_row, feature_columns)
        d = x - self.mu_
        quad = float(d.T @ self.inv_cov_ @ d)
        # constant term included so LL is comparable across different d.
        return float(-0.5 * (quad + self.log_det_cov_ + self.d_ * np.log(2.0 * np.pi)))


def default_scorers() -> list[Scorer]:
    return [
        EuclideanDistanceScorer(),
        CosineSimilarityScorer(),
        MahalanobisDistanceScorer(),
        GaussianLogLikelihoodScorer(),
    ]
