from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ks_eval.data import Dataset, iter_subject_samples
from ks_eval.preprocess import PreprocessConfig, preprocess_fit_transform
from ks_eval.protocol import split_baseline_and_probes
from ks_eval.scoring import (
    CosineSimilarityScorer,
    EuclideanDistanceScorer,
    GaussianLogLikelihoodScorer,
    MahalanobisDistanceScorer,
    Scorer,
)


def make_scorer(metric: str) -> Scorer:
    """Create a scorer instance from a string identifier."""

    m = metric.strip().lower()
    if m == "euclidean":
        return EuclideanDistanceScorer()
    if m == "cosine":
        return CosineSimilarityScorer()
    if m == "mahalanobis":
        return MahalanobisDistanceScorer()
    if m in {"gaussian", "gaussian_ll"}:
        return GaussianLogLikelihoodScorer()
    raise ValueError(f"Unknown metric: {metric}")


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    false_positives: int
    true_positives: int
    false_negatives: int
    n_impostor_trials: int
    n_genuine_trials: int


def compute_trials(
    dataset: Dataset,
    *,
    scorer: Scorer,
    preprocess: PreprocessConfig,
    baseline_fraction: float = 0.10,
    min_baseline: int = 1,
    impostor_per_subject: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (genuine_scores, impostor_scores) for the provided scorer.

    For each subject:
    - fit preprocessing + scorer on that subject's baseline
    - score that subject's probes as genuine scores
    - sample some probes from other subjects as impostors against this subject template

    Score direction:
    - For similarities, higher is more similar.
    - For distances, lower is more similar.
    Use `scorer.greater_is_better` to interpret thresholds.
    """

    rng = np.random.default_rng(random_state)

    # Cache per-subject ordered/probe dataframes once.
    subjects: list[str] = []
    per_subject: dict[str, dict[str, pd.DataFrame]] = {}
    for subject, sdf in iter_subject_samples(dataset):
        baseline_df, probes_df = split_baseline_and_probes(
            sdf, baseline_fraction=baseline_fraction, min_baseline=min_baseline
        )
        if len(probes_df) == 0:
            continue
        subjects.append(subject)
        per_subject[subject] = {"baseline": baseline_df, "probes": probes_df}

    # Pre-extract all probe rows for impostor sampling (by subject).
    all_probe_rows: dict[str, pd.DataFrame] = {
        s: per_subject[s]["probes"] for s in subjects
    }

    genuine: list[float] = []
    impostor: list[float] = []

    for subject in subjects:
        baseline_df = per_subject[subject]["baseline"]
        probes_df = per_subject[subject]["probes"]

        baseline_pp, probes_pp, _ = preprocess_fit_transform(
            baseline_df, probes_df, dataset.feature_columns, preprocess
        )

        # Scorers are stateful per-subject; re-instantiate when possible.
        try:
            local_scorer = type(scorer)()  # type: ignore[call-arg]
        except Exception:
            local_scorer = scorer

        local_scorer = local_scorer.fit(baseline_pp, dataset.feature_columns)

        # Genuine distances.
        for _, probe_row in probes_pp.iterrows():
            genuine.append(local_scorer.score(probe_row, dataset.feature_columns))

        # Impostor trials: sample impostor_per_subject rows total for this subject.
        other_subjects = [s for s in subjects if s != subject]
        if not other_subjects:
            continue

        n_samples = min(
            impostor_per_subject,
            sum(len(all_probe_rows[s]) for s in other_subjects),
        )
        # Choose random (subject, row_index) pairs.
        for _k in range(n_samples):
            imp_subject = rng.choice(other_subjects)
            imp_df = all_probe_rows[imp_subject]
            imp_row = imp_df.iloc[int(rng.integers(0, len(imp_df)))]

            # Apply *this subject's* preprocessing transform to impostor row:
            # easiest is to run preprocess_fit_transform with probes including only this row.
            imp_row_df = pd.DataFrame([imp_row])
            _, imp_pp_df, _ = preprocess_fit_transform(
                baseline_df,
                imp_row_df,
                dataset.feature_columns,
                preprocess,
            )
            imp_pp_row = imp_pp_df.iloc[0]
            impostor.append(local_scorer.score(imp_pp_row, dataset.feature_columns))

    return np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)


def compute_trials_by_name(
    dataset: Dataset,
    *,
    metric: str,
    preprocess: PreprocessConfig,
    baseline_fraction: float = 0.10,
    min_baseline: int = 1,
    impostor_per_subject: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around compute_trials using a metric identifier."""

    return compute_trials(
        dataset,
        scorer=make_scorer(metric),
        preprocess=preprocess,
        baseline_fraction=baseline_fraction,
        min_baseline=min_baseline,
        impostor_per_subject=impostor_per_subject,
        random_state=random_state,
    )


def threshold_at_most_k_fp(
    *,
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    k: int = 1,
    accept_greater: bool = False,
) -> ThresholdResult:
    """Find an acceptance threshold such that FP <= k.

    If accept_greater=False (distance-like): accept if score <= threshold.
    If accept_greater=True  (similarity-like): accept if score >= threshold.
    """

    # Sort impostor scores in the direction that makes the acceptance region contiguous.
    # - distance-like (<=): accepted impostors are the smallest values.
    # - similarity-like (>=): accepted impostors are the largest values.
    imp_sorted = np.sort(impostor_distances)

    if len(imp_sorted) == 0:
        raise ValueError("No impostor distances provided")

    n = len(imp_sorted)

    if not accept_greater:
        # accept if score <= t
        if k <= 0:
            threshold = float(np.nextafter(imp_sorted[0], -np.inf))
        elif k >= n:
            threshold = float(imp_sorted[-1])
        else:
            # Want at most k impostors accepted => t < (k+1)th smallest impostor.
            threshold = float(np.nextafter(imp_sorted[k], -np.inf))

        fp = int(np.sum(impostor_distances <= threshold))
        tp = int(np.sum(genuine_distances <= threshold))
        fn = int(np.sum(genuine_distances > threshold))
    else:
        # accept if score >= t
        if k <= 0:
            threshold = float(np.nextafter(imp_sorted[-1], np.inf))
        elif k >= n:
            threshold = float(imp_sorted[0])
        else:
            # Want at most k impostors accepted => accept region is top-k scores.
            # Set t just above the (n-k-1)th element.
            threshold = float(np.nextafter(imp_sorted[n - k - 1], np.inf))

        fp = int(np.sum(impostor_distances >= threshold))
        tp = int(np.sum(genuine_distances >= threshold))
        fn = int(np.sum(genuine_distances < threshold))

    return ThresholdResult(
        threshold=threshold,
        false_positives=fp,
        true_positives=tp,
        false_negatives=fn,
        n_impostor_trials=int(len(impostor_distances)),
        n_genuine_trials=int(len(genuine_distances)),
    )
