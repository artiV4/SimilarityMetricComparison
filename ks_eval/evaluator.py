from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pandas as pd

from ks_eval.data import Dataset, iter_subject_samples
from ks_eval.preprocess import PreprocessConfig, preprocess_fit_transform
from ks_eval.protocol import split_baseline_and_probes
from ks_eval.scoring import Scorer


ScoreFn = Callable[[pd.DataFrame, pd.Series, list[str]], float]


@dataclass(frozen=True)
class ProbeResult:
    subject: str
    probe_global_index: int
    probe_index_within_subject: int
    sessionIndex: int | float | None
    rep: int | float | None
    # Placeholder: scores keyed by metric name.
    scores: dict[str, float]


def evaluate_sequential_probes(
    dataset: Dataset,
    *,
    baseline_fraction: float = 0.10,
    min_baseline: int = 1,
    preprocess: PreprocessConfig | None = None,
    scorers: Optional[Iterable[Scorer]] = None,
    score_fns: Optional[dict[str, ScoreFn]] = None,
) -> pd.DataFrame:
    """Create baseline templates and sequential probe rows; optionally compute scores.

    This returns a *tidy* dataframe: one row per probe.

    `score_fns` is a mapping `{name: fn}` where `fn(baseline_df, probe_row, feature_cols)`
    returns a scalar. For now you can pass None or an empty dict to only generate
    the probe table.
    """

    if score_fns is None:
        score_fns = {}
    if preprocess is None:
        preprocess = PreprocessConfig()
    if scorers is None:
        scorers = []

    results: list[dict] = []
    global_i = 0

    for subject, sdf in iter_subject_samples(dataset):
        baseline_df, probes_df = split_baseline_and_probes(
            sdf, baseline_fraction=baseline_fraction, min_baseline=min_baseline
        )

        # Fit preprocessing on baseline only; apply to baseline + probes.
        baseline_df, probes_df, _artifacts = preprocess_fit_transform(
            baseline_df, probes_df, dataset.feature_columns, preprocess
        )

        fitted_scorers: list[Scorer] = []
        for s in scorers:
            # Scorers are stateful per-subject. Create a shallow copy by re-instantiating
            # via type(s) when possible; fall back to using the instance as-is.
            # IMPORTANT: preserve any configured init params (e.g., Mahalanobis rank).
            try:
                if hasattr(s, "__dict__"):
                    s_local = type(s)(
                        **{k: v for k, v in s.__dict__.items() if not k.endswith("_")}
                    )
                else:
                    s_local = type(s)()  # type: ignore[call-arg]
            except Exception:
                s_local = s
            fitted_scorers.append(s_local.fit(baseline_df, dataset.feature_columns))

        # If there are no probes, skip (nothing to evaluate).
        if len(probes_df) == 0:
            continue

        for j, (_, probe_row) in enumerate(probes_df.iterrows()):
            row = {
                "subject": subject,
                "probe_global_index": global_i,
                "probe_index_within_subject": j,
                "sessionIndex": probe_row.get("sessionIndex"),
                "rep": probe_row.get("rep"),
            }

            # Scores are placeholders unless scoring fns are provided.
            for name, fn in score_fns.items():
                row[name] = float(fn(baseline_df, probe_row, dataset.feature_columns))

            for s in fitted_scorers:
                row[s.name] = float(s.score(probe_row, dataset.feature_columns))

            results.append(row)
            global_i += 1

    return pd.DataFrame(results)
