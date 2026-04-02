import pandas as pd

from ks_eval.data import Dataset
from ks_eval.evaluator import evaluate_sequential_probes
from ks_eval.preprocess import PreprocessConfig
from ks_eval.scoring import (
    CosineSimilarityScorer,
    EuclideanDistanceScorer,
    GaussianLogLikelihoodScorer,
    MahalanobisDistanceScorer,
)


def test_evaluate_sequential_probes_generates_rows():
    df = pd.DataFrame(
        {
            "subject": ["a"] * 10 + ["b"] * 10,
            "sessionIndex": list(range(10)) + list(range(10)),
            "rep": [1] * 20,
            "f1": list(range(20)),
            "f2": list(range(20, 40)),
        }
    )
    dataset = Dataset(df=df, feature_columns=["f1", "f2"])
    out = evaluate_sequential_probes(dataset, baseline_fraction=0.10, min_baseline=1)

    # baseline=1 per user, probes=9 per user => 18
    assert len(out) == 18
    assert set(out.columns) >= {
        "subject",
        "probe_global_index",
        "probe_index_within_subject",
        "sessionIndex",
        "rep",
    }


def test_evaluate_with_preprocessing_enabled_does_not_change_probe_count():
    df = pd.DataFrame(
        {
            "subject": ["a"] * 10,
            "sessionIndex": list(range(10)),
            "rep": [1] * 10,
            "f1": [1.0, None, 3.0, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0],
            "f2": list(range(10)),
        }
    )
    dataset = Dataset(df=df, feature_columns=["f1", "f2"])

    out_default = evaluate_sequential_probes(dataset, baseline_fraction=0.10, min_baseline=1)
    out_pp = evaluate_sequential_probes(
        dataset,
        baseline_fraction=0.10,
        min_baseline=1,
        preprocess=PreprocessConfig(impute="median", scaler="robust", clip=5.0),
    )

    assert len(out_default) == len(out_pp)


def test_pluggable_scorers_add_score_columns():
    df = pd.DataFrame(
        {
            "subject": ["a"] * 20,
            "sessionIndex": list(range(20)),
            "rep": [1] * 20,
            "f1": [float(i) for i in range(20)],
            "f2": [float(i * 2) for i in range(20)],
        }
    )
    dataset = Dataset(df=df, feature_columns=["f1", "f2"])

    scorers = [
        EuclideanDistanceScorer(),
        CosineSimilarityScorer(),
        MahalanobisDistanceScorer(),
        GaussianLogLikelihoodScorer(),
    ]
    out = evaluate_sequential_probes(
        dataset,
        baseline_fraction=0.10,
        min_baseline=2,
        preprocess=PreprocessConfig(impute="none", scaler="standard"),
        scorers=scorers,
    )

    for col in ["euclidean", "cosine", "mahalanobis", "gaussian_ll"]:
        assert col in out.columns
        assert out[col].notna().all()
