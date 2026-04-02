import pandas as pd
import numpy as np

from ks_eval.analysis_threshold import compute_trials_by_name
from ks_eval.data import Dataset
from ks_eval.preprocess import PreprocessConfig


def _toy_dataset() -> Dataset:
    df = pd.DataFrame(
        {
            "subject": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "sessionIndex": list(range(10)) + list(range(10)) + list(range(10)),
            "rep": [1] * 30,
            "f1": [float(i) for i in range(30)],
            "f2": [float(i * 2) for i in range(30)],
        }
    )
    return Dataset(df=df, feature_columns=["f1", "f2"])


def test_trials_shapes_and_finiteness_all_metrics():
    dataset = _toy_dataset()
    pp = PreprocessConfig(impute="none", scaler="standard")

    metrics = ["euclidean", "cosine", "mahalanobis", "gaussian_ll"]
    for metric in metrics:
        genuine, impostor = compute_trials_by_name(
            dataset,
            metric=metric,
            preprocess=pp,
            baseline_fraction=0.2,
            min_baseline=2,
            impostor_per_subject=5,
            random_state=0,
        )
        assert len(genuine) > 0
        assert len(impostor) > 0
        assert np.isfinite(genuine).all()
        assert np.isfinite(impostor).all()
