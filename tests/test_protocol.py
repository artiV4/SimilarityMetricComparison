import pandas as pd

from ks_eval.protocol import split_baseline_and_probes


def test_split_baseline_and_probes_10pct_floor_min1():
    df = pd.DataFrame({"subject": ["u"] * 10, "sessionIndex": list(range(10)), "rep": [1] * 10})
    baseline, probes = split_baseline_and_probes(df, baseline_fraction=0.10, min_baseline=1)
    assert len(baseline) == 1
    assert len(probes) == 9


def test_split_baseline_and_probes_min_baseline():
    df = pd.DataFrame({"subject": ["u"] * 5, "sessionIndex": list(range(5)), "rep": [1] * 5})
    baseline, probes = split_baseline_and_probes(df, baseline_fraction=0.10, min_baseline=2)
    assert len(baseline) == 2
    assert len(probes) == 3
