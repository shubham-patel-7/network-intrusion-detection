"""Tests of the Streamlit app's data and detector code (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import study_data as S  # noqa: E402
import study_models as M  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_flows.csv"


def test_summary_is_complete_and_consistent():
    assert S.DATASET == "ciciot2023" and len(S.CLASSES) == 8 and S.CLASSES[0] == "Benign"
    for m in S.MODELS:
        assert 0 < S.MAIN[m]["mf1"][0] < 1
        assert len(S.PER_CLASS[m]) == 8
        lo, hi = S.CI[m]
        assert lo < S.MAIN[m]["mf1"][0] + 0.02 and hi > lo
    assert sum(t for _, t in S.DISTRIBUTION.values()) == S.META["n_test"]
    assert sum(t for t, _ in S.DISTRIBUTION.values()) == S.META["n_train"]


def test_unseen_experiment_is_loaded():
    assert len(S.HOLDOUT_TYPES) == 7 and S.HOLDOUT_META["n_unseen"] > 0
    for m in S.MODELS:
        assert 0 <= S.UNSEEN[m]["acc"] <= S.UNSEEN[m]["det"] <= 100


def test_category_mapping():
    assert M.category("BenignTraffic") == "Benign" and M.category("DDoS-SYN_Flood") == "DDoS"
    assert M.category("DOS-SYN_FLOOD") == "DoS" and M.category("Mirai-greeth_flood") == "Mirai"
    assert M.category("DictionaryBruteForce") == "Brute Force" and M.category("nonsense") is None


def test_read_flows_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="lacks"):
        M.read_flows(p)


@pytest.mark.skipif(not M.available(), reason="exported models are not present")
@pytest.mark.parametrize("key", M.KEYS)
def test_every_model_classifies_the_bundled_sample(key):
    df, label = M.read_flows(SAMPLE)
    pre = M.load_preprocessor()
    out = M.classify(key, M.load_model(key), pre, df)
    assert len(out) == len(df) and set(out["prediction"]) <= set(M.classes())
    assert (out["alert"] == (out["prediction"] != "Benign")).all()
    truth = df[label].map(M.category)
    # the sample is drawn from the held-out test split; every model is far above the 1-in-8 chance level
    assert (truth.values == out["prediction"].values).mean() > 0.5
