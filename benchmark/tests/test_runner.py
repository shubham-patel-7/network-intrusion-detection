"""Tests for the runnable model tool: CSV reading, classification, saving and loading (CICIoT2023 flows)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from idsbench import ciciot
from idsbench import common as c
from idsbench import runner

COLS = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "syn_flag_number", "ack_count", "Tot size", "IAT"]


def _flows(n=40, label=True):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(n, len(COLS)) * 100, columns=COLS)
    if label:
        df["Label"] = (["BENIGN", "DDOS-SYN_FLOOD", "MIRAI-GREETH_FLOOD", "RECON-PORTSCAN"] * n)[:n]
    return df


def _pre(df):
    scaler = MinMaxScaler(clip=True).fit(df[COLS].to_numpy())
    return ciciot.Preprocessor(COLS, scaler)


def test_read_records_keeps_columns_and_label(tmp_path):
    p = tmp_path / "a.csv"
    _flows().to_csv(p, index=False)
    df = runner.read_records(p)
    assert len(df) == 40 and "Label" in df.columns and set(COLS) <= set(df.columns)


def test_preprocessor_rejects_files_with_missing_columns():
    df = _flows()
    pre = _pre(df)
    assert pre.transform(df).shape == (40, len(COLS))
    with pytest.raises(ValueError, match="Rate"):
        pre.transform(df.drop(columns=["Rate"]))


def test_saved_model_round_trip_and_classification(tmp_path):
    df = _flows(200)
    pre = _pre(df)
    X = pre.transform(df)
    y = np.array([ciciot.CLASSES.index(ciciot.category(l)) for l in df["Label"]])
    cfg = dict(max_depth=8, min_samples_leaf=2, criterion="entropy")
    m = c.make_model("DT", cfg, seed=0, k=len(c.CLASSES)).fit(X, y)
    runner.save_model(m, pre, dict(model="DT", config=cfg), tmp_path / "DT")
    m2, pre2, meta = runner.load_model("DT", tmp_path)
    assert meta["config"] == cfg
    assert np.array_equal(m.predict(X), m2.predict(X))
    out = runner.classify(m2, pre2, df)
    assert set(out["prediction"]) <= set(c.CLASSES) and (out["alert"] == (out["prediction"] != "Benign")).all()


def test_neural_model_save_and_load(tmp_path):
    pytest.importorskip("keras")
    c.init_backend()
    rng = np.random.RandomState(3)
    X = rng.rand(200, 8).astype(np.float32)
    y = (X[:, 0] > .5).astype(int)
    cfg = dict(units=8, dropout=0.0, lr=0.01, batch=64, clip=0)
    for kind in ("MLP", "LSTM"):
        m = c.make_model(kind, cfg, seed=0, k=5).fit(X, y, epochs=2)
        runner.save_model(m, None, dict(model=kind, config=cfg), tmp_path / kind)
        m2, _, _ = runner.load_model(kind, tmp_path)
        assert np.allclose(m.scores(X), m2.scores(X), atol=1e-5)
