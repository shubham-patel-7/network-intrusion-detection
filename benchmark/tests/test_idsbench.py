"""Tests: pure-logic checks run offline; checks marked `data` need the CICIoT2023 files (see README for the download)."""
import numpy as np
import pytest

from idsbench import common as c


# ------------------------------------------------------------------ offline unit tests
def test_class_weights_follow_equation_2():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = c.class_weights(y)
    assert w[0] == pytest.approx(100 / (3 * 90))
    assert w[2] == pytest.approx(100 / (3 * 1))
    assert c.class_weights(y, clip=10)[2] == 10  # clipped variant used by the neural search space


def test_svm_subsample_keeps_every_rare_class_record_and_respects_cap():
    rng = np.random.RandomState(0)
    y = rng.choice(8, size=5000, p=[.2, .2, .2, .2, .1, .06, .03, .01])
    X = rng.rand(len(y), 4)
    idx = c.svm_subsample(X, y, cap=1000, seed=0)
    assert np.isin(np.where(np.isin(y, c.RARE))[0], idx).all()
    assert len(idx) <= 1010 and len(set(idx)) == len(idx)


@pytest.mark.parametrize("name", ["DT", "RF", "SVM", "MLP", "LSTM"])
def test_random_search_is_reproducible_and_inside_the_declared_space(name):
    a, b = c.sample_configs(name, n=20), c.sample_configs(name, n=20)
    assert a == b and len(a) == 20
    for cfg in a:
        if name == "DT":
            assert cfg["max_depth"] in (None, 5, 10, 15, 20, 30, 40) and 1 <= cfg["min_samples_leaf"] <= 20
        elif name == "RF":
            assert 100 <= cfg["n_estimators"] <= 400 and cfg["max_features"] in ("sqrt", "log2")
        elif name == "SVM":
            assert 0.1 <= cfg["C"] <= 100 and 0.001 <= cfg["gamma"] <= 1
        else:
            lo, hi = (64, 256) if name == "MLP" else (32, 128)
            assert lo <= cfg["units"] <= hi and 0 <= cfg["dropout"] <= 0.5
            assert 1e-4 <= cfg["lr"] <= 1e-2 and cfg["batch"] in (128, 256, 512) and cfg["clip"] in (0, 50)


def test_metrics_on_a_known_confusion_matrix():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4, 4])
    p = np.array([0, 0, 0, 1, 1, 0, 2, 0, 4, 4])
    m = c.full_metrics(y, p, k=5)
    assert m["acc"] == pytest.approx(0.7)
    assert m["recall"] == pytest.approx([0.75, 0.5, 1.0, 0.0, 1.0])
    # attacks are records 4-9 (6 of them); the flagged ones are 4, 6, 8, 9 -> detection rate 4/6
    assert m["det_rate"] == pytest.approx(4 / 6)
    assert m["fpr_normal"] == pytest.approx(1 / 4)  # one of four Normal records raised as an alert
    assert c.macro_f1(y, p, k=5) == pytest.approx(m["macro_f1"])


def test_decision_tree_and_svm_models_train_and_report_size():
    rng = np.random.RandomState(1)
    X = rng.rand(300, 6).astype(np.float32)
    y = (X[:, 0] > .5).astype(int) + (X[:, 1] > .8).astype(int) * 2
    for name, cfg in (("DT", dict(max_depth=5, min_samples_leaf=2, criterion="gini")), ("SVM", dict(C=1.0, gamma=1.0))):
        m = c.make_model(name, cfg, seed=0).fit(X, y)
        assert (m.predict(X) == y).mean() > .8
        assert m.scores(X).shape[0] == len(X)
        mb, cnt = m.size_info()
        assert mb > 0 and cnt


def test_neural_models_learn_and_early_stop():
    pytest.importorskip("keras")
    c.init_backend()
    rng = np.random.RandomState(2)
    X = rng.rand(600, 8).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    for kind in ("MLP", "LSTM"):
        cfg = dict(units=16, dropout=0.0, lr=0.01, batch=64, clip=0)
        m = c.make_model(kind, cfg, seed=0, k=2).fit(X[:450], y[:450], X[450:], y[450:], max_epochs=25)
        assert 1 <= m.epochs <= 25
        assert (m.predict(X[450:]) == y[450:]).mean() > .8
        assert m.predict_one(X[:1]).shape == (1,)
        mb, cnt = m.size_info()
        assert mb > 0 and cnt["parameters"] > 0


# ------------------------------------------------------------------ dataset checks (need the CICIoT2023 files)
@pytest.fixture(scope="module")
def D():
    try:
        return c.prepare()
    except Exception as e:  # the dataset is not on this machine
        pytest.skip(f"CICIoT2023 files unavailable: {e}")


@pytest.mark.data
def test_classes_and_feature_matrix(D):
    assert len(c.CLASSES) == 8 and c.CLASSES[0] == "Benign"
    assert D["X"].shape[1] == D["Xte"].shape[1] == len(D["names"]) == 39
    assert set(np.unique(D["y"])) == set(range(8)) and set(np.unique(D["yte"])) == set(range(8))


@pytest.mark.data
def test_no_leakage_between_the_splits(D):
    tr = D["X"][D["idx_tr"]]
    assert tr.min() >= 0 and tr.max() <= 1                                # scaler fitted on TRAIN only
    assert D["Xte"].min() >= 0 and D["Xte"].max() <= 1                    # test values are clipped, never rescaled
    assert not set(D["idx_tr"]) & set(D["idx_val"])
    assert abs(len(D["idx_val"]) / len(D["y"]) - 0.2) < 0.001
    # stratified: validation keeps the class proportions of the training part
    share = lambda idx: np.bincount(D["y"][idx], minlength=8) / len(idx)
    assert np.allclose(share(D["idx_tr"]), share(D["idx_val"]), atol=0.005)
