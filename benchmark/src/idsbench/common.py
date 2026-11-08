"""Shared code for the ML vs DL intrusion-detection benchmark (CICIoT2023).

Implements the methodology of the report: preprocessing, split, models, metrics, cost timing.
Every fitted step (encoder, scaler, resampler, feature selector) is fitted on training data only.
"""
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

# Keras 3 runs on several backends. The report used TensorFlow; fall back to PyTorch where
# TensorFlow has no wheel (for example on very new Python versions). Must be set before keras is imported.
os.environ.setdefault("KERAS_BACKEND", "tensorflow" if importlib.util.find_spec("tensorflow") else "torch")

# CICIoT2023: class names (index 0 is the benign class) and the indices of the rare classes (Web-based, Brute Force).
_DATASETS = {
    "ciciot2023": dict(classes=["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based", "Brute Force"], rare=[6, 7]),
}
DATASET = "ciciot2023"

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("IDS_RES", ROOT / "results"))
FIGS = Path(os.environ.get("IDS_FIGS", ROOT / "figures"))
THREADS = int(os.environ.get("IDS_THREADS", "2"))
SEEDS = [int(s) for s in os.environ.get("IDS_SEEDS", "0,1,2").split(",")]
SPLIT_SEED = 42
CLASSES = _DATASETS[DATASET]["classes"]
RARE = _DATASETS[DATASET]["rare"]  # kept in full when the SVM is fitted on a subsample, and oversampled by SMOTE


def prepare():
    """Returns a dict of arrays for CICIoT2023 (see ciciot.py). Every fitted step uses training data only."""
    from . import ciciot
    return ciciot.prepare()


def class_weights(y, k=None, clip=0):
    """Equation (2): w_c = N / (K * n_c), optionally clipped at `clip`."""
    k = k or len(np.unique(y))
    n = len(y)
    w = {int(c): n / (k * (y == c).sum()) for c in np.unique(y)}
    if clip:
        w = {c: min(v, clip) for c, v in w.items()}
    return w


def svm_subsample(X, y, cap=30000, seed=0):
    """Class-stratified subsample of at most `cap` records that keeps every record of the rare classes."""
    rng = np.random.RandomState(seed)
    keep = np.where(np.isin(y, RARE))[0]
    others = np.where(~np.isin(y, RARE))[0]
    fill = max(cap - len(keep), 0)
    if fill >= len(others):
        return np.sort(np.concatenate([keep, others]))
    pick = []
    for c in np.unique(y[others]):
        ic = others[y[others] == c]
        pick.append(rng.choice(ic, int(round(fill * len(ic) / len(others))), replace=False))
    return np.sort(np.concatenate([keep] + pick))


def sample_configs(name, n=None, seed=123):
    """Random search: the same budget (default 8) of configurations for every model (Table IV)."""
    n = n or int(os.environ.get("IDS_N", "8"))
    rng = np.random.RandomState(seed + sum(map(ord, name)))

    def lu(lo, hi):  # log-uniform
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    cfgs = []
    for _ in range(n):
        if name == "DT":
            d = int(rng.choice([5, 10, 15, 20, 30, 40, 0]))
            cfgs.append(dict(max_depth=d or None, min_samples_leaf=int(rng.randint(1, 21)),
                             criterion=str(rng.choice(["gini", "entropy"]))))
        elif name == "RF":
            d = int(rng.choice([10, 20, 30, 40, 0]))
            cfgs.append(dict(n_estimators=int(rng.randint(100, 401)), max_depth=d or None,
                             max_features=str(rng.choice(["sqrt", "log2"])), min_samples_leaf=int(rng.randint(1, 6))))
        elif name == "SVM":
            cfgs.append(dict(C=lu(0.1, 100), gamma=lu(0.001, 1)))
        else:  # MLP, LSTM
            lo, hi = (64, 257) if name == "MLP" else (32, 129)
            cfgs.append(dict(units=int(rng.randint(lo, hi)), dropout=float(rng.uniform(0, 0.5)), lr=lu(1e-4, 1e-2),
                             batch=int(rng.choice([128, 256, 512])), clip=int(rng.choice([0, 50]))))
    return cfgs


# ---------------------------------------------------------------- models
class SKModel:
    """Decision Tree, Random Forest or RBF SVM (scikit-learn)."""

    def __init__(self, name, params, seed=0, weighted=True):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        self.name, self.seed = name, seed
        cw = "balanced" if weighted else None
        if name == "DT":
            self.m = DecisionTreeClassifier(class_weight=cw, random_state=seed, **params)
        elif name == "RF":
            self.m = RandomForestClassifier(class_weight=cw, n_jobs=THREADS, random_state=seed, **params)
        else:
            self.m = SVC(kernel="rbf", class_weight=cw, decision_function_shape="ovr",
                         cache_size=1000, max_iter=200000, random_state=seed, **params)
        self.svm_cap = 30000

    def fit(self, X, y, **kw):
        if self.name == "SVM":
            i = svm_subsample(X, y, self.svm_cap, self.seed)
            X, y = X[i], y[i]
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def scores(self, X):
        return self.m.decision_function(X) if self.name == "SVM" else self.m.predict_proba(X)

    def predict_one(self, x):
        return self.m.predict(x)

    def single_mode(self):
        if self.name == "RF":
            self.m.set_params(n_jobs=1)

    def batch_mode(self):
        if self.name == "RF":
            self.m.set_params(n_jobs=THREADS)

    def size_info(self):
        import joblib
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "m.joblib")
            joblib.dump(self.m, f)
            mb = os.path.getsize(f) / 1e6
        if self.name == "DT":
            cnt = {"nodes": int(self.m.tree_.node_count)}
        elif self.name == "RF":
            cnt = {"trees": len(self.m.estimators_), "nodes": int(sum(t.tree_.node_count for t in self.m.estimators_))}
        else:
            cnt = {"support_vectors": int(self.m.n_support_.sum())}
        return mb, cnt


class DLModel:
    """MLP or LSTM (Keras 3). The LSTM sees every record as a sequence of one time step."""

    def __init__(self, kind, params, seed=0, n_classes=None, weighted=True):
        self.kind, self.p, self.seed, self.weighted = kind, dict(params), seed, weighted
        self.k = n_classes or len(CLASSES)
        self.model = None
        self.epochs = None

    def _build(self, n_in):
        import keras
        L = keras.layers
        keras.utils.set_random_seed(self.seed)
        u, d = self.p["units"], self.p["dropout"]
        if self.kind == "MLP":
            inp = keras.Input(shape=(n_in,))
            x = L.Dense(u, activation="relu")(inp)
        else:
            inp = keras.Input(shape=(1, n_in))
            x = L.LSTM(u)(inp)
        x = L.Dropout(d)(x)
        x = L.Dense(max(u // 2, 16), activation="relu")(x)
        x = L.Dropout(d)(x)
        out = L.Dense(self.k, activation="softmax")(x)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(self.p["lr"]), loss="sparse_categorical_crossentropy")
        return m

    def _x(self, X):
        return X if self.kind == "MLP" else X.reshape(len(X), 1, -1)

    def fit(self, X, y, Xval=None, yval=None, epochs=None, max_epochs=60):
        """With `epochs=None`: early stopping on the validation loss (patience 10, capped at `max_epochs`).
        With a fixed `epochs`: refit for exactly that many epochs (the number found on validation)."""
        import keras
        self.model = self._build(X.shape[1])
        cw = class_weights(y, self.k, self.p.get("clip", 0)) if self.weighted else None
        if epochs is None:
            cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
            h = self.model.fit(self._x(X), y, batch_size=self.p["batch"], epochs=max_epochs, verbose=0,
                               validation_data=(self._x(Xval), yval), callbacks=cb, class_weight=cw)
            self.epochs = int(np.argmin(h.history["val_loss"]) + 1)
        else:
            self.model.fit(self._x(X), y, batch_size=self.p["batch"], epochs=epochs, verbose=0, class_weight=cw)
            self.epochs = epochs
        return self

    def scores(self, X):
        return self.model.predict(self._x(X), batch_size=4096, verbose=0)

    def predict(self, X):
        return self.scores(X).argmax(1)

    def predict_one(self, x):
        import keras
        return keras.ops.convert_to_numpy(self.model(self._x(x), training=False)).argmax(1)

    def single_mode(self):
        pass

    def batch_mode(self):
        pass

    def size_info(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "m.keras")
            self.model.save(f)
            mb = os.path.getsize(f) / 1e6
        return mb, {"parameters": int(self.model.count_params())}


def make_model(name, params, seed=0, k=None, weighted=True):
    if name in ("DT", "RF", "SVM"):
        return SKModel(name, params, seed, weighted)
    return DLModel(name, params, seed, k, weighted)


def init_backend():
    """Pin the CPU thread count so that timings are comparable (2 cores in the report)."""
    import keras
    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    elif backend == "torch":
        import torch
        torch.set_num_threads(THREADS)
    return backend


# ---------------------------------------------------------------- metrics
def macro_f1(y, p, k=None):
    k = k or len(CLASSES)
    return float(precision_recall_fscore_support(y, p, labels=list(range(k)), average="macro", zero_division=0)[2])


def full_metrics(y, p, score=None, k=None):
    """Equations (7) and (8): per-class precision, recall, F1, FPR, macro-F1, plus detection rate and ROC-AUC."""
    k = k or len(CLASSES)
    cm = confusion_matrix(y, p, labels=list(range(k)))
    P, R, F, S = precision_recall_fscore_support(y, p, labels=list(range(k)), zero_division=0)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    out = dict(acc=float((p == y).mean()), macro_f1=float(F.mean()), precision=P.tolist(), recall=R.tolist(),
               f1=F.tolist(), fpr=(fp / np.maximum(fp + tn, 1)).tolist(), support=S.tolist(), cm=cm.tolist())
    at, ap = y != 0, p != 0
    out["det_rate"] = float((ap & at).sum() / max(at.sum(), 1))           # attack flagged as any attack
    out["fpr_normal"] = float((ap & ~at).sum() / max((~at).sum(), 1))     # normal raised as an alert
    if score is not None:
        aucs = [float(roc_auc_score(y == c, score[:, c])) for c in range(k) if 0 < (y == c).sum() < len(y)]
        out["auc_macro"] = float(np.mean(aucs))
        out["auc_class"] = aucs
        out["auc_binary"] = float(roc_auc_score(at, -score[:, 0]))
    return out


def view_metrics(D, p, score):
    """Metrics on the full test set and, when attack types were held out of training (IDS_CICIOT_HOLDOUT), on the test
    records of the seen types and on the records of the unseen attack types."""
    y = D["yte"]
    res = {"full": full_metrics(y, p, score)}
    u = D.get("unseen")
    if u is not None and u.any():
        if (~u).any():  # the test records whose attack types were all seen in training
            res["seen"] = full_metrics(y[~u], p[~u], score[~u])
        yu, pu = y[u], p[u]
        res["unseen"] = dict(n=int(u.sum()), acc=float((yu == pu).mean()), det_rate=float((pu != 0).mean()),
                             recall={CLASSES[c]: float((pu[yu == c] == c).mean()) for c in range(1, len(CLASSES)) if (yu == c).any()},
                             n_class={CLASSES[c]: int((yu == c).sum()) for c in range(1, len(CLASSES))})
    return res


def time_inference(m, Xte, n_single=1000):
    """Batch time on the whole test file and per-record latency, after one untimed warm-up pass."""
    m.batch_mode()
    m.predict(Xte[:512])
    t0 = time.perf_counter()
    m.predict(Xte)
    batch = time.perf_counter() - t0
    m.single_mode()
    m.predict_one(Xte[:1])
    lat = []
    for i in range(n_single):
        t0 = time.perf_counter()
        m.predict_one(Xte[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1e3)
    m.batch_mode()
    return dict(batch_s=batch, lat_median_ms=float(np.median(lat)), lat_p95_ms=float(np.percentile(lat, 95)))


def save_json(obj, name):
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / name, "w") as f:
        json.dump(obj, f)


def load_json(name):
    with open(RES / name) as f:
        return json.load(f)


def exists(name):
    return (RES / name).exists()
