"""The five models of the study, loaded for prediction on CICIoT2023 flows.

The files in models/study/ are plain scikit-learn objects (Decision Tree, Random Forest, SVM) and Keras files (MLP, LSTM)
plus the fitted preprocessing, exported by the benchmark project. They were trained on the full training split with the
settings chosen by the benchmark's random search (see models/study/meta.json for each model's own test result).
"""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before keras is imported

DIR = Path(__file__).resolve().parents[1] / "models" / "study"
KEYS = ["DT", "RF", "SVM", "MLP", "LSTM"]
NAME = {"DT": "Decision Tree", "RF": "Random Forest", "SVM": "SVM (RBF)", "MLP": "MLP", "LSTM": "LSTM"}
KEY_OF = {v: k for k, v in NAME.items()}
LABEL_COLUMNS = ("label", "class", "attack")


def available():
    return all((DIR / f).exists() for f in ["preprocessor.joblib", "meta.json"]) and all(
        (DIR / f"{k}.joblib").exists() or (DIR / f"{k}.keras").exists() for k in KEYS)


def meta():
    return json.loads((DIR / "meta.json").read_text())


def classes():
    return meta()["classes"]  # the benign class is first


def load_preprocessor():
    return joblib.load(DIR / "preprocessor.joblib")


def load_model(key):
    if key in ("MLP", "LSTM"):
        import keras
        return keras.saving.load_model(DIR / f"{key}.keras")
    return joblib.load(DIR / f"{key}.joblib")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def category(raw):
    """Category of a raw CICIoT2023 label (any capitalisation), or None if it is not recognised."""
    n = _norm(raw)
    if n in ("benigntraffic", "benign"):
        return "Benign"
    if n.startswith("ddos"):
        return "DDoS"
    if n.startswith("dos"):
        return "DoS"
    if n.startswith("mirai"):
        return "Mirai"
    if n.startswith("recon") or n == "vulnerabilityscan":
        return "Recon"
    if n in ("dnsspoofing", "arpspoofing") or n.startswith("mitm"):
        return "Spoofing"
    if n in ("browserhijacking", "backdoormalware", "xss", "uploadingattack", "sqlinjection", "commandinjection"):
        return "Web-based"
    if n == "dictionarybruteforce":
        return "Brute Force"
    return None


def read_flows(source):
    """Read a CSV of CICIoT2023 flows: the feature columns the models were trained on, plus an optional label column.
    Returns the DataFrame and the name of the label column (or None)."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    need = load_preprocessor()["cols"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"the file lacks these CICIoT2023 columns: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    label = next((c for c in df.columns if c.lower() in LABEL_COLUMNS), None)
    return df, label


def transform(pre, df):
    """Raw flows -> the scaled inputs the models were trained on (same steps as the benchmark's preprocessor)."""
    x = df[pre["cols"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return pre["scaler"].transform(x).astype(np.float32)


def predict(key, model, X):
    """Returns (predicted class index, confidence). The SVM outputs decision scores, so its confidence is NaN."""
    if key in ("MLP", "LSTM"):
        import keras
        x = X if key == "MLP" else X.reshape(len(X), 1, -1)
        p = keras.ops.convert_to_numpy(model.predict(x, batch_size=4096, verbose=0))
        return p.argmax(1), p.max(1)
    if key == "SVM":
        return model.predict(X), np.full(len(X), np.nan)
    p = model.predict_proba(X)
    return p.argmax(1), p.max(1)


def classify(key, model, pre, df):
    """DataFrame with the predicted category, a confidence and an alert flag (anything but benign) for every flow."""
    cls = classes()
    idx, conf = predict(key, model, transform(pre, df))
    pred = [cls[i] for i in idx]
    return pd.DataFrame({"prediction": pred, "confidence": conf, "alert": [i != 0 for i in idx]})
