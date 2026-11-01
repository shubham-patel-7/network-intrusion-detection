"""The five models of the study, loaded for prediction.

The files in models/study/ are plain scikit-learn objects (Decision Tree, Random Forest, SVM) and Keras files (MLP, LSTM)
plus the fitted preprocessing, exported by the benchmark project. They were trained on the full KDDTrain+ with the
settings frozen in the report (see models/study/meta.json for each model's own KDDTest+ result).
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before keras is imported

DIR = Path(__file__).resolve().parents[1] / "models" / "study"
KEYS = ["DT", "RF", "SVM", "MLP", "LSTM"]
NAME = {"DT": "Decision Tree", "RF": "Random Forest", "SVM": "SVM (RBF)", "MLP": "MLP", "LSTM": "LSTM"}
KEY_OF = {v: k for k, v in NAME.items()}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CATEGORICAL = ["protocol_type", "service", "flag"]
LOG_COLS = ("duration", "src_bytes", "dst_bytes")


def available():
    return all((DIR / f).exists() for f in ["preprocessor.joblib", "meta.json"]) and all(
        (DIR / f"{k}.joblib").exists() or (DIR / f"{k}.keras").exists() for k in KEYS)


def meta():
    return json.loads((DIR / "meta.json").read_text())


def load_preprocessor():
    return joblib.load(DIR / "preprocessor.joblib")


def load_model(key):
    if key in ("MLP", "LSTM"):
        import keras
        return keras.saving.load_model(DIR / f"{key}.keras")
    return joblib.load(DIR / f"{key}.joblib")


def transform(pre, df):
    """Raw NSL-KDD records (41 attribute columns) -> the 121 scaled inputs the models were trained on."""
    num = df[pre["num"]].astype(float).copy()
    for c in LOG_COLS:
        num[c] = np.log1p(num[c])
    raw = np.hstack([num.values, pre["enc"].transform(df[CATEGORICAL])])
    return pre["scaler"].transform(raw).astype(np.float32)


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
    """DataFrame with the predicted family, a confidence and an alert flag (anything but Normal) for every record."""
    idx, conf = predict(key, model, transform(pre, df))
    pred = [CLASSES[i] for i in idx]
    return pd.DataFrame({"prediction": pred, "confidence": conf, "alert": [p != "Normal" for p in pred]})
