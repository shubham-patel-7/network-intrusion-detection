"""Ablations (Table VII): frozen hyperparameters, single seed (0).

Usage: python -m idsbench.ablate task|imbalance|features|size
Restrict the models with IDS_MODELS=DT,RF (default: all five).
"""
import os
import sys

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from . import common as c

ALL = os.environ.get("IDS_MODELS", "DT,RF,SVM,MLP,LSTM").split(",")
DL = ("MLP", "LSTM")


def fit_eval(D, name, X, y, Xte, k=None, weighted=True, epochs=None, val=None):
    best = c.load_json(f"best_{name}.json")
    cfg = best["cfg"]
    m = c.make_model(name, cfg, seed=0, k=k, weighted=weighted)
    if name in DL:
        if val is not None:
            m.fit(X, y, val[0], val[1])
        else:
            m.fit(X, y, epochs=epochs or best["epochs"])
    else:
        m.fit(X, y)
    p = m.predict(Xte)
    s = m.scores(Xte)
    return m, p, s


def binary_metrics(D, p, s):
    y = D["yte"]
    yb = (y != 0).astype(int)
    out = {}
    views = [("full", np.ones(len(y), bool))] + ([("hard21", D["hard21"])] if D.get("hard21") is not None else [])
    for tag, mask in views:
        yy, pp = yb[mask], p[mask]
        tp = ((pp == 1) & (yy == 1)).sum()
        fp = ((pp == 1) & (yy == 0)).sum()
        fn = ((pp == 0) & (yy == 1)).sum()
        tn = ((pp == 0) & (yy == 0)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[tag] = dict(acc=float((pp == yy).mean()), f1_attack=float(2 * prec * rec / max(prec + rec, 1e-9)),
                        det_rate=float(rec), fpr=float(fp / max(fp + tn, 1)),
                        auc=float(c.roc_auc_score(yy, s[mask][:, 1] if s.ndim == 2 else s[mask])))
    u = D.get("unseen")
    if u is not None and u.any():
        out["unseen_det_rate"] = float((p[u] == 1).mean())
    return out


def main(which):
    D = c.prepare()
    X, y, Xte = D["X"], D["y"], D["Xte"]
    c.init_backend()
    fn = f"ablate_{which}.json"
    out = c.load_json(fn) if c.exists(fn) else {}

    def done(key):
        return key in out

    def save():
        c.save_json(out, fn)

    if which == "task":  # five-class vs binary (Normal versus Attack)
        yb = (y != 0).astype(int)
        for name in ALL:
            if done(name):
                continue
            _, p, s = fit_eval(D, name, X, yb, Xte, k=2)
            out[name] = binary_metrics(D, p, s)
            save()
            print("task", name, out[name]["full"], flush=True)
    elif which == "imbalance":  # no correction vs SMOTE (class weights are the main run)
        from imblearn.over_sampling import SMOTE
        # Raise each rare class to a floor (the median class size), or leave it alone if it is already larger. The rare classes
        # of CICIoT2023 already have 10,000+ training records, so a smaller floor would change nothing.
        counts = np.bincount(y, minlength=len(c.CLASSES))
        floor = int(np.median(counts))
        target = {r: max(floor, int(counts[r])) for r in c.RARE}
        Xs, ys = SMOTE(sampling_strategy=target, k_neighbors=5, random_state=0).fit_resample(X, y)
        for name in ALL:
            for strat in ("none", "smote"):
                key = f"{name}|{strat}"
                if done(key):
                    continue
                if strat == "none":
                    _, p, s = fit_eval(D, name, X, y, Xte, weighted=False)
                else:
                    _, p, s = fit_eval(D, name, Xs.astype(np.float32), ys, Xte, weighted=False)
                out[key] = c.view_metrics(D, p, s)
                save()
                print("imb", key, round(out[key]["full"]["macro_f1"], 4), flush=True)
    elif which == "features":  # drop each group; top-k by mutual information (best classical + best neural)
        val_f1 = {n: c.load_json(f"best_{n}.json")["val_macro_f1"] for n in ["DT", "RF", "SVM", "MLP", "LSTM"]}
        cl = max([m for m in ("DT", "RF", "SVM") if m in ALL], key=val_f1.get)
        nn = max([m for m in DL if m in ALL], key=val_f1.get)  # only among the models being ablated
        out["models"] = [cl, nn]
        tr = D["idx_tr"]
        rng = np.random.RandomState(0)
        sub = rng.choice(tr, min(30000, len(tr)), replace=False)  # a 30,000-record sample for the mutual information
        mi = mutual_info_classif(X[sub], y[sub], random_state=0)  # training portion only
        order = np.argsort(-mi)
        variants = {f"drop_{g}": np.where(D["groups"] != g)[0] for g in D["group_names"]}
        variants["top20_mi"] = np.sort(order[:20])
        variants["top40_mi"] = np.sort(order[:40])
        out["top20_names"] = [D["names"][i] for i in np.sort(order[:20])]
        for name in (cl, nn):
            for v, cols in variants.items():
                key = f"{name}|{v}"
                if done(key):
                    continue
                _, p, s = fit_eval(D, name, X[:, cols], y, Xte[:, cols])
                out[key] = dict(n_features=int(len(cols)), **c.view_metrics(D, p, s))
                save()
                print("feat", key, len(cols), round(out[key]["full"]["macro_f1"], 4), flush=True)
    elif which == "size":  # 10%, 25%, 50% of the training sample (100% is the main run)
        for frac in (0.10, 0.25, 0.50):
            idx, _ = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=0)
            for name in ALL:
                key = f"{name}|{frac}"
                if done(key):
                    continue
                if name in DL:  # hold out 15% of the subset for early stopping
                    a, b = train_test_split(idx, test_size=0.15, stratify=y[idx], random_state=0)
                    _, p, s = fit_eval(D, name, X[a], y[a], Xte, val=(X[b], y[b]))
                else:
                    _, p, s = fit_eval(D, name, X[idx], y[idx], Xte)
                out[key] = dict(n_train=int(len(idx)), **c.view_metrics(D, p, s))
                save()
                print("size", key, round(out[key]["full"]["macro_f1"], 4), flush=True)
    else:
        raise SystemExit(f"unknown ablation {which!r}; choose task, imbalance, features or size")


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
