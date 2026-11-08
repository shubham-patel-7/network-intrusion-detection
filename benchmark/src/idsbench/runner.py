"""Train a model from the benchmark and use it: a small command-line tool that runs from any IDE.

    python run_model.py                          # demo: train a Decision Tree, test it, classify 10 test flows
    python run_model.py train --model RF         # train on the training split, evaluate on the test split, save to models/RF
    python run_model.py predict --model RF --csv my_flows.csv

Models: DT, RF, SVM, MLP, LSTM. Settings are the ones found by this repository's own tuning run (results/best_*.json)
when that file exists, otherwise the configurations stored below.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ciciot
from . import common as c

MODELS_DIR = c.ROOT / "models"
NAMES = {"DT": "Decision Tree", "RF": "Random Forest", "SVM": "SVM (RBF)", "MLP": "MLP", "LSTM": "LSTM"}
# Configurations chosen by the tuning run of this benchmark (results/best_*.json); used only when those files are missing.
# Epochs for the neural models are the early-stopping result.
REPORT_CONFIGS = {
    "DT": dict(max_depth=15, min_samples_leaf=13, criterion="entropy"),
    "RF": dict(n_estimators=146, max_depth=30, max_features="log2", min_samples_leaf=2),
    "SVM": dict(C=6.81, gamma=0.156),
    "MLP": dict(units=208, dropout=0.0029, lr=0.000676, batch=256, clip=0),
    "LSTM": dict(units=79, dropout=0.0832, lr=0.00376, batch=128, clip=50),
}
REPORT_EPOCHS = {"MLP": 59, "LSTM": 57}
LABEL_COLUMNS = ("label", "Label")


def config_for(name):
    """Own tuning result if available, else the stored configuration."""
    if c.exists(f"best_{name}.json"):
        best = c.load_json(f"best_{name}.json")
        return best["cfg"], best.get("epochs"), "%s/best_%s.json (this repository's tuning run)" % (c.RES.name, name)
    return REPORT_CONFIGS[name], REPORT_EPOCHS.get(name), "the configurations stored in runner.py"


# ---------------------------------------------------------------- saving and loading
def save_model(m, pre, meta, out):
    import joblib
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, out / "preprocessor.joblib")
    if isinstance(m, c.SKModel):
        joblib.dump(m, out / "model.joblib")
    else:
        m.model.save(out / "model.keras")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_model(name, models_dir=MODELS_DIR):
    import joblib
    d = Path(models_dir) / name
    if not (d / "meta.json").exists():
        raise SystemExit(f"no saved {name} model in {d}; train it first:  python run_model.py train --model {name}")
    meta = json.loads((d / "meta.json").read_text())
    pre = joblib.load(d / "preprocessor.joblib")
    if name in ("MLP", "LSTM"):
        import keras
        m = c.DLModel(name, meta["config"])
        m.model = keras.saving.load_model(d / "model.keras")
    else:
        m = joblib.load(d / "model.joblib")
    return m, pre, meta


# ---------------------------------------------------------------- training
def train(name, models_dir=MODELS_DIR, quiet=False):
    """Fit one model on the training split, evaluate once on the test split and save it with its preprocessing."""
    say = (lambda *a: None) if quiet else print
    cfg, epochs, source = config_for(name)
    say(f"{NAMES[name]}: settings from {source}")
    D = c.prepare()
    if name in ("MLP", "LSTM"):
        c.init_backend()
    m = c.make_model(name, cfg, seed=0)
    t0 = time.perf_counter()
    if name in ("MLP", "LSTM"):
        m.fit(D["X"], D["y"], epochs=epochs)
    else:
        m.fit(D["X"], D["y"])
    secs = time.perf_counter() - t0
    p, s = m.predict(D["Xte"]), m.scores(D["Xte"])
    res = c.view_metrics(D, p, s)
    full = res["full"]
    say(f"trained in {secs:.1f} s on {len(D['y']):,} records; test split ({len(D['yte']):,} records, evaluated once):")
    say(f"  accuracy {full['acc']:.3f}   macro-F1 {full['macro_f1']:.3f}   attack detection rate "
        f"{full['det_rate']:.1%}   false-positive rate {full['fpr_normal']:.1%}")
    say("  recall per class: " + "   ".join(f"{k} {r:.2f}" for k, r in zip(c.CLASSES, full["recall"])))
    if res.get("unseen"):
        say(f"  records of attack types never seen in training: {res['unseen']['det_rate']:.1%} flagged as attacks")
    meta = dict(model=name, config=cfg, epochs=epochs, settings_from=source, train_seconds=secs,
                test=dict(acc=full["acc"], macro_f1=full["macro_f1"], det_rate=full["det_rate"],
                          fpr_normal=full["fpr_normal"], recall=dict(zip(c.CLASSES, full["recall"]))),
                seed=0, trained_on=f"{c.DATASET}: {len(D['y']):,} training records")
    save_model(m, D["pre"], meta, Path(models_dir) / name)
    say(f"saved to {Path(models_dir) / name}")
    return m, D


# ---------------------------------------------------------------- using a model
def read_records(path):
    """Read a CSV of CICIoT2023 flows: the 39 feature columns of the dataset and, optionally, a Label column. Missing
    feature columns are reported when the flows are classified (Preprocessor.transform)."""
    return pd.read_csv(path)


def classify(m, pre, df):
    """Returns a DataFrame with the predicted family, its confidence and an alert flag for every record."""
    X = pre.transform(df)
    s = m.scores(X)
    p = m.predict(X)
    if isinstance(m, c.SKModel) and m.name == "SVM":
        conf = np.full(len(p), np.nan)  # the SVM outputs decision scores, not probabilities
    else:
        conf = s.max(1)
    return pd.DataFrame({"prediction": [c.CLASSES[i] for i in p], "confidence": conf,
                         "alert": [i != 0 for i in p]})  # class 0 is the benign class


def cmd_predict(args):
    m, pre, meta = load_model(args.model, args.models_dir)
    try:
        df = read_records(args.csv)
        out = classify(m, pre, df)
    except (ValueError, OSError) as e:
        raise SystemExit(f"could not read {args.csv}: {e}")
    print(f"{NAMES[args.model]} classified {len(out):,} records from {args.csv}")
    counts = out["prediction"].value_counts()
    for k in c.CLASSES:
        if k in counts:
            print(f"  {k:<7}{counts[k]:>8,}  ({counts[k] / len(out):.1%})")
    print(f"  alerts (anything but Benign): {int(out['alert'].sum()):,}")
    label_col = next((x for x in LABEL_COLUMNS if x in df.columns), None)
    if label_col is not None:
        truth = df[label_col].map(ciciot.category)
        known = truth.notna()
        if known.any():
            acc = float((truth[known].values == out["prediction"][known].values).mean())
            print(f"  accuracy against the {label_col!r} column: {acc:.1%} on {int(known.sum()):,} records with a known label")
    result = pd.concat([df.reset_index(drop=True), out], axis=1)
    dest = Path(args.out)
    result.to_csv(dest, index=False)
    print(f"predictions written to {dest}")


def cmd_demo(args):
    print("Demo: train a Decision Tree on the CICIoT2023 sample, test it, classify 10 flows from the test split.")
    print()
    m, D = train(args.model, args.models_dir)
    rng = np.random.RandomState(1)
    te = D["frame"].iloc[D["test_index"]]
    te = te.iloc[rng.choice(len(te), 10, replace=False)]
    out = classify(m, D["pre"], te)
    print()
    print("Ten random test-split flows:")
    print(f"  {'true attack type':<26}{'true category':<14}{'predicted':<12}{'alert'}")
    for (_, row), (_, o) in zip(te.iterrows(), out.iterrows()):
        mark = "" if row["category"] == o["prediction"] else "   <- wrong"
        print(f"  {row['label']:<26}{row['category']:<14}{o['prediction']:<12}{'ALERT' if o['alert'] else '-'}{mark}")
    print()
    print("Next: python run_model.py train --model MLP    (or RF, SVM, LSTM)")
    print("      python run_model.py predict --model DT --csv your_flows.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(MODELS_DIR), help="where trained models are saved (default: models/)")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("demo", help="train a fast model, test it, classify 10 flows (default command)")
    d.add_argument("--model", default="DT", choices=list(NAMES))
    t = sub.add_parser("train", help="train on the training split, evaluate on the test split, save")
    t.add_argument("--model", required=True, choices=list(NAMES))
    p = sub.add_parser("predict", help="classify a CSV of flows with a saved model")
    p.add_argument("--model", required=True, choices=list(NAMES))
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="predictions.csv")
    args = ap.parse_args(argv)
    if args.cmd in (None, "demo"):
        if args.cmd is None:
            args.model = "DT"
        cmd_demo(args)
    elif args.cmd == "train":
        train(args.model, args.models_dir)
    else:
        cmd_predict(args)


if __name__ == "__main__":
    main()
