"""Export the five trained models as plain scikit-learn / Keras files that need no code from this repository.

    python scripts/export_for_app.py [output_dir]

Run it after the pipeline has finished tuning and after `python run_model.py train --model <NAME>` has been run for the
five models (DT, RF, SVM, MLP, LSTM) after tuning. It reads models/<NAME>/ and writes, to the output directory:
  preprocessor.joblib   dict with the kept feature columns and the fitted scaler
  DT.joblib RF.joblib SVM.joblib   the bare scikit-learn estimators (compressed)
  MLP.keras LSTM.keras
  meta.json             settings, class names and the test result each model got when it was trained here
  sample_flows.csv      500 stratified flows from the TEST split, with their raw `Label`, for trying the app
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idsbench import common as c  # noqa: E402
from idsbench import runner  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "export"
out.mkdir(parents=True, exist_ok=True)
APP_RF_TREES = 40  # the full forest (146 trees) is 545 MB in memory and 98 MB on disk; the app ships its first 40 trees
D = c.prepare()
meta_all = {}
for name in ("DT", "RF", "SVM", "MLP", "LSTM"):
    m, pre, meta = runner.load_model(name)
    if name == "RF":
        full_trees = len(m.m.estimators_)
        m.m.estimators_ = m.m.estimators_[:APP_RF_TREES]
        m.m.n_estimators = APP_RF_TREES
        res = c.full_metrics(D["yte"], m.predict(D["Xte"]))  # test result of the reduced forest, so the app reports what it runs
        meta["test"] = dict(acc=res["acc"], macro_f1=res["macro_f1"], det_rate=res["det_rate"], fpr_normal=res["fpr_normal"],
                            recall=dict(zip(c.CLASSES, res["recall"])))
        meta["settings_from"] += f"; the app ships the first {APP_RF_TREES} of {full_trees} trees"
    if name == "DT":
        joblib.dump(dict(cols=pre.cols, scaler=pre.scaler), out / "preprocessor.joblib", compress=3)
    if name in ("DT", "RF", "SVM"):
        joblib.dump(m.m, out / f"{name}.joblib", compress=3)
    else:
        shutil.copy(runner.MODELS_DIR / name / "model.keras", out / f"{name}.keras")
    meta_all[name] = dict(config=meta["config"], epochs=meta.get("epochs"), settings_from=meta["settings_from"],
                          train_seconds=round(meta["train_seconds"], 1), test=meta["test"])
meta_all["classes"] = c.CLASSES
meta_all["dataset"] = c.DATASET
(out / "meta.json").write_text(json.dumps(meta_all, indent=1))

# 500 flows from the test split (stratified, fixed seed): raw feature values plus the raw label, as in the CSV files
test = D["frame"].loc[D["test_index"]]
per = max(1, 500 // len(c.CLASSES))
sample = pd.concat([g.sample(min(per, len(g)), random_state=0) for _, g in test.groupby("category")])
sample = sample.sample(frac=1, random_state=0).drop(columns=["category"]).rename(columns={"label": "Label"})
sample.to_csv(out / "sample_flows.csv", index=False)
print(f"sample_flows.csv: {len(sample)} flows, {sample['Label'].nunique()} different labels")

for f in sorted(out.iterdir()):
    print(f"{f.name:<22}{f.stat().st_size / 1e6:8.2f} MB")
