"""Random-search tuning: 20 configurations per model, scored by macro-F1 on the validation portion.

Usage: python -m idsbench.tune DT RF SVM MLP LSTM
Finished configurations are saved after every fit, so an interrupted run resumes where it stopped.
"""
import sys
import time

from . import common as c


def tune(name):
    fn = f"tune_{name}.json"
    D = c.prepare()
    X, y, tr, va = D["X"], D["y"], D["idx_tr"], D["idx_val"]
    if name in ("MLP", "LSTM"):
        c.init_backend()
    cfgs = c.sample_configs(name)
    done = c.load_json(fn) if c.exists(fn) else []
    for i, cfg in enumerate(cfgs):
        if i < len(done):
            continue
        m = c.make_model(name, cfg, seed=0)
        t0 = time.perf_counter()
        if name in ("MLP", "LSTM"):
            m.fit(X[tr], y[tr], X[va], y[va])
        else:
            m.fit(X[tr], y[tr])
        fit_s = time.perf_counter() - t0
        f1 = c.macro_f1(y[va], m.predict(X[va]))
        done.append(dict(i=i, cfg=cfg, val_macro_f1=f1, fit_s=fit_s, epochs=getattr(m, "epochs", None)))
        c.save_json(done, fn)
        print(name, i, round(f1, 4), round(fit_s, 1), cfg, flush=True)
    best = max(done, key=lambda r: r["val_macro_f1"])
    c.save_json(best, f"best_{name}.json")
    print("BEST", name, best, flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        tune(n)
