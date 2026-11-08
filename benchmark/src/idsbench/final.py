"""Final stage: refit the frozen configurations on the full training split, several seeds, evaluate once on the test split,
and time everything.

Usage: python -m idsbench.final [DT RF SVM MLP LSTM]
"""
import sys
import time

import numpy as np

from . import common as c


def run(names):
    D = c.prepare()
    X, y, Xte = D["X"], D["y"], D["Xte"]
    c.init_backend()
    for name in names:
        best = c.load_json(f"best_{name}.json")
        cfg, ep = best["cfg"], best.get("epochs")
        for seed in c.SEEDS:
            fn = f"final_{name}_{seed}.json"
            if c.exists(fn):
                continue
            m = c.make_model(name, cfg, seed=seed)
            t0 = time.perf_counter()
            if name in ("MLP", "LSTM"):
                m.fit(X, y, epochs=ep)  # epochs found by early stopping on the validation portion
            else:
                m.fit(X, y)
            train_s = time.perf_counter() - t0
            cost = c.time_inference(m, Xte)
            cost["train_s"] = train_s
            mb, cnt = m.size_info()
            cost["size_mb"] = mb
            cost["counts"] = cnt
            if name in ("MLP", "LSTM"):
                cost["epochs"] = ep
                cost["s_per_epoch"] = train_s / ep
            p = m.predict(Xte)
            s = m.scores(Xte)
            res = dict(model=name, seed=seed, cfg=cfg, cost=cost, **c.view_metrics(D, p, s))
            c.RES.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(c.RES / f"preds_{name}_{seed}.npz", p=p.astype(np.int8), s=s.astype(np.float32))
            c.save_json(res, fn)
            print(name, seed, "acc", round(res["full"]["acc"], 4), "mF1", round(res["full"]["macro_f1"], 4),
                  "train", round(train_s, 1), "batch", round(cost["batch_s"], 2), flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or ["DT", "RF", "SVM", "MLP", "LSTM"])
