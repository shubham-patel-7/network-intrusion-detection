"""Aggregate the saved results into tables, statistics and figures.

Usage: python -m idsbench.report
Writes results/summary.json, results/results_tables.md and figures/*.png.
"""
import itertools
import json
import os

import matplotlib
import numpy as np
from sklearn.metrics import roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from . import common as c

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

M = ["DT", "RF", "SVM", "MLP", "LSTM"]
NAME = {"DT": "Decision Tree", "RF": "Random Forest", "SVM": "SVM (RBF)", "MLP": "MLP", "LSTM": "LSTM"}
COL = {"DT": "#7fa8d1", "RF": "#2b6cb0", "SVM": "#1a365d", "MLP": "#f0a050", "LSTM": "#c05621"}
SHORT = {"DT": "Dec. Tree", "RF": "Rand. Forest", "SVM": "SVM", "MLP": "MLP", "LSTM": "LSTM"}


def ms(v):
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def summarise(R):
    S = {}
    for m in M:
        r = R[m]

        def g(f):
            return [f(x) for x in r]

        S[m] = dict(
            acc=ms(g(lambda x: x["full"]["acc"])), mf1=ms(g(lambda x: x["full"]["macro_f1"])),
            fpr=ms(g(lambda x: x["full"]["fpr_normal"])), det=ms(g(lambda x: x["full"]["det_rate"])),
            auc=ms(g(lambda x: x["full"]["auc_macro"])), aucb=ms(g(lambda x: x["full"]["auc_binary"])),
            prec=np.mean(g(lambda x: x["full"]["precision"]), 0).tolist(),
            rec=np.mean(g(lambda x: x["full"]["recall"]), 0).tolist(),
            f1=np.mean(g(lambda x: x["full"]["f1"]), 0).tolist(),
            f1_sd=(np.std(g(lambda x: x["full"]["f1"]), 0, ddof=1).tolist() if len(r) > 1 else [0.0] * len(c.CLASSES)),
            train=ms(g(lambda x: x["cost"]["train_s"])), batch=ms(g(lambda x: x["cost"]["batch_s"])),
            lat=ms(g(lambda x: x["cost"]["lat_median_ms"])), lat95=ms(g(lambda x: x["cost"]["lat_p95_ms"])),
            size=ms(g(lambda x: x["cost"]["size_mb"])), counts=r[0]["cost"]["counts"], cfg=r[0]["cfg"],
            epochs=r[0]["cost"].get("epochs"), spe=ms(g(lambda x: x["cost"].get("s_per_epoch", 0))),
            seed0_f1=r[0]["full"]["f1"],  # per-class F1 of the seed-0 run (the ablations use seed 0)
            seed0_acc=r[0]["full"]["acc"],
            val_macro_f1=(c.load_json(f"best_{m}.json")["val_macro_f1"] if c.exists(f"best_{m}.json") else None))
        if "seen" in r[0]:  # held-out run: the test records of attack types that were in training
            S[m].update(s_acc=ms(g(lambda x: x["seen"]["acc"])), s_mf1=ms(g(lambda x: x["seen"]["macro_f1"])),
                        s_det=ms(g(lambda x: x["seen"]["det_rate"])), s_fpr=ms(g(lambda x: x["seen"]["fpr_normal"])))
        if "unseen" in r[0]:  # attack types absent from training
            S[m].update(u_det=ms(g(lambda x: x["unseen"]["det_rate"])), u_acc=ms(g(lambda x: x["unseen"]["acc"])),
                        u_recall={k: float(np.mean([x["unseen"]["recall"][k] for x in r])) for k in r[0]["unseen"]["recall"]})
    return S


def statistics(S, y, pred):
    """Bootstrap intervals for macro-F1 and McNemar tests with Holm correction (seed-0 predictions)."""
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(y), (1000, len(y)))
    for m in M:
        v = [c.macro_f1(y[i], pred[m][i]) for i in idx]
        S[m]["ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        S[m]["seed0_mf1"] = c.macro_f1(y, pred[m])  # the run the interval and the ablations refer to
    pairs = list(itertools.combinations(M, 2))
    pv, mc = [], {}
    for a, b in pairs:
        ca, cb = pred[a] == y, pred[b] == y
        t = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
        pv.append(float(mcnemar(t, exact=False, correction=True).pvalue))
        mc[f"{a}|{b}"] = dict(a_only=t[0][1], b_only=t[1][0])
    adj = multipletests(pv, method="holm")[1]
    for (a, b), p0, p1 in zip(pairs, pv, adj):
        mc[f"{a}|{b}"].update(p=p0, p_holm=float(p1))
    S["mcnemar"] = mc


def rq4_rule(S):
    """The decision rule of Section R, fixed before any result is seen."""
    best = max(M, key=lambda m: S[m]["mf1"][0])
    tie = [m for m in M if m != best and S[best]["mf1"][0] - S[m]["mf1"][0] < 0.01]
    constrained = [m for m in tie if S[m]["lat"][0] < S[best]["lat"][0]]
    runner_up = max((S[m]["mf1"][0] for m in M if m != best), default=0)
    accuracy_first = best if S[best]["mf1"][0] - runner_up > 0.01 else None
    return dict(best=best, practical_ties=tie, resource_constrained=constrained, accuracy_first=accuracy_first)


def tables_markdown(S):
    pm = lambda t, d=3, pct=False: (f"{t[0] * 100:.1f}% ± {t[1] * 100:.1f}" if pct else f"{t[0]:.{d}f} ± {t[1]:.{d}f}")
    test = "the held-out test split"
    L = [f"### Detection quality on {test} (mean ± standard deviation over seeds)", "",
         "| Model | Accuracy | Macro-F1 | Attack detection rate | False-positive rate (Normal) | ROC-AUC (macro) | ROC-AUC (Normal vs Attack) |",
         "|---|---|---|---|---|---|---|"]
    for m in M:
        s = S[m]
        L.append(f"| {NAME[m]} | {pm(s['acc'])} | {pm(s['mf1'])} | {pm(s['det'], pct=True)} | "
                 f"{pm(s['fpr'], pct=True)} | {pm(s['auc'])} | {pm(s['aucb'])} |")
    L += ["", "### Recall / F1 per class (mean over seeds)", "", "| Model | " + " | ".join(c.CLASSES) + " |",
          "|---|" + "---|" * len(c.CLASSES)]
    for m in M:
        L.append(f"| {NAME[m]} | " + " | ".join(f"{r:.2f} / {f:.2f}" for r, f in zip(S[m]["rec"], S[m]["f1"])) + " |")
    if "u_det" in S[M[0]]:  # held-out attack types
        names = [k for k in c.CLASSES[1:]]
        L += ["", "### Held-out (unseen) attack types", "",
              "| Model | Detection rate | Correct category | Recall per category (" + " / ".join(names) + ") |", "|---|---|---|---|"]
        for m in M:
            s = S[m]
            ur = " / ".join(f"{s['u_recall'][k]:.2f}" if k in s["u_recall"] else "-" for k in names)
            L.append(f"| {NAME[m]} | {pm(s['u_det'], pct=True)} | {pm(s['u_acc'], pct=True)} | {ur} |")
    L += ["", "### Computational cost (CPU, mean ± standard deviation)", "",
          "| Model | Training time (s) | Epochs | Batch inference, whole test set (s) | Latency per record, median (p95) ms | Model size (MB) | Size counts |",
          "|---|---|---|---|---|---|---|"]
    for m in M:
        s = S[m]
        cnt = ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in s["counts"].items())
        L.append(f"| {NAME[m]} | {pm(s['train'], 1)} | {s['epochs'] or '-'} | {pm(s['batch'], 3)} | "
                 f"{s['lat'][0]:.2f} ({s['lat95'][0]:.2f}) | {s['size'][0]:.2f} | {cnt} |")
    if "ci" in S["DT"]:
        L += ["", "### 95% bootstrap intervals for macro-F1 (seed 0)", "", "| Model | Interval |", "|---|---|"]
        for m in M:
            L.append(f"| {NAME[m]} | {S[m]['ci'][0]:.3f} to {S[m]['ci'][1]:.3f} |")
    return "\n".join(L) + "\n"


def figures(S, R, A, y, P):
    c.FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .25, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})

    def save(fig, n):
        fig.savefig(c.FIGS / f"{n}.png")
        plt.close(fig)

    # 1 main bars
    panels = [("acc", "Accuracy"), ("mf1", "Macro-F1")]
    K = len(c.CLASSES)
    fig, ax = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.4), squeeze=False)
    for a, (k, t) in zip(ax[0], panels):
        v = [S[m][k][0] for m in M]
        e = [S[m][k][1] for m in M]
        a.bar(range(5), v, yerr=e, color=[COL[m] for m in M], capsize=3)
        a.set_xticks(range(5))
        a.set_xticklabels([SHORT[m] for m in M], rotation=25, ha="right")
        a.set_title(t)
        lo = min(v) - .08
        a.set_ylim(max(lo, 0), min(max(v) + .04, 1))
        for i, x in enumerate(v):
            a.text(i, x + (e[i] or 0) + .003, f"{x:.3f}", ha="center", fontsize=8)
    save(fig, "fig_main")
    # 2 per-class F1
    fig, ax = plt.subplots(figsize=(max(8, 1.15 * K), 3.5))
    w = .16
    for j, m in enumerate(M):
        ax.bar(np.arange(K) + (j - 2) * w, S[m]["f1"], w, yerr=S[m]["f1_sd"], color=COL[m], label=NAME[m], capsize=2)
    ax.set_xticks(range(K))
    ax.set_xticklabels(c.CLASSES, rotation=0 if K <= 5 else 30, ha="center" if K <= 5 else "right")
    ax.set_ylabel("F1")
    ax.legend(ncol=5, fontsize=8, loc="lower left", frameon=False, bbox_to_anchor=(0, 1))
    save(fig, "fig_perclass")
    # 3 confusion matrices (seed 0, row-normalised)
    if K <= 5:
        fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
    else:  # eight classes: two rows of three panels keep the numbers legible
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.8))
        ax = axes.ravel()
        for extra in ax[len(M):]:
            extra.axis("off")
    for a, m in zip(ax, M):
        cm = np.array(R[m][0]["full"]["cm"])
        cn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        a.grid(False)
        for i in range(K):
            for j in range(K):
                a.text(j, i, f"{cn[i, j]:.2f}", ha="center", va="center", fontsize=8 if K <= 5 else 7,
                       color="white" if cn[i, j] > .5 else "black")
        a.set_xticks(range(K))
        a.set_yticks(range(K))
        a.set_xticklabels(c.CLASSES, rotation=45, fontsize=8)
        a.set_yticklabels(c.CLASSES, fontsize=8)
        a.set_title(NAME[m], fontsize=10)
        a.set_xlabel("Predicted")
        if K > 5 and a not in (ax[0], ax[3]):
            a.set_ylabel("")
    for a in (ax[0], ax[3]) if K > 5 else (ax[0],):
        a.set_ylabel("True (row-normalised)")
    if K > 5:
        fig.subplots_adjust(wspace=0.5, hspace=0.55)  # long class names would otherwise touch the next panel
    save(fig, "fig_cm")
    # 4 ROC: one-versus-rest per class + the binary curve
    nrows = int(np.ceil((K + 1) / 3))
    fig, ax = plt.subplots(nrows, 3, figsize=(11, 3.4 * nrows), constrained_layout=True, squeeze=False)
    for k, a in enumerate(ax.ravel()):
        if k > K:
            a.axis("off")  # unused cell of the grid
            continue
        for m in M:
            s = P[m]["s"]
            yy, sc = ((y == k).astype(int), s[:, k]) if k < K else ((y != 0).astype(int), -s[:, 0])
            f, t, _ = roc_curve(yy, sc)
            a.plot(f, t, color=COL[m], label=NAME[m], lw=1.4)
        a.plot([0, 1], [0, 1], color="grey", lw=.8, ls=":")
        a.set_title(c.CLASSES[k] if k < K else f"Binary: {c.CLASSES[0]} vs Attack")
        a.set_xlabel("False-positive rate")
        a.set_ylabel("True-positive rate")
    ax[0, 0].legend(fontsize=8, frameon=False)
    save(fig, "fig_roc")
    # 5 accuracy-versus-cost trade-off (RQ4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (k, t) in zip(ax, [("lat", "Median single-record latency (ms, log scale)"),
                              ("train", "Training time (s, log scale)")]):
        for m in M:
            a.errorbar(S[m][k][0], S[m]["mf1"][0], yerr=S[m]["mf1"][1], fmt="o", color=COL[m], ms=8, capsize=3)
            off, ha = {"RF": ((-8, 8), "right"), "MLP": ((-8, -14), "right"), "SVM": ((8, -15), "left"),
                       "LSTM": ((8, 6), "left")}.get(m, ((7, 5), "left"))
            a.annotate(NAME[m], (S[m][k][0], S[m]["mf1"][0]), textcoords="offset points", xytext=off, ha=ha, fontsize=8)
        a.margins(x=0.25)
        a.set_xscale("log")
        a.set_xlabel(t)
        a.set_ylabel("Macro-F1 on the test split")
    save(fig, "fig_tradeoff")
    # 6 learning curve (training-size ablation)
    if "size" in A:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        fr = [0.1, 0.25, 0.5]
        for m in M:
            if not all(f"{m}|{f}" in A["size"] for f in fr):
                continue  # a model skipped in a partial run
            v = [A["size"][f"{m}|{f}"]["full"]["macro_f1"] for f in fr] + [R[m][0]["full"]["macro_f1"]]
            ax.plot([10, 25, 50, 100], v, "o-", color=COL[m], label=NAME[m])
        ax.set_xlabel("Share of the training sample used (%)")
        ax.set_ylabel("Macro-F1")
        ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(-0.05, 1.0), columnspacing=1)
        save(fig, "fig_size")


def describe(D):
    """What the results are about, so summary.json (and anything that reads it, such as the app) is self-describing."""
    k = len(c.CLASSES)
    return dict(
        dataset=c.DATASET, classes=list(c.CLASSES), rare_classes=[c.CLASSES[i] for i in c.RARE],
        models=M, seeds=[s for s in c.SEEDS if c.exists(f"final_DT_{s}.json")],
        n_train=int(len(D["y"])), n_test=int(len(D["yte"])), n_features=int(D["X"].shape[1]),
        train_counts=[int(n) for n in np.bincount(D["y"], minlength=k)],
        test_counts=[int(n) for n in np.bincount(D["yte"], minlength=k)],
        n_unseen=int(D["unseen"].sum()) if D.get("unseen") is not None else 0,
        group_names=list(D.get("group_names") or []), feature_names=[str(n) for n in D["names"]],
        tuning_configs=(len(c.load_json("tune_DT.json")) if c.exists("tune_DT.json") else int(os.environ.get("IDS_N", "8"))),
        sample_cap_per_class=int(os.environ.get("IDS_CICIOT_CAP", "0")) or None,
        data_info=D.get("info"),
    )


def main():
    D = c.prepare()
    y = D["yte"]
    R = {m: [c.load_json(f"final_{m}_{s}.json") for s in c.SEEDS if c.exists(f"final_{m}_{s}.json")] for m in M}
    missing = [m for m in M if not R[m]]
    if missing:
        raise SystemExit(f"no final results for {missing}; run `python -m idsbench.final` first")
    S = summarise(R)
    P = {m: np.load(c.RES / f"preds_{m}_0.npz") for m in M}
    statistics(S, y, {m: P[m]["p"].astype(int) for m in M})
    S["rq4"] = rq4_rule(S)
    S["meta"] = describe(D)
    A = {k: c.load_json(f"ablate_{k}.json") for k in ["task", "imbalance", "features", "size"] if c.exists(f"ablate_{k}.json")}
    S["ablate"] = A
    c.RES.mkdir(parents=True, exist_ok=True)
    (c.RES / "summary.json").write_text(json.dumps(S, indent=1))
    (c.RES / "results_tables.md").write_text(tables_markdown(S), encoding="utf-8")
    figures(S, R, A, y, P)
    print("ok:", c.RES / "summary.json", c.RES / "results_tables.md", c.FIGS)


if __name__ == "__main__":
    main()
