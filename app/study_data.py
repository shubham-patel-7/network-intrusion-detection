"""Results of the five-model benchmark on CICIoT2023, loaded from app/data/summary.json.

The file is written by the benchmark project (`python -m idsbench.report`). Nothing in this module is typed in by hand
except the descriptions of what each model is: every number, and every sentence about the numbers, is computed from the
summary, so the app cannot drift away from the results. Set IDS_APP_SUMMARY to read a different summary file.
"""
import json
import os
from pathlib import Path

SUMMARY = Path(os.environ.get("IDS_APP_SUMMARY", Path(__file__).resolve().parent / "data" / "summary.json"))
_S = json.loads(SUMMARY.read_text(encoding="utf-8"))
META = _S["meta"]

CODES = ["DT", "RF", "SVM", "MLP", "LSTM"]
MODELS = ["Decision Tree", "Random Forest", "SVM (RBF)", "MLP", "LSTM"]
CODE = dict(zip(MODELS, CODES))
FAMILY = {"Decision Tree": "Classical", "Random Forest": "Classical", "SVM (RBF)": "Classical",
          "MLP": "Deep learning", "LSTM": "Deep learning"}
# Colours that read on a dark background; blue/indigo = classical, amber/orange = deep learning.
COLORS = {"Decision Tree": "#93c5fd", "Random Forest": "#3b82f6", "SVM (RBF)": "#818cf8",
          "MLP": "#fbbf24", "LSTM": "#f97316"}
CLASSES = META["classes"]
RARE = META["rare_classes"]  # names of the rare classes
DATASET = META["dataset"]
IS_CIC = DATASET == "ciciot2023"
DATASET_NAME = "CICIoT2023"


def _m(m):
    return _S[CODE[m]]


def _pct(t):
    return (t[0] * 100, t[1] * 100)


# class distribution of the sample the models were trained and tested on
DISTRIBUTION = {c: (META["train_counts"][i], META["test_counts"][i]) for i, c in enumerate(CLASSES)}

# main results on the held-out test split: mean and standard deviation over the seeds (rates in percent)
MAIN = {m: dict(acc=tuple(_m(m)["acc"]), mf1=tuple(_m(m)["mf1"]), det=_pct(_m(m)["det"]), fpr=_pct(_m(m)["fpr"]),
                auc=tuple(_m(m)["auc"]), aucb=tuple(_m(m)["aucb"])) for m in MODELS}
PER_CLASS = {m: {c: (_m(m)["rec"][i], _m(m)["f1"][i]) for i, c in enumerate(CLASSES)} for m in MODELS}
CI = {m: tuple(_m(m)["ci"]) for m in MODELS}
SEED0_MF1 = {m: _m(m)["seed0_mf1"] for m in MODELS}
COST = {m: dict(train=tuple(_m(m)["train"]), epochs=_m(m)["epochs"], batch=tuple(_m(m)["batch"]), lat=_m(m)["lat"][0],
                lat95=_m(m)["lat95"][0], size=_m(m)["size"][0],
                counts=", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in _m(m)["counts"].items())) for m in MODELS}


def _config_text(cfg):
    return ", ".join(f"{k.replace('_', ' ')} {v:.4g}" if isinstance(v, float) else f"{k.replace('_', ' ')} {v}"
                     for k, v in cfg.items())


CONFIG = {m: (_config_text(_m(m)["cfg"]), _m(m).get("val_macro_f1")) for m in MODELS}

A = _S.get("ablate", {})


def _rare_idx():
    return [CLASSES.index(r) for r in RARE]


# binary task (benign versus attack)
BINARY = {}
for _m_ in MODELS:
    _b = A.get("task", {}).get(CODE[_m_])
    if _b:
        BINARY[_m_] = dict(acc=_b["full"]["acc"], f1=_b["full"]["f1_attack"], det=_b["full"]["det_rate"] * 100,
                           fpr=_b["full"]["fpr"] * 100, auc=_b["full"]["auc"])
        if "unseen_det_rate" in _b:
            BINARY[_m_]["unseen"] = _b["unseen_det_rate"] * 100

# imbalance handling: (macro-F1, F1 of the first rare class, F1 of the second rare class)
IMBALANCE = {}
for _m_ in MODELS:
    _row = {}
    for _key, _label in (("none", "No correction"), ("smote", "SMOTE")):
        _e = A.get("imbalance", {}).get(f"{CODE[_m_]}|{_key}")
        if _e:
            _row[_label] = (_e["full"]["macro_f1"], *[_e["full"]["f1"][i] for i in _rare_idx()])
    if _row:
        _row["Class weights"] = (SEED0_MF1[_m_], *[_m(_m_)["seed0_f1"][i] for i in _rare_idx()])  # the main run, seed 0
        IMBALANCE[_m_] = {k: _row[k] for k in ("No correction", "Class weights", "SMOTE") if k in _row}

# feature ablation (best classical and best neural model on validation): (inputs, macro-F1, accuracy, unseen detection %)
FEATURES = {}
_feat = A.get("features")
if _feat:
    for _code in _feat.get("models", []):
        _name = MODELS[CODES.index(_code)]
        _rows = {"All features": (META["n_features"], SEED0_MF1[_name], _m(_name)["seed0_acc"], None)}
        for _key, _v in _feat.items():
            if _key.startswith(_code + "|") and _v["n_features"] < META["n_features"]:  # top-40 of 39 features is "all"
                _var = _key.split("|", 1)[1]
                _label = ("Without " + _var[5:].replace("_", " ") if _var.startswith("drop_")
                          else f"Top {_var[3:5]} by mutual information")
                _un = _v.get("unseen", {}).get("det_rate")
                _rows[_label] = (_v["n_features"], _v["full"]["macro_f1"], _v["full"]["acc"], None if _un is None else _un * 100)
        FEATURES[_name] = _rows

# training-set size: macro-F1 at 10%, 25%, 50% and 100% of the training sample
SIZE_PERCENT = [10, 25, 50, 100]
SIZE = {}
for _m_ in MODELS:
    _vals = [A.get("size", {}).get(f"{CODE[_m_]}|{f}", {}).get("full", {}).get("macro_f1") for f in (0.1, 0.25, 0.5)]
    if all(v is not None for v in _vals):
        SIZE[_m_] = _vals + [SEED0_MF1[_m_]]

HAS_UNSEEN = bool(META.get("n_unseen"))

# held-out attack-type experiment (optional file next to the summary): seven attack types removed from training
HOLDOUT_SUMMARY = SUMMARY.with_name("summary_holdout.json")
UNSEEN, HOLDOUT_TYPES, HOLDOUT_META = {}, [], {}
if HOLDOUT_SUMMARY.exists():
    _H = json.loads(HOLDOUT_SUMMARY.read_text(encoding="utf-8"))
    HOLDOUT_META = _H["meta"]
    HOLDOUT_TYPES = HOLDOUT_META["data_info"]["holdout"]
    for _m_ in MODELS:
        _h = _H[CODE[_m_]]
        UNSEEN[_m_] = dict(mf1=_h["mf1"][0], seen_mf1=_h["s_mf1"][0], det=_h["u_det"][0] * 100, acc=_h["u_acc"][0] * 100,
                           recall=_h["u_recall"])
MCNEMAR = _S.get("mcnemar", {})
RQ4 = _S.get("rq4", {})


# ---------------------------------------------------------------- what each model is (dataset independent)
HOW = {
    "Decision Tree": "A CART tree that splits on the feature and threshold with the largest drop in impurity. Depth and leaf "
                     "size are the main controls on overfitting.",
    "Random Forest": "An ensemble of trees, each grown on a bootstrap sample and restricted to a random subset of features at "
                     "every split. The predicted class is the one with the highest averaged probability.",
    "SVM (RBF)": "A soft-margin support vector machine with the RBF kernel, extended to several classes one-versus-one. Training "
                 "cost grows faster than linearly, so it is fitted on a class-stratified subsample that keeps every record of "
                 "the rare classes.",
    "MLP": "A feed-forward network with two hidden ReLU layers, dropout after each, and a softmax output over the classes. "
           "It is the plain neural baseline.",
    "LSTM": "A long short-term memory cell followed by dropout, a dense layer, dropout and a softmax output. Each flow is "
            "reshaped to a sequence of one time step, so the cell acts as a gated feed-forward layer.",
}
TAGLINE = {
    "Decision Tree": "One tree. The cheapest to train and to run.",
    "Random Forest": "Many trees voting together.",
    "SVM (RBF)": "A curved decision boundary on a stratified subsample.",
    "MLP": "Two hidden layers.",
    "LSTM": "A recurrent cell over a sequence of one time step.",
}


# ---------------------------------------------------------------- sentences computed from the numbers
def _rank(key, idx=0, reverse=True):
    return sorted(MODELS, key=lambda m: MAIN[m][key][idx], reverse=reverse)


def _best_of(models, key="mf1"):
    return max(models, key=lambda m: MAIN[m][key][0])


CLASSICAL = [m for m in MODELS if FAMILY[m] == "Classical"]
NEURAL = [m for m in MODELS if FAMILY[m] != "Classical"]
BEST = _rank("mf1")[0]
WORST = _rank("mf1")[-1]


def _rare_recall(m):
    return [PER_CLASS[m][r][0] for r in RARE]


def _hardest_class():
    mean_f1 = {c: sum(PER_CLASS[m][c][1] for m in MODELS) / len(MODELS) for c in CLASSES}
    return min(mean_f1, key=mean_f1.get), mean_f1


def _significant(a, b):
    """Holm-adjusted McNemar p-value of two models on seed 0 (order of the pair does not matter)."""
    e = MCNEMAR.get(f"{CODE[a]}|{CODE[b]}") or MCNEMAR.get(f"{CODE[b]}|{CODE[a]}")
    return None if not e else e["p_holm"]


def _findings():
    out = []
    srt = _rank("mf1")
    out.append((f"{BEST} leads on macro-F1",
                f"{BEST} scores macro-F1 {MAIN[BEST]['mf1'][0]:.3f}, ahead of {srt[1]} ({MAIN[srt[1]]['mf1'][0]:.3f}); "
                f"the lowest is {WORST} ({MAIN[WORST]['mf1'][0]:.3f})."))
    lows = sorted(((m, min(_rare_recall(m))) for m in MODELS), key=lambda t: t[1])
    out.append((f"Rare classes are the hard part",
                f"Recall on {' and '.join(RARE)} ranges from {lows[0][1]:.2f} ({lows[0][0]}) to {lows[-1][1]:.2f} "
                f"({lows[-1][0]}) at its worst, while the large classes are mostly handled well."))
    t = {m: COST[m]["train"][0] for m in MODELS}
    fast, slow = min(t, key=t.get), max(t, key=t.get)
    out.append(("Cost differs by orders of magnitude",
                f"Training takes {t[fast]:.1f} s for {fast} and {t[slow]:.0f} s for {slow}; per-record latency runs from "
                f"{min(COST[m]['lat'] for m in MODELS):.2f} to {max(COST[m]['lat'] for m in MODELS):.2f} ms."))
    gaps = {m: (CONFIG[m][1] - MAIN[m]["mf1"][0]) for m in MODELS if CONFIG[m][1] is not None}
    if gaps:
        g = max(gaps, key=gaps.get)
        out.append(("Validation overstates test scores",
                    f"Validation macro-F1 is higher than test macro-F1 for {sum(v > 0 for v in gaps.values())} of 5 models, by up "
                    f"to {gaps[g]:.3f} ({g})."))
    acc = [MAIN[m]["acc"][0] for m in MODELS]
    mf = [MAIN[m]["mf1"][0] for m in MODELS]
    out.append(("Accuracy and macro-F1 tell different stories",
                f"Accuracy ranges {min(acc):.3f} to {max(acc):.3f} while macro-F1 ranges {min(mf):.3f} to {max(mf):.3f}: macro-F1 "
                f"counts every class equally, so a model cannot hide a weak rare class behind the large ones."))
    return out


FINDINGS = _findings()


def _research_questions():
    qs = []
    bc, bn = _best_of(CLASSICAL), _best_of(NEURAL)
    d = MAIN[bn]["mf1"][0] - MAIN[bc]["mf1"][0]
    p = _significant(bc, bn)
    lead = bn if d > 0 else bc
    if abs(d) < 0.01:  # the study's rule: a macro-F1 gap below 0.01 is a practical tie, whatever the p-value
        verdict = f"differ by only {abs(d):.3f}, below the 0.01 margin the study treats as a practical tie"
    else:
        sig = (", a difference that is statistically significant after correction" if p is not None and p < 0.05
               else ", not a statistically clear difference" if p is not None else "")
        verdict = f"differ by {abs(d):.3f} in favour of {lead}{sig}"
    qs.append(("Q1. Do the deep learning models detect attacks better than the classical ones?",
               f"The best neural model ({bn}, macro-F1 {MAIN[bn]['mf1'][0]:.3f}) and the best classical model ({bc}, "
               f"{MAIN[bc]['mf1'][0]:.3f}) {verdict}. The overall best is {BEST}. Accuracy: {bn} {MAIN[bn]['acc'][0]:.3f} "
               f"against {bc} {MAIN[bc]['acc'][0]:.3f}."))
    ct = {m: COST[m]["train"][0] for m in MODELS}
    qs.append(("Q2. What does any gain cost?",
               f"Training time: classical models {min(ct[m] for m in CLASSICAL):.1f} to {max(ct[m] for m in CLASSICAL):.1f} s, "
               f"neural models {min(ct[m] for m in NEURAL):.0f} to {max(ct[m] for m in NEURAL):.0f} s. Per-record latency: "
               f"{min(COST[m]['lat'] for m in MODELS):.2f} ms ({min(MODELS, key=lambda m: COST[m]['lat'])}) to "
               f"{max(COST[m]['lat'] for m in MODELS):.2f} ms ({max(MODELS, key=lambda m: COST[m]['lat'])}); the largest model is "
               f"{max(MODELS, key=lambda m: COST[m]['size'])} at {max(COST[m]['size'] for m in MODELS):.1f} MB. Timings are CPU-only."))
    hardest, mean_f1 = _hardest_class()
    qs.append(("Q3. Which classes are hardest to detect?",
               f"The hardest class on average is {hardest} (mean F1 {mean_f1[hardest]:.2f}). The rare classes {' and '.join(RARE)} "
               f"reach recall of " + "; ".join(f"{m} {min(_rare_recall(m)):.2f}-{max(_rare_recall(m)):.2f}" for m in MODELS) + "."))
    if RQ4:
        ties = RQ4.get("practical_ties") or []
        cons = RQ4.get("resource_constrained") or []
        af = RQ4.get("accuracy_first")
        qs.append(("Q4. Which model suits a resource-constrained deployment, and which an accuracy-first one?",
                   f"The best macro-F1 is {RQ4['best']}'s. Models within 0.01 of it (a practical tie): " +
                   (", ".join(ties) if ties else "none") + ". Of those, the ones that are also faster per record: " +
                   (", ".join(cons) if cons else "none") + ". Accuracy-first choice (best by more than 0.01): " +
                   (af if af else "no model leads by that margin") + "."))
    return qs


RESEARCH_QUESTIONS = _research_questions()


def _stood_out(m):
    out = []
    srt = _rank("mf1")
    out.append(f"Ranks {srt.index(m) + 1} of 5 on macro-F1 ({MAIN[m]['mf1'][0]:.3f}) and {_rank('acc').index(m) + 1} of 5 on "
               f"accuracy ({MAIN[m]['acc'][0]:.3f}).")
    cls = PER_CLASS[m]
    best_c, worst_c = max(CLASSES, key=lambda c: cls[c][1]), min(CLASSES, key=lambda c: cls[c][1])
    out.append(f"Best class: {best_c} (F1 {cls[best_c][1]:.2f}); weakest: {worst_c} (F1 {cls[worst_c][1]:.2f}, recall {cls[worst_c][0]:.2f}).")
    c = COST[m]
    tr, lat = sorted(MODELS, key=lambda x: COST[x]["train"][0]).index(m) + 1, sorted(MODELS, key=lambda x: COST[x]["lat"]).index(m) + 1
    out.append(f"Cost: {c['train'][0]:.1f} s to train (rank {tr} of 5 from fastest), {c['lat']:.2f} ms per record (rank {lat}), "
               f"{c['size']:.2f} MB.")
    return out


MODEL_NOTES = {m: dict(tagline=TAGLINE[m], how=HOW[m], stood_out=_stood_out(m)) for m in MODELS}

LIMITATIONS = [
    f"CICIoT2023 comes from a laboratory IoT testbed with simulated attacks, so results describe offline classification of "
    f"pre-extracted flow features, not a live network.",
    f"The models were trained and tested on a sample of at most {META.get('sample_cap_per_class') or 'all'} flows per class "
    f"({META['n_train'] + META['n_test']:,} flows after removing exact duplicates), not on every flow of the dataset; class "
    f"proportions in the sample differ from those of the full data.",
    "The train and test sets are a random 80/20 split of flows, so flows from the same attack run can fall on both sides; scores "
    "are probably higher than they would be on traffic from a different capture or testbed.",
    "Ablations use one seed and frozen hyperparameters; differences smaller than the seed-to-seed spread are not interpreted.",
    f"The tuning budget was {META['tuning_configs']} random configurations per model and {len(META['seeds'])} seeds, kept small "
    "because of the size of the data; every model got the same budget.",
    "The SVM is fitted on a stratified subsample of at most 30,000 records, which shortens its training and changes the class "
    "balance it sees.",
    "The LSTM sees each flow as a sequence of one time step, so it is a gated feed-forward layer and not a sequence model.",
    "Costs were measured on one CPU machine without a GPU; absolute times will differ on other hardware. No decision thresholds "
    "were tuned.",
]

REFERENCES = [
    "E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani, CICIoT2023: A real-time dataset and benchmark "
    "for large-scale attacks in IoT environment, Sensors 23(13), 5941, 2023.",
    "N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, SMOTE: Synthetic minority over-sampling technique, JAIR 16, 2002.",
    "L. Breiman, Random forests, Machine Learning 45(1), 2001.",
    "S. Hochreiter and J. Schmidhuber, Long short-term memory, Neural Computation 9(8), 1997.",
    "D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, ICLR, 2015.",
    "T. G. Dietterich, Approximate statistical tests for comparing supervised classification learning algorithms, Neural Computation 10(7), 1998.",
    "F. Pedregosa et al., Scikit-learn: Machine learning in Python, JMLR 12, 2011.",
]

AUTHORS = ["Shubham Patel", "Vinamra Lilariaa", "Tejas Baid", "Anushka Chitranshi"]
