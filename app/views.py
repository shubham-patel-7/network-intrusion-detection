"""The study pages: Overview, Models, Results, Ablations and About. Every number comes from study_data.py, which reads
the benchmark's summary.json; nothing here is typed in by hand."""
import altair as alt
import pandas as pd
import streamlit as st

import study_data as S
import ui

M, COLORS = S.MODELS, S.COLORS
MODEL_SCALE = ui.color_scale(M, COLORS)
MODEL_COLOR = alt.Color("Model:N", scale=MODEL_SCALE, legend=None)  # models are labelled on the axis
MODEL_COLOR_LEGEND = alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(title=None))
META = S.META
SRC = (f"Source: the benchmark project's results on {S.DATASET_NAME} ({META['n_train']:,} training and {META['n_test']:,} test "
       f"flows; {META['tuning_configs']} tuning configurations and {len(META['seeds'])} seeds per model).")


def _go(page):
    st.session_state["page"] = page


def _try(model):
    """Open the Detector with this model selected."""
    st.session_state["detector_model"] = model
    st.session_state["page"] = "🛡️ Detector"


def _domain(values, pad=0.05, lo_floor=0.0, hi_cap=1.0):
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.02)
    return [max(lo_floor, round(lo - pad - span * 0.1, 2)), min(hi_cap, round(hi + pad * 0.5, 2))]


def _card_items():
    items = []
    for m in M:
        c = S.COST[m]
        items.append(dict(name=m, family=S.FAMILY[m], color=COLORS[m], tagline=S.MODEL_NOTES[m]["tagline"],
                          stats=[(f"{S.MAIN[m]['mf1'][0]:.3f}", "macro-F1"), (f"{S.MAIN[m]['acc'][0]:.3f}", "accuracy"),
                                 (f"{c['train'][0]:.1f} s", "training time"), (f"{c['lat']:.2f} ms", "per record")]))
    return items


# ---------------------------------------------------------------- Overview
def overview():
    best = S.BEST
    ui.hero(f"Five classifiers · {S.DATASET_NAME}",
            "Can <em>deep learning</em> beat classical machine learning at spotting IoT attacks?",
            f"A controlled benchmark of five classifiers on {S.DATASET_NAME}: same data, same split, same tuning budget, and results "
            f"reported per class, for the rare classes and for computational cost, not on accuracy alone.",
            [("Decision Tree", "c"), ("Random Forest", "c"), ("SVM", "c"), ("MLP", "d"), ("LSTM", "d"),
             (S.DATASET_NAME, ""), (f"{len(META['seeds'])} seeds", "")])
    ct = {m: S.COST[m]["train"][0] for m in M}
    nt = [ct[m] for m in S.NEURAL]
    cl = [ct[m] for m in S.CLASSICAL]
    ui.kpis([("5", "classifiers compared", "3 classical, 2 deep"),
             (f"{META['n_test']:,}", "test flows", f"{len(S.CLASSES)} classes: benign + {len(S.CLASSES) - 1} attack categories"),
             (f"{META['n_features']}", "flow features", "scaled to 0-1"),
             (f"{S.MAIN[best]['mf1'][0]:.3f}", f"best macro-F1 ({best})", f"lowest: {S.MAIN[S.WORST]['mf1'][0]:.3f} ({S.WORST})"),
             (f"{min(nt):.0f}-{max(nt):.0f} s", "neural training time on CPU", f"vs {min(cl):.1f} to {max(cl):.1f} s classical")])

    ui.section("Meet the five models",
               f"Three classical models and two neural networks, trained and tested under exactly the same conditions. Scores are "
               f"from the held-out test split (mean over {len(META['seeds'])} seeds); times are on CPU.")
    ui.model_cards(_card_items())
    st.button("🤖 Explore each model in detail", on_click=_go, args=("🤖 Models",))

    ui.section("What the study found")
    ui.cards(S.FINDINGS)

    ui.section("Where to look next")
    c1, c2, c3 = st.columns(3)
    c1.button("📊 Full results", on_click=_go, args=("📊 Results",), width="stretch")
    c2.button("🧪 Ablations", on_click=_go, args=("🧪 Ablations",), width="stretch")
    c3.button("🛡️ Run the five models", on_click=_go, args=("🛡️ Detector",), width="stretch")

    tr_tot, te_tot = sum(v[0] for v in S.DISTRIBUTION.values()), sum(v[1] for v in S.DISTRIBUTION.values())
    ui.section("The data",
               f"{S.DATASET_NAME} records IoT traffic with {len(S.CLASSES) - 1} attack categories plus benign flows. The models were "
               f"trained and tested on a per-class sample (at most {META.get('sample_cap_per_class') or 'all'} flows per class), "
               f"after removing exact duplicates. The rare classes, {' and '.join(S.RARE)}, are the smallest.")
    rows = []
    for cls, (tr, te) in S.DISTRIBUTION.items():
        rows += [dict(Class=cls, Set="Training", Share=tr / tr_tot * 100, Records=tr),
                 dict(Class=cls, Set="Test", Share=te / te_tot * 100, Records=te)]
    chart = (alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=-25, title=None)),
                     y=alt.Y("Share:Q", title="Share of records (%)"), xOffset="Set:N",
                     color=alt.Color("Set:N", scale=alt.Scale(domain=["Training", "Test"], range=["#64748b", "#38bdf8"]),
                                     legend=alt.Legend(title=None)),
                     tooltip=["Class", "Set", alt.Tooltip("Share:Q", format=".2f"), alt.Tooltip("Records:Q", format=",")]))
    st.altair_chart(ui.style(chart, 300), width="stretch")
    smallest = min(S.DISTRIBUTION, key=lambda c: S.DISTRIBUTION[c][0])
    ui.note(f"{smallest} has {S.DISTRIBUTION[smallest][0]:,} training records against {max(v[0] for v in S.DISTRIBUTION.values()):,} "
            f"for the largest class, so macro-F1, which counts every class equally, is the headline score.", warn=True)

    ui.section("How the benchmark works")
    st.graphviz_chart(f"""
    digraph G {{ rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#243044", fontcolor="#e6edf7", fillcolor="#111827", penwidth=1.2];
      edge [color="#64748b", arrowsize=0.7];
      raw [label="{S.DATASET_NAME}\\nall flows (CSV files)"]; samp [label="Sample per class,\\ndrop exact duplicates"];
      split [label="Stratified 80/20\\ntrain / test"]; val [label="Train: 80/20\\ntraining / validation"];
      prep [label="Min-max scaling\\nfit on training only"];
      cl [label="Decision Tree\\nRandom Forest\\nSVM", color="#818cf8"]; dl [label="MLP\\nLSTM", color="#f97316"];
      tune [label="{META['tuning_configs']} random configs each,\\nvalidation macro-F1"]; final [label="Refit on full train\\n{len(META['seeds'])} seeds"];
      test [label="Evaluate once\\non the test split", color="#38bdf8"];
      out [label="Metrics, cost, bootstrap,\\nMcNemar + Holm", color="#38bdf8"];
      raw -> samp -> split -> val -> prep; prep -> cl; prep -> dl; cl -> tune; dl -> tune; tune -> final -> test -> out; }}
    """, width="stretch")

    ui.section("Four questions, answered from the numbers")
    for q, a in S.RESEARCH_QUESTIONS:
        with st.expander(q):
            st.write(a)
    ui.source(SRC)


# ---------------------------------------------------------------- Models
def models():
    ui.section("The five models",
               f"Every model gets the same data, the same split, the same metric code and the same tuning budget "
               f"({META['tuning_configs']} random configurations, scored by validation macro-F1). Scores are from the held-out test "
               f"split, mean over {len(META['seeds'])} seeds.")
    ui.model_cards(_card_items())

    ui.section("Look closer")
    pick = st.radio("Model", M, horizontal=True, key="model_pick", label_visibility="collapsed")
    note, fam = S.MODEL_NOTES[pick], S.FAMILY[pick]
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"### {pick}  ·  {fam}")
        st.write(note["how"])
        cfg, val = S.CONFIG[pick]
        st.caption(f"Chosen configuration: {cfg}." + (f" Validation macro-F1 {val:.3f}, against {S.MAIN[pick]['mf1'][0]:.3f} on the test split." if val is not None else ""))
        st.markdown("**What stood out**")
        for s in note["stood_out"]:
            st.markdown(f"- {s}")
        st.button(f"🛡️ Run {pick} on your own flows", on_click=_try, args=(pick,))
    with right:
        rows = [dict(Class=c, Measure=k, Value=S.PER_CLASS[pick][c][i]) for c in S.CLASSES for i, k in enumerate(["Recall", "F1"])]
        ch = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=-30, title=None)), xOffset="Measure:N",
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Recall", "F1"], range=[COLORS[pick], "#64748b"]),
                            legend=alt.Legend(title=None)),
            tooltip=["Class", "Measure", alt.Tooltip("Value:Q", format=".2f")])
        st.altair_chart(ui.style(ch, 260), width="stretch")
        st.caption(f"{pick}: recall and F1 per class on the test split (mean over {len(META['seeds'])} seeds).")
    s, c = S.MAIN[pick], S.COST[pick]
    rare_rec = S._rare_recall(pick)
    ui.kpis([(f"{s['acc'][0]:.3f}", "accuracy", f"± {s['acc'][1]:.3f}"), (f"{s['mf1'][0]:.3f}", "macro-F1", f"± {s['mf1'][1]:.3f}"),
             (f"{s['det'][0]:.1f}%", "attacks flagged", f"{s['fpr'][0]:.1f}% false alarms on benign"),
             (f"{min(rare_rec):.2f}", "weakest rare-class recall", " / ".join(f"{r} {v:.2f}" for r, v in zip(S.RARE, rare_rec))),
             (f"{c['lat']:.2f} ms", "per record", f"{c['train'][0]:.1f} s to train · {c['size']:g} MB")])

    ui.section("All five side by side")
    rows = []
    for m in M:
        r = S.MAIN[m]
        row = {"Model": m, "Family": S.FAMILY[m], "Macro-F1": r["mf1"][0], "Accuracy": r["acc"][0], "Attacks flagged (%)": r["det"][0],
               "False alarms (%)": r["fpr"][0]}
        for rc in S.RARE:
            row[f"{rc} recall"] = S.PER_CLASS[m][rc][0]
        row.update({"Training (s)": S.COST[m]["train"][0], "Per record (ms)": S.COST[m]["lat"], "Size (MB)": S.COST[m]["size"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    mf = [S.MAIN[m]["mf1"][0] for m in M]
    ac = [S.MAIN[m]["acc"][0] for m in M]
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Macro-F1": st.column_config.ProgressColumn(format="%.3f", min_value=max(0.0, min(mf) - 0.1), max_value=min(1.0, max(mf) + 0.05)),
        "Accuracy": st.column_config.ProgressColumn(format="%.3f", min_value=max(0.0, min(ac) - 0.1), max_value=min(1.0, max(ac) + 0.05))})
    ui.source(SRC)


# ---------------------------------------------------------------- Results
def _bar_with_error(df, value, err, title, fmt):
    d = df.copy()
    d["lo"], d["hi"] = d[value] - d[err], d[value] + d[err]
    base = alt.Chart(d)
    bars = base.mark_bar(cornerRadiusEnd=5, size=46).encode(
        x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y(f"{value}:Q", title=title, scale=alt.Scale(zero=True)), color=MODEL_COLOR,
        tooltip=["Model", alt.Tooltip(f"{value}:Q", format=fmt, title=title), alt.Tooltip(f"{err}:Q", format=fmt, title="± std dev")])
    rules = base.mark_rule(color="#e6edf7", strokeWidth=1.6).encode(x=alt.X("Model:N", sort=M), y="lo:Q", y2="hi:Q")
    labels = base.mark_text(dy=-10, color="#e6edf7", fontSize=12, fontWeight=600).encode(
        x=alt.X("Model:N", sort=M), y="hi:Q", text=alt.Text(f"{value}:Q", format=fmt))
    return bars + rules + labels


_TYPE_NAMES = {"DDOS-PSHACK_FLOOD": "DDoS-PSHACK flood", "DOS-SYN_FLOOD": "DoS-SYN flood", "MIRAI-GREIP_FLOOD": "Mirai-GRE IP flood",
               "RECON-HOSTDISCOVERY": "Recon-host discovery", "DNS_SPOOFING": "DNS spoofing", "XSS": "XSS",
               "SQLINJECTION": "SQL injection"}


def _unseen_section():
    n_unseen = S.HOLDOUT_META["n_unseen"]
    ui.section("Attack types the models never saw",
               f"A second, smaller run removed {len(S.HOLDOUT_TYPES)} attack types from training ({', '.join(_TYPE_NAMES.get(t, t) for t in S.HOLDOUT_TYPES)}). "
               f"They appear only in the test split, where they are {n_unseen:,} of {S.HOLDOUT_META['n_test']:,} flows. One seed per model.")
    rows = []
    for m in M:
        u = S.UNSEEN[m]
        rows += [dict(Model=m, Measure="Flagged as an attack at all", Value=u["det"]),
                 dict(Model=m, Measure="Given the correct category", Value=u["acc"])]
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Model:N", sort=M, title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("Measure:N"),
        y=alt.Y("Value:Q", title="Share of unseen flows (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Measure:N", scale=alt.Scale(range=["#38bdf8", "#f59e0b"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Measure", alt.Tooltip("Value:Q", format=".1f")])
    st.altair_chart(ui.style(chart, 320), width="stretch")
    cats = [c for c in S.CLASSES if c in S.UNSEEN[M[0]]["recall"]]
    tab = pd.DataFrame([dict(Model=m, **{"Macro-F1, all test flows": f"{S.UNSEEN[m]['mf1']:.3f}",
                                         "Macro-F1, seen types only": f"{S.UNSEEN[m]['seen_mf1']:.3f}"},
                             **{f"Unseen {c}: correct category": f"{S.UNSEEN[m]['recall'][c]:.2f}" for c in cats}) for m in M])
    st.dataframe(tab, hide_index=True, width="stretch")
    spread = {c: (min(S.UNSEEN[m]["recall"][c] for m in M), max(S.UNSEEN[m]["recall"][c] for m in M)) for c in cats}
    widest = max(cats, key=lambda c: spread[c][1] - spread[c][0])
    hardest = min(cats, key=lambda c: spread[c][1])
    dets, accs = [S.UNSEEN[m]["det"] for m in M], [S.UNSEEN[m]["acc"] for m in M]
    ui.note(f"Between {min(dets):.0f}% and {max(dets):.0f}% of the unseen flows are noticed as attacks, but only {min(accs):.0f}% to "
            f"{max(accs):.0f}% get the right category. Models disagree most on {widest} ({spread[widest][0]:.2f} to "
            f"{spread[widest][1]:.2f} correct), and no model recognises the unseen {hardest} type (best {spread[hardest][1]:.2f}).")


def results():
    ui.section("Results", f"Bars show the mean over {len(META['seeds'])} seeds on the held-out test split of {META['n_test']:,} flows; "
               "whiskers are one standard deviation.")
    metrics = {"Macro-F1 (headline)": ("mf1", "Macro-F1", ".3f"), "Accuracy": ("acc", "Accuracy", ".3f"),
               "Attack detection rate (%)": ("det", "Attack detection rate (%)", ".1f"),
               "False-positive rate on benign (%)": ("fpr", "False-positive rate (%)", ".1f"),
               "ROC-AUC, macro one-vs-rest": ("auc", "ROC-AUC (macro)", ".3f"),
               "ROC-AUC, benign vs attack": ("aucb", "ROC-AUC (binary)", ".3f")}
    pick = st.selectbox("Metric", list(metrics))
    key, title, fmt = metrics[pick]
    df = pd.DataFrame([dict(Model=m, Value=S.MAIN[m][key][0], SD=S.MAIN[m][key][1], Family=S.FAMILY[m]) for m in M])
    st.altair_chart(ui.style(_bar_with_error(df, "Value", "SD", title, fmt), 340), width="stretch")
    if key == "mf1":
        ui.note("Macro-F1 gives each class equal weight, so it is the headline score: accuracy is dominated by the large classes.")

    ui.section("How sure can we be? 95% bootstrap intervals for macro-F1",
               "1,000 resamples of the test flows for the seed-0 run (the dot). Models whose intervals overlap are not clearly "
               f"different. Seed 0 can sit above or below the {len(META['seeds'])}-seed mean shown in the bar chart.")
    ci = pd.DataFrame([dict(Model=m, Low=S.CI[m][0], High=S.CI[m][1], Mid=S.SEED0_MF1[m]) for m in M])
    dom = _domain(list(ci.Low) + list(ci.High))
    base = alt.Chart(ci)
    chart = (base.mark_rule(strokeWidth=7, strokeCap="round").encode(
        y=alt.Y("Model:N", sort=M, title=None), x=alt.X("Low:Q", title="Macro-F1", scale=alt.Scale(domain=dom)),
        x2="High:Q", color=MODEL_COLOR, tooltip=["Model", alt.Tooltip("Low:Q", format=".3f"), alt.Tooltip("High:Q", format=".3f")])
             + base.mark_point(color="#0a0f1c", filled=True, size=70, stroke="#e6edf7", strokeWidth=1.5)
             .encode(y=alt.Y("Model:N", sort=M), x="Mid:Q"))
    st.altair_chart(ui.style(chart, 230), width="stretch")

    hardest, _ = S._hardest_class()
    ui.section("Where the models differ: the classes",
               f"The hardest class on average is {hardest}. Rare classes ({' and '.join(S.RARE)}) are where a model's weakness shows.")
    which = st.radio("Show", ["Recall", "F1"], horizontal=True, label_visibility="collapsed")
    idx = 0 if which == "Recall" else 1
    heat_df = pd.DataFrame([dict(Model=m, Class=c, Value=S.PER_CLASS[m][c][idx]) for m in M for c in S.CLASSES])
    heat = alt.Chart(heat_df).mark_rect(cornerRadius=4).encode(
        x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=-30, title=None, orient="top")),
        y=alt.Y("Model:N", sort=M, title=None),
        color=alt.Color("Value:Q", scale=alt.Scale(domain=[0, 1], range=["#1e293b", "#0ea5e9", "#a7f3d0"]), legend=None),
        tooltip=["Model", "Class", alt.Tooltip("Value:Q", format=".2f", title=which)])
    text = alt.Chart(heat_df).mark_text(fontSize=12, fontWeight=600).encode(
        x=alt.X("Class:N", sort=S.CLASSES), y=alt.Y("Model:N", sort=M), text=alt.Text("Value:Q", format=".2f"),
        color=alt.condition("datum.Value > 0.6", alt.value("#0a0f1c"), alt.value("#e6edf7")))
    st.altair_chart(ui.style(heat + text, 250), width="stretch")

    ui.section("What each gain costs", "Macro-F1 against computational cost on CPU. Up and to the left is better.")
    cost_axis = st.radio("Cost axis", ["Per-record latency (ms)", "Training time (s)", "Model size (MB)"], horizontal=True,
                         label_visibility="collapsed")
    ckey = {"Per-record latency (ms)": "lat", "Training time (s)": "train", "Model size (MB)": "size"}[cost_axis]
    pts = []
    for m in M:
        x = S.COST[m][ckey][0] if ckey == "train" else S.COST[m][ckey]
        pts.append(dict(Model=m, Cost=x, MacroF1=S.MAIN[m]["mf1"][0], SD=S.MAIN[m]["mf1"][1], Family=S.FAMILY[m]))
    pdf = pd.DataFrame(pts)
    pdf["lo"], pdf["hi"] = pdf.MacroF1 - pdf.SD, pdf.MacroF1 + pdf.SD
    b = alt.Chart(pdf)
    xs = alt.Scale(type="log", nice=True)           # one scale object for every layer, so they cannot disagree
    ys = alt.Scale(domain=_domain(list(pdf.lo) + list(pdf.hi), pad=0.04), zero=False)
    sc = (b.mark_rule(color="#64748b").encode(x=alt.X("Cost:Q", scale=xs, title=cost_axis, axis=alt.Axis(grid=False)), y=alt.Y("lo:Q", scale=ys, title="Macro-F1"), y2="hi:Q")
          + b.mark_circle(size=260, opacity=.95).encode(x=alt.X("Cost:Q", scale=xs), y=alt.Y("MacroF1:Q", scale=ys),
                                                     color=alt.Color("Model:N", scale=MODEL_SCALE, legend=None),
                                                     tooltip=["Model", alt.Tooltip("Cost:Q", format=".2f", title=cost_axis),
                                                              alt.Tooltip("MacroF1:Q", format=".3f")])
          + b.mark_text(dx=14, align="left", color="#e6edf7", fontSize=12, fontWeight=600).encode(
              x=alt.X("Cost:Q", scale=xs), y=alt.Y("MacroF1:Q", scale=ys), text="Model:N"))
    st.altair_chart(ui.style(sc, 360), width="stretch")
    cdf = pd.DataFrame([dict(Model=m, **{"Training time (s)": f"{S.COST[m]['train'][0]:.1f} ± {S.COST[m]['train'][1]:.1f}",
                                         "Epochs": str(S.COST[m]["epochs"] or "-"),
                                         f"Batch inference, {META['n_test']:,} flows (s)": f"{S.COST[m]['batch'][0]:.4g}",
                                         "Latency per record, median (p95) ms": f"{S.COST[m]['lat']:.2f} ({S.COST[m]['lat95']:.2f})",
                                         "Model size (MB)": S.COST[m]["size"], "Size counts": S.COST[m]["counts"]}) for m in M])
    st.dataframe(cdf, hide_index=True, width="stretch")
    ui.note("Per-record neural latency includes framework call overhead; GPU time was not measured.")

    if S.UNSEEN:
        _unseen_section()

    with st.expander("All numbers as tables"):
        t1 = pd.DataFrame([dict(Model=m, Accuracy=f"{S.MAIN[m]['acc'][0]:.3f} ± {S.MAIN[m]['acc'][1]:.3f}",
                                **{"Macro-F1": f"{S.MAIN[m]['mf1'][0]:.3f} ± {S.MAIN[m]['mf1'][1]:.3f}",
                                   "Attack detection (%)": f"{S.MAIN[m]['det'][0]:.1f} ± {S.MAIN[m]['det'][1]:.1f}",
                                   "False-positive rate (%)": f"{S.MAIN[m]['fpr'][0]:.1f} ± {S.MAIN[m]['fpr'][1]:.1f}",
                                   "ROC-AUC macro": f"{S.MAIN[m]['auc'][0]:.3f} ± {S.MAIN[m]['auc'][1]:.3f}",
                                   "ROC-AUC binary": f"{S.MAIN[m]['aucb'][0]:.3f} ± {S.MAIN[m]['aucb'][1]:.3f}"}) for m in M])
        st.caption("Test split, mean ± standard deviation over seeds")
        st.dataframe(t1, hide_index=True, width="stretch")
        t2 = pd.DataFrame([dict(Model=m, **{c: f"{S.PER_CLASS[m][c][0]:.2f} / {S.PER_CLASS[m][c][1]:.2f}" for c in S.CLASSES}) for m in M])
        st.caption("Recall / F1 per class")
        st.dataframe(t2, hide_index=True, width="stretch")
        t3 = pd.DataFrame([dict(Model=m, **{"Chosen configuration": S.CONFIG[m][0], "Validation macro-F1": S.CONFIG[m][1],
                                            "Test macro-F1": S.MAIN[m]["mf1"][0]}) for m in M])
        st.caption("Chosen configurations (random search, equal budget for every model)")
        st.dataframe(t3, hide_index=True, width="stretch")
    ui.source(SRC)


# ---------------------------------------------------------------- Ablations
def ablations():
    ui.section("Ablations", "Each ablation changes one factor and keeps the rest fixed. They use one seed and frozen hyperparameters, "
               "so differences smaller than the seed-to-seed spread of the main run should not be interpreted.")
    if len(S.IMBALANCE) < len(M) or len(S.BINARY) < len(M):  # a reduced run: say which models each ablation covers
        st.caption(f"To keep the run short, the binary task covers {', '.join(S.BINARY)}; the imbalance, feature-group and "
                   f"training-size ablations cover {', '.join(S.IMBALANCE)}.")
    t_task, t_imb, t_feat, t_size = st.tabs(["Binary task", "Class imbalance", "Feature groups", "Training size"])

    with t_task:
        if not S.BINARY:
            st.info("The binary-task ablation has not been run.")
        else:
            st.markdown("**Does the ranking hold when the task is only benign vs attack?**")
            met = st.selectbox("Metric", ["Accuracy", "F1 (attack)", "Detection rate (%)", "False-positive rate (%)", "ROC-AUC"], key="bin_metric")
            k = {"Accuracy": "acc", "F1 (attack)": "f1", "Detection rate (%)": "det", "False-positive rate (%)": "fpr", "ROC-AUC": "auc"}[met]
            df = pd.DataFrame([dict(Model=m, Value=S.BINARY[m][k]) for m in M if m in S.BINARY])
            ch = alt.Chart(df).mark_bar(cornerRadiusEnd=5, size=46).encode(
                x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)), y=alt.Y("Value:Q", title=met), color=MODEL_COLOR,
                tooltip=["Model", alt.Tooltip("Value:Q", format=".3f")])
            lab = alt.Chart(df).mark_text(dy=-9, color="#e6edf7", fontWeight=600).encode(
                x=alt.X("Model:N", sort=M), y="Value:Q", text=alt.Text("Value:Q", format=".3g"))
            st.altair_chart(ui.style(ch + lab, 320), width="stretch")

    with t_imb:
        if not S.IMBALANCE:
            st.info("The imbalance ablation has not been run.")
        else:
            st.markdown("**How much of the rare-class performance comes from the correction and not from the model?**")
            r1, r2 = S.RARE[0], S.RARE[1]
            rows = [dict(Model=m, Strategy=s, MacroF1=v[0], R1=v[1], R2=v[2]) for m in M if m in S.IMBALANCE for s, v in S.IMBALANCE[m].items()]
            df = pd.DataFrame(rows)
            order = ["No correction", "Class weights", "SMOTE"]
            ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)), xOffset=alt.XOffset("Strategy:N", sort=order),
                y=alt.Y("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Strategy:N", sort=order, scale=alt.Scale(domain=order, range=["#64748b", "#38bdf8", "#f472b6"]),
                                legend=alt.Legend(title=None)),
                tooltip=["Model", "Strategy", alt.Tooltip("MacroF1:Q", format=".3f"), alt.Tooltip("R1:Q", title=f"{r1} F1", format=".2f"),
                         alt.Tooltip("R2:Q", title=f"{r2} F1", format=".2f")])
            st.altair_chart(ui.style(ch, 340), width="stretch")
            tab = pd.DataFrame([dict(Model=m, **{s: f"{v[0]:.3f}  ({r1} {v[1]:.2f}, {r2} {v[2]:.2f})" for s, v in S.IMBALANCE[m].items()})
                                for m in M if m in S.IMBALANCE])
            st.dataframe(tab, hide_index=True, width="stretch")
            st.caption("Class weights are the main run (seed 0). SMOTE raises the rare classes to the median class size.")

    with t_feat:
        if not S.FEATURES:
            st.info("The feature-group ablation has not been run.")
        else:
            st.markdown("**Which information does each family rely on?** The best classical and the best neural model on validation.")
            mdl = st.radio("Model", list(S.FEATURES), horizontal=True, key="feat_model")
            rows = [dict(Features=f, Inputs=v[0], MacroF1=v[1], Accuracy=v[2]) for f, v in S.FEATURES[mdl].items()]
            df = pd.DataFrame(rows)
            order = list(S.FEATURES[mdl])
            base_f1 = S.FEATURES[mdl]["All features"][1]
            ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Features:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(alt.datum.Features == "All features", alt.value("#38bdf8"), alt.value(S.COLORS[mdl])),
                tooltip=["Features", "Inputs", alt.Tooltip("MacroF1:Q", format=".3f"), alt.Tooltip("Accuracy:Q", format=".3f")])
            ref = alt.Chart(pd.DataFrame({"x": [base_f1]})).mark_rule(color="#e6edf7", strokeDash=[4, 4]).encode(x="x:Q")
            st.altair_chart(ui.style(ch + ref, 330), width="stretch")
            st.caption("Features are grouped as flags and counts, protocols, rates and timing, and size statistics; 'Top 20/40' keep "
                       "the features with the highest mutual information with the class (computed on the training data only).")

    with t_size:
        if not S.SIZE:
            st.info("The training-size ablation has not been run.")
        else:
            st.markdown("**Does deep learning need more data than classical learning to compete?**")
            rows = [dict(Model=m, Share=p, MacroF1=v) for m in M if m in S.SIZE for p, v in zip(S.SIZE_PERCENT, S.SIZE[m])]
            df = pd.DataFrame(rows)
            ln = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=2.5).encode(
                x=alt.X("Share:Q", title="Share of the training sample used (%)", scale=alt.Scale(domain=[5, 105]),
                        axis=alt.Axis(values=S.SIZE_PERCENT)),
                y=alt.Y("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=_domain(list(df.MacroF1)))), color=MODEL_COLOR_LEGEND,
                tooltip=["Model", "Share", alt.Tooltip("MacroF1:Q", format=".3f")])
            st.altair_chart(ui.style(ln, 340), width="stretch")
            st.caption("The neural curves can be non-monotonic: each run uses one seed and early stopping on a small validation set, "
                       "so only the broad pattern should be read.")
    ui.source(SRC)


# ---------------------------------------------------------------- About
def about():
    ui.section("About this project")
    st.markdown(
        f"This app presents a controlled comparison of classical machine learning and deep learning for network intrusion "
        f"detection, run on **{S.DATASET_NAME}**. The study design comes from an **Information Security Lab mini project at MIT "
        f"Manipal**; the results shown here come from running the benchmark on "
        f"{S.DATASET_NAME}. The **Detector** page runs copies of the five models on your own CSV files.")
    st.markdown("**Study design by:** " + ", ".join(S.AUTHORS) + ".")

    ui.section("Method in brief")
    st.markdown(
        f"- **Data.** {S.DATASET_NAME}: {len(S.CLASSES) - 1} attack categories plus benign flows, {META['n_features']} numeric flow "
        f"features. A per-class sample (at most {META.get('sample_cap_per_class') or 'all'} flows per class) with exact duplicates removed: "
        f"{META['n_train'] + META['n_test']:,} flows.\n"
        f"- **Split.** Stratified 80/20 split into training and test; the training part is split 80/20 again into training and "
        f"validation. The test split is used once, after every choice is frozen.\n"
        f"- **Preprocessing.** Min-max scaling fitted on the training partition only; constant columns dropped.\n"
        f"- **Imbalance.** Class weights by default, compared with no correction and SMOTE.\n"
        f"- **Models.** Decision Tree, Random Forest and RBF SVM (fitted on a subsample of at most 30,000 records that keeps every "
        f"record of the rare classes), and a Keras MLP and LSTM. The LSTM sees each flow as one time step.\n"
        f"- **Tuning.** Random search, {META['tuning_configs']} configurations per model, scored by validation macro-F1.\n"
        f"- **Metrics.** Accuracy, per-class precision, recall and F1, macro-F1, false-positive rate, ROC-AUC, confusion matrices, "
        f"bootstrap intervals and McNemar tests with Holm correction. A macro-F1 difference below 0.01 is called a practical tie.\n"
        f"- **Cost.** Training time, batch inference time, per-record latency and model size on CPU, after a warm-up pass.")

    ui.section("Limitations")
    for item in S.LIMITATIONS:
        st.markdown(f"- {item}")

    with st.expander("References"):
        for r in S.REFERENCES:
            st.markdown(f"- {r}")
    ui.note(f"{S.DATASET_NAME} is a research dataset from a laboratory testbed. Nothing in this app is a production intrusion "
            f"detection system.", warn=True)
