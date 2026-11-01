"""The study pages: Overview, Results, Ablations and About. Every number comes from study_data.py (the lab report)."""
import altair as alt
import pandas as pd
import streamlit as st

import study_data as S
import ui

M, COLORS = S.MODELS, S.COLORS
MODEL_SCALE = ui.color_scale(M, COLORS)
MODEL_COLOR = alt.Color("Model:N", scale=MODEL_SCALE, legend=None)  # models are labelled on the axis
MODEL_COLOR_LEGEND = alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(title=None))
SRC = "Source: lab report (original run: 2 CPU cores, scikit-learn 1.9.1, TensorFlow 2.21)."


def _go(page):
    st.session_state["page"] = page


def _try(model):
    """Open the Detector with this model selected."""
    st.session_state["detector_set"] = "The five study models"
    st.session_state["detector_model"] = model
    st.session_state["page"] = "🛡️ Detector"


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
    ui.hero("Information Security Lab · MIT Manipal",
            "Can <em>deep learning</em> beat classical machine learning at spotting network intrusions?",
            "A controlled benchmark of five classifiers on NSL-KDD. Same data, same split, same tuning budget, and "
            "results reported on rare attacks, unseen attacks and computational cost, not on accuracy alone.",
            [("Decision Tree", "c"), ("Random Forest", "c"), ("SVM", "c"), ("MLP", "d"), ("LSTM", "d"),
             ("NSL-KDD", ""), ("5 seeds", "")])
    ui.kpis([("5", "classifiers compared", "3 classical, 2 deep"),
             ("22,544", "test records", "official KDDTest+"),
             ("17", "attack types never seen in training", "3,750 test records"),
             ("0.641", "best macro-F1 (LSTM)", "0.631 for the MLP, 0.510 to 0.612 classical"),
             ("70-79 s", "neural training time on 2 CPU cores", "vs 0.9 to 11.3 s classical")])

    ui.section("Meet the five models",
               "Three classical models and two neural networks, trained and tested under exactly the same conditions. Scores are "
               "from the full KDDTest+ file (mean over five seeds); times are on 2 CPU cores.")
    ui.model_cards(_card_items())
    st.button("🤖 Explore each model in detail", on_click=_go, args=("🤖 Models",))

    ui.section("What the study found")
    ui.cards(S.FINDINGS)

    ui.section("Where to look next")
    c1, c2, c3, c4 = st.columns(4)
    c1.button("📊 Full results", on_click=_go, args=("📊 Results",), width="stretch")
    c2.button("🧪 Ablations", on_click=_go, args=("🧪 Ablations",), width="stretch")
    c3.button("🛡️ Run the five models", on_click=_go, args=("🛡️ Detector",), width="stretch")
    c4.button("🔬 Pick a record", on_click=_go, args=("🔬 Bench",), width="stretch")

    ui.section("Why the test set is a hard one",
               "The official test file is deliberately different from the training file: R2L attacks grow from 0.79% of "
               "the training records to 12.80% of the test records, and 17 attack types appear only in the test file.")
    rows = []
    for cls, (tr, te) in S.DISTRIBUTION.items():
        rows += [dict(Class=cls, Set="KDDTrain+", Share=tr / sum(v[0] for v in S.DISTRIBUTION.values()) * 100, Records=tr),
                 dict(Class=cls, Set="KDDTest+", Share=te / sum(v[1] for v in S.DISTRIBUTION.values()) * 100, Records=te)]
    df = pd.DataFrame(rows)
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=0, title=None)),
                     y=alt.Y("Share:Q", title="Share of records (%)"),
                     xOffset="Set:N",
                     color=alt.Color("Set:N", scale=alt.Scale(domain=["KDDTrain+", "KDDTest+"], range=["#64748b", "#38bdf8"]),
                                     legend=alt.Legend(title=None)),
                     tooltip=["Class", "Set", alt.Tooltip("Share:Q", format=".2f"), alt.Tooltip("Records:Q", format=",")]))
    st.altair_chart(ui.style(chart, 300), width="stretch")
    ui.note("U2R has 52 training records against 67,343 Normal ones (about 1 to 1,300), and only 67 test records, so its "
            "scores are unstable.", warn=True)

    ui.section("How the benchmark works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#243044", fontcolor="#e6edf7", fillcolor="#111827", penwidth=1.2];
      edge [color="#64748b", arrowsize=0.7];
      raw [label="NSL-KDD\\nKDDTrain+ / KDDTest+"]; clean [label="Check, map attack\\ntypes to 5 classes"];
      split [label="Stratified 80/20\\ntrain / validation"]; prep [label="One-hot, log, min-max\\nfit on train only"];
      cl [label="Decision Tree\\nRandom Forest\\nSVM", color="#818cf8"]; dl [label="MLP\\nLSTM", color="#f97316"];
      tune [label="20 random configs\\neach, validation macro-F1"]; final [label="Refit on full train\\n5 seeds"];
      test [label="Evaluate once on KDDTest+\\n+ KDDTest-21 + unseen types", color="#38bdf8"];
      out [label="Metrics, cost, bootstrap,\\nMcNemar + Holm", color="#38bdf8"];
      raw -> clean -> split -> prep; prep -> cl; prep -> dl; cl -> tune; dl -> tune; tune -> final -> test -> out; }
    """, width="stretch")

    ui.section("The four research questions")
    for q, a in S.RESEARCH_QUESTIONS:
        with st.expander(q):
            st.write(a)
    ui.source(SRC)


# ---------------------------------------------------------------- Models
def models():
    ui.section("The five models",
               "Every model gets the same data, the same split, the same metric code and the same tuning budget (20 random "
               "configurations, scored by validation macro-F1). Scores are from the full KDDTest+ file, mean over five seeds.")
    ui.model_cards(_card_items())

    ui.section("Look closer")
    pick = st.radio("Model", M, horizontal=True, key="model_pick", label_visibility="collapsed")
    note, fam = S.MODEL_NOTES[pick], S.FAMILY[pick]
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"### {pick}  ·  {fam}")
        st.write(note["how"])
        st.caption(f"Frozen configuration: {S.CONFIG[pick][0]}. Validation macro-F1 {S.CONFIG[pick][1]:.3f}, "
                   f"against {S.MAIN[pick]['mf1'][0]:.3f} on the test file.")
        st.markdown("**What stood out in the report**")
        for s in note["stood_out"]:
            st.markdown(f"- {s}")
        st.button(f"🛡️ Run {pick} on your own records", on_click=_try, args=(pick,))
    with right:
        rows = [dict(Class=c, Measure=k, Value=S.PER_CLASS[pick][c][i]) for c in S.CLASSES for i, k in enumerate(["Recall", "F1"])]
        ch = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=0, title=None)), xOffset="Measure:N",
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Recall", "F1"], range=[COLORS[pick], "#64748b"]),
                            legend=alt.Legend(title=None)),
            tooltip=["Class", "Measure", alt.Tooltip("Value:Q", format=".2f")])
        st.altair_chart(ui.style(ch, 260), width="stretch")
        st.caption(f"{pick}: recall and F1 per class on KDDTest+ (mean over five seeds).")
    s, c = S.MAIN[pick], S.COST[pick]
    h = S.HARD[pick]
    ui.kpis([(f"{s['acc'][0]:.3f}", "accuracy", f"± {s['acc'][1]:.3f}"), (f"{s['mf1'][0]:.3f}", "macro-F1", f"± {s['mf1'][1]:.3f}"),
             (f"{s['det'][0]:.1f}%", "attacks flagged", f"{s['fpr'][0]:.1f}% false alarms on Normal"),
             (f"{h['u_det'][0]:.1f}%", "unseen attack types flagged", f"{h['u_acc'][0]:.1f}% given the right family"),
             (f"{c['lat']:.2f} ms", "per record", f"{c['train'][0]:.1f} s to train · {c['size']:g} MB")])

    ui.section("All five side by side")
    rows = []
    for m in M:
        r = S.MAIN[m]
        rows.append({"Model": m, "Family": S.FAMILY[m], "Macro-F1": r["mf1"][0], "Accuracy": r["acc"][0], "Attacks flagged (%)": r["det"][0],
                     "False alarms (%)": r["fpr"][0], "R2L recall": S.PER_CLASS[m]["R2L"][0], "U2R recall": S.PER_CLASS[m]["U2R"][0],
                     "Unseen flagged (%)": S.HARD[m]["u_det"][0], "Training (s)": S.COST[m]["train"][0], "Per record (ms)": S.COST[m]["lat"],
                     "Size (MB)": S.COST[m]["size"]})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Macro-F1": st.column_config.ProgressColumn(format="%.3f", min_value=0.4, max_value=0.7),
        "Accuracy": st.column_config.ProgressColumn(format="%.3f", min_value=0.6, max_value=0.85)})
    st.caption("The ranking depends on what you value: the LSTM and MLP lead on macro-F1 and accuracy, the Random Forest on ROC-AUC and "
               "false alarms, the Decision Tree and SVM on speed and size.")
    try:
        import study_models
        if study_models.available():
            meas = study_models.meta()
            with st.expander("The copies of these models running in this app"):
                st.markdown("The Detector page runs copies of the five models that were retrained for this app on the full KDDTrain+, "
                            "with the report's settings, and tested once on KDDTest+. They match the report closely but are not the same "
                            "runs (one seed, and a different deep-learning backend for the MLP and LSTM), so small differences are expected.")
                st.dataframe(pd.DataFrame([{"Model": m,
                                            "Accuracy (this app)": meas[study_models.KEY_OF[m]]["test"]["acc"],
                                            "Accuracy (report)": S.MAIN[m]["acc"][0],
                                            "Macro-F1 (this app)": meas[study_models.KEY_OF[m]]["test"]["macro_f1"],
                                            "Macro-F1 (report)": S.MAIN[m]["mf1"][0]} for m in M]).round(3),
                             hide_index=True, width="stretch")
    except Exception:  # the page must not fail if the exported models are absent
        pass
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


def results():
    ui.section("Results", "Every number on this page is from the lab report. Bars show the mean over five seeds; whiskers are one "
               "standard deviation.")
    view = st.radio("Test view", ["KDDTest+ (all 22,544 records)", "KDDTest-21 (harder: 11,850 records)"], horizontal=True,
                    label_visibility="collapsed")
    full = view.startswith("KDDTest+")
    if full:
        metrics = {"Macro-F1 (headline)": ("mf1", "Macro-F1", ".3f"), "Accuracy": ("acc", "Accuracy", ".3f"),
                   "Attack detection rate (%)": ("det", "Attack detection rate (%)", ".1f"),
                   "False-positive rate on Normal (%)": ("fpr", "False-positive rate (%)", ".1f"),
                   "ROC-AUC, macro one-vs-rest": ("auc", "ROC-AUC (macro)", ".3f"),
                   "ROC-AUC, Normal vs Attack": ("aucb", "ROC-AUC (binary)", ".3f")}
        src = S.MAIN
    else:
        metrics = {"Macro-F1 (headline)": ("t21_mf1", "Macro-F1", ".3f"), "Accuracy": ("t21_acc", "Accuracy", ".3f"),
                   "Attack detection rate (%)": ("t21_det", "Attack detection rate (%)", ".1f")}
        src = S.HARD
    pick = st.selectbox("Metric", list(metrics))
    key, title, fmt = metrics[pick]
    df = pd.DataFrame([dict(Model=m, Value=src[m][key][0], SD=src[m][key][1], Family=S.FAMILY[m]) for m in M])
    st.altair_chart(ui.style(_bar_with_error(df, "Value", "SD", title, fmt), 340), width="stretch")
    if key in ("mf1", "t21_mf1"):
        ui.note("Macro-F1 gives each of the five classes equal weight, so it is the headline score: accuracy is dominated by "
                "Normal and DoS, which make up 76% of the test file.")
    if key == "fpr":
        ui.note("The Random Forest has the lowest false-positive rate (2.8%) but also the lowest detection rate (58.9%): a "
                "stricter alert threshold, not a better detector.", warn=True)

    ui.section("How sure can we be? 95% bootstrap intervals for macro-F1",
               "1,000 resamples of the test records for the seed-0 run (the dot). The models fall into three groups; the order "
               "inside the top and bottom groups is not settled. Seed 0 can sit above the five-seed mean shown in the bar chart "
               "(the LSTM scores 0.670 on seed 0 and 0.641 on average).")
    ci = pd.DataFrame([dict(Model=m, Low=S.CI[m][0], High=S.CI[m][1], Mid=S.SEED0_MF1[m]) for m in M])
    base = alt.Chart(ci)
    chart = (base.mark_rule(strokeWidth=7, strokeCap="round").encode(
        y=alt.Y("Model:N", sort=M, title=None), x=alt.X("Low:Q", title="Macro-F1", scale=alt.Scale(domain=[0.45, 0.72])),
        x2="High:Q", color=MODEL_COLOR, tooltip=["Model", alt.Tooltip("Low:Q", format=".3f"), alt.Tooltip("High:Q", format=".3f")])
             + base.mark_point(color="#0a0f1c", filled=True, size=70, stroke="#e6edf7", strokeWidth=1.5)
             .encode(y=alt.Y("Model:N", sort=M), x="Mid:Q"))
    st.altair_chart(ui.style(chart, 230), width="stretch")

    ui.section("Where the models differ: the rare classes",
               "Normal, DoS and Probe are handled well by every model. R2L and U2R are where the models separate, and no model "
               "detects R2L well.")
    which = st.radio("Show", ["Recall", "F1"], horizontal=True, label_visibility="collapsed")
    idx = 0 if which == "Recall" else 1
    rows = [dict(Model=m, Class=c, Value=S.PER_CLASS[m][c][idx]) for m in M for c in S.CLASSES]
    heat_df = pd.DataFrame(rows)
    heat = alt.Chart(heat_df).mark_rect(cornerRadius=4).encode(
        x=alt.X("Class:N", sort=S.CLASSES, axis=alt.Axis(labelAngle=0, title=None, orient="top")),
        y=alt.Y("Model:N", sort=M, title=None),
        color=alt.Color("Value:Q", scale=alt.Scale(domain=[0, 1], range=["#1e293b", "#0ea5e9", "#a7f3d0"]), legend=None),
        tooltip=["Model", "Class", alt.Tooltip("Value:Q", format=".2f", title=which)])
    text = alt.Chart(heat_df).mark_text(fontSize=13, fontWeight=600).encode(
        x=alt.X("Class:N", sort=S.CLASSES), y=alt.Y("Model:N", sort=M), text=alt.Text("Value:Q", format=".2f"),
        color=alt.condition("datum.Value > 0.6", alt.value("#0a0f1c"), alt.value("#e6edf7")))
    st.altair_chart(ui.style(heat + text, 250), width="stretch")
    ui.note("Most missed attacks are labelled Normal: the Random Forest labels 97% of R2L and 82% of U2R records as Normal, the "
            "two neural networks about 71% to 72% and 43% to 45%. R2L is also 12.8% of the test file but 0.79% of training, "
            "so low R2L recall reflects distribution shift as well as model weakness.", warn=True)

    ui.section("What each gain costs",
               "Macro-F1 against computational cost on 2 CPU cores. Up and to the left is better.")
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
    ys = alt.Scale(domain=[0.48, 0.68], zero=False)
    sc = (b.mark_rule(color="#64748b").encode(x=alt.X("Cost:Q", scale=xs, title=cost_axis, axis=alt.Axis(grid=False)), y=alt.Y("lo:Q", scale=ys, title="Macro-F1"), y2="hi:Q")
          + b.mark_circle(size=260, opacity=.95).encode(x=alt.X("Cost:Q", scale=xs), y=alt.Y("MacroF1:Q", scale=ys),
                                                     color=alt.Color("Model:N", scale=MODEL_SCALE, legend=None),
                                                     tooltip=["Model", alt.Tooltip("Cost:Q", format=".2f", title=cost_axis),
                                                              alt.Tooltip("MacroF1:Q", format=".3f")])
          + b.mark_text(dx=14, align="left", color="#e6edf7", fontSize=12, fontWeight=600).encode(
              x=alt.X("Cost:Q", scale=xs), y=alt.Y("MacroF1:Q", scale=ys), text="Model:N"))
    st.altair_chart(ui.style(sc, 360), width="stretch")
    cdf = pd.DataFrame([dict(Model=m, **{"Training time (s)": f"{S.COST[m]['train'][0]:.1f} ± {S.COST[m]['train'][1]:.1f}",
                                         "Epochs": S.COST[m]["epochs"] or "-",
                                         "Batch inference, 22,544 records (s)": f"{S.COST[m]['batch'][0]:.4g}",
                                         "Latency per record, median (p95) ms": f"{S.COST[m]['lat']:.2f} ({S.COST[m]['lat95']:.2f})",
                                         "Model size (MB)": S.COST[m]["size"], "Size counts": S.COST[m]["counts"]}) for m in M])
    st.dataframe(cdf, hide_index=True, width="stretch")
    ui.note("The SVM gets within 0.03 of the best macro-F1 at 0.34 ms per record and 6.8 s of training. The Random Forest is the "
            "largest (38 MB) and the slowest per record (17.9 ms) while ranking lowest on macro-F1. Per-record neural latency "
            "includes framework call overhead; GPU time was not measured.")

    ui.section("Attack types the models have never seen",
               "3,750 test records of 17 attack types that do not occur in training: 1,719 DoS, 1,315 Probe, 686 R2L, 30 U2R.")
    udf = pd.DataFrame([dict(Model=m, Detection=S.HARD[m]["u_det"][0], SD=S.HARD[m]["u_det"][1],
                             CorrectFamily=S.HARD[m]["u_acc"][0]) for m in M])
    cmp_df = pd.melt(udf, id_vars=["Model"], value_vars=["Detection", "CorrectFamily"], var_name="Measure", value_name="Percent")
    cmp_df["Measure"] = cmp_df["Measure"].map({"Detection": "Flagged as any attack", "CorrectFamily": "Given the correct family"})
    uc = (alt.Chart(cmp_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)), xOffset="Measure:N",
        y=alt.Y("Percent:Q", title="Share of the 3,750 unseen-type records (%)"),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=["Flagged as any attack", "Given the correct family"],
                                                     range=["#38bdf8", "#a78bfa"]), legend=alt.Legend(title=None)),
        tooltip=["Model", "Measure", alt.Tooltip("Percent:Q", format=".1f")]))
    st.altair_chart(ui.style(uc, 320), width="stretch")
    rec = pd.DataFrame([dict(Model=m, Family=f, Recall=S.HARD[m]["u_rec"][f]) for m in M for f in ["DoS", "Probe", "R2L", "U2R"]])
    h = alt.Chart(rec).mark_rect(cornerRadius=4).encode(
        x=alt.X("Family:N", sort=["DoS", "Probe", "R2L", "U2R"], axis=alt.Axis(labelAngle=0, title="Recall on unseen-type records", orient="top")),
        y=alt.Y("Model:N", sort=M, title=None),
        color=alt.Color("Recall:Q", scale=alt.Scale(domain=[0, 0.6], range=["#1e293b", "#0ea5e9", "#a7f3d0"]), legend=None))
    t = alt.Chart(rec).mark_text(fontSize=13, fontWeight=600).encode(
        x=alt.X("Family:N", sort=["DoS", "Probe", "R2L", "U2R"]), y=alt.Y("Model:N", sort=M), text=alt.Text("Recall:Q", format=".2f"),
        color=alt.condition("datum.Recall > 0.35", alt.value("#0a0f1c"), alt.value("#e6edf7")))
    st.altair_chart(ui.style(h + t, 230), width="stretch")
    ui.note("These results do not support the idea that any of the five models detects zero-day-style attacks reliably on this "
            "dataset. The Random Forest, strongest on validation, transfers worst.", warn=True)

    with st.expander("All numbers as tables"):
        t1 = pd.DataFrame([dict(Model=m, Accuracy=f"{S.MAIN[m]['acc'][0]:.3f} ± {S.MAIN[m]['acc'][1]:.3f}",
                                **{"Macro-F1": f"{S.MAIN[m]['mf1'][0]:.3f} ± {S.MAIN[m]['mf1'][1]:.3f}",
                                   "Attack detection (%)": f"{S.MAIN[m]['det'][0]:.1f} ± {S.MAIN[m]['det'][1]:.1f}",
                                   "False-positive rate (%)": f"{S.MAIN[m]['fpr'][0]:.1f} ± {S.MAIN[m]['fpr'][1]:.1f}",
                                   "ROC-AUC macro": f"{S.MAIN[m]['auc'][0]:.3f} ± {S.MAIN[m]['auc'][1]:.3f}",
                                   "ROC-AUC binary": f"{S.MAIN[m]['aucb'][0]:.3f} ± {S.MAIN[m]['aucb'][1]:.3f}"}) for m in M])
        st.caption("KDDTest+ (22,544 records), mean ± standard deviation over five seeds")
        st.dataframe(t1, hide_index=True, width="stretch")
        t2 = pd.DataFrame([dict(Model=m, **{c: f"{S.PER_CLASS[m][c][0]:.2f} / {S.PER_CLASS[m][c][1]:.2f}" for c in S.CLASSES}) for m in M])
        st.caption("Recall / F1 per class")
        st.dataframe(t2, hide_index=True, width="stretch")
        t3 = pd.DataFrame([dict(Model=m, **{"Frozen configuration": S.CONFIG[m][0], "Validation macro-F1": S.CONFIG[m][1],
                                            "Test macro-F1": S.MAIN[m]["mf1"][0]}) for m in M])
        st.caption("Frozen configurations (random search, 20 each). Validation did not predict test performance.")
        st.dataframe(t3, hide_index=True, width="stretch")
    ui.source(SRC)


# ---------------------------------------------------------------- Ablations
def ablations():
    ui.section("Ablations", "Each ablation changes one factor and keeps the rest fixed. They use one seed and frozen hyperparameters, "
               "so differences smaller than the seed-to-seed spread (macro-F1 std dev 0.002 to 0.004 for the tree models and SVM, "
               "0.014 for the MLP, 0.027 for the LSTM) should not be interpreted.")
    t_task, t_imb, t_feat, t_size = st.tabs(["Binary task", "Class imbalance", "Feature groups", "Training size"])

    with t_task:
        st.markdown("**Does the ranking hold when the task is only Normal vs Attack?** The SVM catches up with the neural networks, "
                    "so the neural gain in the five-class task comes mainly from separating the attack families.")
        met = st.selectbox("Metric", ["Accuracy", "F1 (attack)", "Detection rate (%)", "False-positive rate (%)", "ROC-AUC",
                                      "Unseen-type detection (%)"], key="bin_metric")
        k = {"Accuracy": "acc", "F1 (attack)": "f1", "Detection rate (%)": "det", "False-positive rate (%)": "fpr",
             "ROC-AUC": "auc", "Unseen-type detection (%)": "unseen"}[met]
        df = pd.DataFrame([dict(Model=m, Value=S.BINARY[m][k]) for m in M])
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=5, size=46).encode(
            x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)), y=alt.Y("Value:Q", title=met), color=MODEL_COLOR,
            tooltip=["Model", alt.Tooltip("Value:Q", format=".3f")])
        lab = alt.Chart(df).mark_text(dy=-9, color="#e6edf7", fontWeight=600).encode(
            x=alt.X("Model:N", sort=M), y="Value:Q", text=alt.Text("Value:Q", format=".3g"))
        st.altair_chart(ui.style(ch + lab, 320), width="stretch")
        st.caption("The Random Forest again has the lowest accuracy (0.777) and the highest AUC (0.961). The SVM flags the largest "
                   "share of unseen attack types here (58.8%, against 30.6% in the five-class task).")

    with t_imb:
        st.markdown("**How much of the rare-class performance comes from the correction and not from the model?** The order neural "
                    "networks, SVM, then trees holds under all three strategies.")
        rows = [dict(Model=m, Strategy=s, MacroF1=v[0], R2L=v[1], U2R=v[2]) for m in M for s, v in S.IMBALANCE[m].items()]
        df = pd.DataFrame(rows)
        order = ["No correction", "Class weights", "SMOTE"]
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Model:N", sort=M, axis=alt.Axis(labelAngle=0, title=None)), xOffset=alt.XOffset("Strategy:N", sort=order),
            y=alt.Y("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 0.8])),
            color=alt.Color("Strategy:N", sort=order, scale=alt.Scale(domain=order, range=["#64748b", "#38bdf8", "#f472b6"]),
                            legend=alt.Legend(title=None)),
            tooltip=["Model", "Strategy", alt.Tooltip("MacroF1:Q", format=".3f"), alt.Tooltip("R2L:Q", title="R2L F1", format=".2f"),
                     alt.Tooltip("U2R:Q", title="U2R F1", format=".2f")])
        st.altair_chart(ui.style(ch, 340), width="stretch")
        st.caption("Class weights did not help the MLP (0.674 without, 0.649 with): its U2R F1 fell from 0.55 to 0.34, probably because "
                   "a weight near 485 for U2R trades precision for recall. SMOTE gave the Random Forest the one clear gain beyond the "
                   "seed spread (0.498 to 0.538).")
        tab = pd.DataFrame([dict(Model=m, **{s: f"{v[0]:.3f}  (R2L {v[1]:.2f}, U2R {v[2]:.2f})" for s, v in S.IMBALANCE[m].items()}) for m in M])
        st.dataframe(tab, hide_index=True, width="stretch")

    with t_feat:
        st.markdown("**Which information does each family rely on?** The Random Forest and the LSTM were the best classical and best "
                    "neural models on validation.")
        mdl = st.radio("Model", ["LSTM", "Random Forest"], horizontal=True, key="feat_model")
        rows = [dict(Features=f, Inputs=v[0], MacroF1=v[1], Accuracy=v[2], Unseen=v[3]) for f, v in S.FEATURES[mdl].items()]
        df = pd.DataFrame(rows)
        order = list(S.FEATURES[mdl])
        base_f1 = S.FEATURES[mdl]["All features"][1]
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Features:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 0.8])),
            color=alt.condition(alt.datum.Features == "All features", alt.value("#38bdf8"), alt.value(S.COLORS[mdl])),
            tooltip=["Features", "Inputs", alt.Tooltip("MacroF1:Q", format=".3f"), alt.Tooltip("Accuracy:Q", format=".3f"),
                     alt.Tooltip("Unseen:Q", title="Unseen-type detection (%)", format=".1f")])
        ref = alt.Chart(pd.DataFrame({"x": [base_f1]})).mark_rule(color="#e6edf7", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(ui.style(ch + ref, 330), width="stretch")
        ch2 = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color="#a78bfa").encode(
            y=alt.Y("Features:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("Unseen:Q", title="Unseen-attack-type detection rate (%)"),
            tooltip=["Features", alt.Tooltip("Unseen:Q", format=".1f")])
        st.altair_chart(ui.style(ch2, 300), width="stretch")
        ui.note("The basic-connection group matters most. The expectation that content features matter most for R2L and U2R was "
                "not supported. Removing the host-based traffic features raised unseen-type detection for both models (Random Forest "
                "18.1% to 45.9%, LSTM 44.9% to 72.3%); one possible explanation is that these features tie the models to the "
                "training traffic, but the experiment does not show this. Univariate mutual-information selection did not help.")

    with t_size:
        st.markdown("**Does deep learning need more data than classical learning to compete?** No: the neural networks already beat "
                    "every tree model with 10% of the training file.")
        rows = [dict(Model=m, Share=p, MacroF1=v) for m in M for p, v in zip(S.SIZE_PERCENT, S.SIZE[m])]
        df = pd.DataFrame(rows)
        ln = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=2.5).encode(
            x=alt.X("Share:Q", title="Share of KDDTrain+ used for training (%)", scale=alt.Scale(domain=[5, 105]),
                    axis=alt.Axis(values=S.SIZE_PERCENT)),
            y=alt.Y("MacroF1:Q", title="Macro-F1", scale=alt.Scale(domain=[0.48, 0.7])), color=MODEL_COLOR_LEGEND,
            tooltip=["Model", "Share", alt.Tooltip("MacroF1:Q", format=".3f")])
        st.altair_chart(ui.style(ln, 340), width="stretch")
        st.caption("Training sizes: 12,597, 31,493, 62,986 and 125,973 records. The SVM climbs from 0.535 to 0.617 at half the data "
                   "and then stays flat; the Decision Tree and Random Forest do not improve with more data. The neural curves are "
                   "not monotonic, which fits the seed spread of a single run, so only the broad pattern should be read.")
    ui.source(SRC)


# ---------------------------------------------------------------- About
def about():
    ui.section("About this project")
    st.markdown(
        "This app presents the results of an **Information Security Lab mini project at MIT Manipal**: a controlled comparison of "
        "classical machine learning and deep learning for network intrusion detection on the NSL-KDD dataset. The study is written up "
        "as an IEEE-format report; the **Models**, **Results** and **Ablations** pages reproduce its numbers. The **Detector** page runs "
        "copies of the five models on your own CSV files (retrained for this app with the report's settings, so close to but not "
        "identical to the report's runs), and the **Bench** page lets you pick a test record and watch five classifiers label it. "
        "An earlier study in this repository, with four other models (Logistic Regression, Random Forest, XGBoost and an MLP), is "
        "still available in the Detector under \"Earlier four models\".")
    st.markdown("**Authors:** " + ", ".join(S.AUTHORS) + ".")

    ui.section("Method in brief")
    st.markdown(
        "- **Data.** NSL-KDD `KDDTrain+` (125,973 records) and `KDDTest+` (22,544 records), used unchanged. 41 features, five classes "
        "(Normal, DoS, Probe, R2L, U2R). The test file has 37 attack types, 17 of which never occur in training.\n"
        "- **Preprocessing.** Drop the constant `num_outbound_cmds`, one-hot encode `protocol_type`, `service` and `flag`, apply "
        "`log(1+x)` to `duration`, `src_bytes` and `dst_bytes`, min-max scale. 121 inputs. Every fitted step uses the training "
        "partition only.\n"
        "- **Split.** Stratified 80/20 split of `KDDTrain+` into training and validation; `KDDTest+` is used once, after every choice "
        "is frozen.\n"
        "- **Imbalance.** Class weights by default (about 0.37 for Normal, about 485 for U2R), compared with no correction and SMOTE.\n"
        "- **Models.** Decision Tree, Random Forest and RBF SVM (fitted on a 30,000-record subsample that keeps every R2L and U2R "
        "record), and a Keras MLP and LSTM. The LSTM sees each record as one time step.\n"
        "- **Tuning.** Random search, 20 configurations per model, scored by validation macro-F1.\n"
        "- **Metrics.** Accuracy, per-class precision, recall and F1, macro-F1, false-positive rate, ROC-AUC, confusion matrices, "
        "bootstrap intervals and McNemar tests with Holm correction. A macro-F1 difference below 0.01 was declared a practical tie "
        "in advance.\n"
        "- **Cost.** Training time, batch inference time, per-record latency and model size, measured on the CPU after a warm-up pass.")

    ui.section("Limitations")
    for item in S.LIMITATIONS:
        st.markdown(f"- {item}")

    ui.section("Future work")
    st.markdown("Hold out attack types in the validation split so that model selection rewards generalisation, tune decision thresholds "
                "for the rare classes, repeat the ablations over several seeds, measure GPU training time, and replicate the "
                "benchmark on CICIDS2017.")

    with st.expander("References"):
        for r in S.REFERENCES:
            st.markdown(f"- {r}")
    ui.note("NSL-KDD is a research dataset of 1998-era traffic features. Nothing in this app is a production intrusion detection "
            "system.", warn=True)
