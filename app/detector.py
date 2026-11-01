"""The Detector page: upload a CSV of connection records and classify them with the models trained in this repo,
and the Bench page: an embedded standalone page that compares five classifiers on one chosen test record."""
from pathlib import Path

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import study_data as S  # noqa: E402
import study_models  # noqa: E402
import ui  # noqa: E402
from ids import config as C  # noqa: E402
from ids.predict import Detector, read_traffic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GRID = "#38bdf8", "#f97316", "#243044"
SAMPLE = ROOT / "sample_data" / "sample_traffic.csv"
BENCH = ROOT / "app" / "bench.html"


@st.cache_resource
def get_detector():
    return Detector()


def _read_source(label="Upload a CSV of connection records"):
    """File upload or the bundled sample; returns a DataFrame with the 41 attributes (+ `attack` if labelled)."""
    up = st.file_uploader(label, type=["csv", "txt"])
    use_sample = st.checkbox("…or use the bundled sample (500 KDDTest+ records)", value=up is None)
    source = up if up is not None else (SAMPLE if use_sample and SAMPLE.exists() else None)
    if source is None:
        st.info("Upload a file to begin. Expected: the 41 NSL-KDD feature columns (header optional); a label column is optional.")
        st.stop()
    try:
        return read_traffic(source)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()


@st.cache_resource(show_spinner="Loading the model…")
def _study_model(key):
    return study_models.load_model(key)


@st.cache_resource
def _study_pre():
    return study_models.load_preprocessor()


def _truth(df):
    """True family of every record when a label column was supplied, else None."""
    if "attack" not in df.columns:
        return None
    return df["attack"].map(C.ATTACK_FAMILY).replace({"normal": "Normal"})


def detector():
    ui.section("Detector",
               "Upload a CSV of NSL-KDD connection records and classify every record. The default is the five models of the study; "
               "the earlier four models trained in this repository are still available.")
    have_study = study_models.available()
    sets = ["The five study models", "Earlier four models"] if have_study else ["Earlier four models"]
    which = st.radio("Model set", sets, horizontal=True, key="detector_set", label_visibility="collapsed")
    if which == "The five study models":
        _five_models()
    else:
        _earlier_models()


def _five_models():
    meas = study_models.meta()
    c1, c2 = st.columns([1.3, 1])
    st.session_state.setdefault("detector_model", "LSTM")  # the model with the best macro-F1; the Models page can preselect another
    name = c1.radio("Model", S.MODELS, horizontal=True, key="detector_model")
    key = study_models.KEY_OF[name]
    c2.markdown(f"<div style='padding-top:1.9rem;color:#94a3b8;font-size:.85rem'>{S.FAMILY[name]} · report macro-F1 "
                f"<b style='color:#e6edf7'>{S.MAIN[name]['mf1'][0]:.3f}</b> · {S.COST[name]['lat']:.2f} ms per record</div>",
                unsafe_allow_html=True)
    df = _read_source()
    model, pre = _study_model(key), _study_pre()
    res = study_models.classify(key, model, pre, df)
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    n = len(out)
    flagged = int(out["alert"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Records", f"{n:,}")
    k2.metric("Flagged as attack", f"{flagged:,}  ({flagged / n:.1%})")
    k3.metric("Model", name)

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted classes")
        counts = out["prediction"].value_counts().reindex(study_models.CLASSES).fillna(0).reset_index()
        counts.columns = ["Class", "Records"]
        ch = alt.Chart(counts).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("Class:N", sort=study_models.CLASSES, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Records:Q", title=None),
            color=alt.condition(alt.datum.Class == "Normal", alt.value(BLUE), alt.value(ORANGE)),
            tooltip=["Class", alt.Tooltip("Records:Q", format=",")])
        txt = alt.Chart(counts).mark_text(dy=-8, color="#e6edf7", fontWeight=600).encode(
            x=alt.X("Class:N", sort=study_models.CLASSES), y="Records:Q", text=alt.Text("Records:Q", format=","))
        st.altair_chart(ui.style(ch + txt, 280), width="stretch")
    truth = _truth(df)
    with right:
        if truth is not None:
            st.subheader("Check against your labels")
            known = truth.notna()
            acc = float((truth[known].values == out["prediction"][known].values).mean())
            st.metric("Accuracy on this file", f"{acc:.1%}")
            st.dataframe(pd.crosstab(truth[known].rename("actual"), out["prediction"][known].rename("predicted")))
            if (~known).any():
                st.caption("Some labels were not recognised NSL-KDD attack names and were ignored.")
        else:
            st.subheader("No label column")
            st.caption("Add a `label` column (for example `neptune`, `normal`) to see accuracy and a confusion table.")

    st.subheader("Records raising an alert")
    cols = ["prediction", "confidence", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"] + (
        ["attack"] if "attack" in out.columns else [])
    top = out[out["alert"]].sort_values("confidence", ascending=False, na_position="last")
    st.dataframe(top[cols].head(200).round(3), hide_index=True, width="stretch", height=300)
    if key == "SVM":
        st.caption("The SVM outputs decision scores, not probabilities, so it has no confidence value.")
    st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    ui.section("Run all five on this file", "Same records, five different verdicts. Where the models disagree is where the interesting "
               "cases are.")
    if st.toggle("Compare all five models", key="compare_all"):
        rows, verdicts = [], {}
        for m in S.MODELS:
            k = study_models.KEY_OF[m]
            r = study_models.classify(k, _study_model(k), pre, df)
            verdicts[m] = r["prediction"].values
            row = {"Model": m, "Family": S.FAMILY[m], "Flagged as attack": int(r["alert"].sum()), "Share (%)": round(r["alert"].mean() * 100, 1)}
            if truth is not None:
                row["Accuracy (%)"] = round(float((truth[truth.notna()].values == r["prediction"][truth.notna()].values).mean()) * 100, 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        agree = pd.DataFrame(verdicts)
        same = (agree.nunique(axis=1) == 1).mean()
        st.caption(f"All five models give the same class for {same:.1%} of the records.")
        disagree = pd.concat([df.reset_index(drop=True)[["protocol_type", "service"] + (["attack"] if "attack" in df.columns else [])],
                              agree], axis=1)[agree.nunique(axis=1) > 1]
        if len(disagree):
            with st.expander(f"The {len(disagree):,} records the models disagree on"):
                st.dataframe(disagree.head(200), hide_index=True, width="stretch", height=300)
    with st.expander("About these models"):
        st.markdown(
            "These are the study's five models retrained for this app on the full KDDTrain+ with the report's settings (one seed; "
            "the MLP and LSTM run on a different deep-learning backend), so their scores are close to, not identical to, the "
            "report's. Each was tested once on KDDTest+ when it was trained here:\n\n" +
            "\n".join(f"- **{m}**: accuracy {meas[study_models.KEY_OF[m]]['test']['acc']:.3f}, macro-F1 "
                      f"{meas[study_models.KEY_OF[m]]['test']['macro_f1']:.3f} (report: {S.MAIN[m]['acc'][0]:.3f} and "
                      f"{S.MAIN[m]['mf1'][0]:.3f})" for m in S.MODELS) +
            "\n\nThe classifiers see pre-extracted connection features, mostly miss R2L and U2R attacks and attack types never seen "
            "in training, and are **not** a production intrusion detection system.")


def _earlier_models():
    st.caption("Four other models trained in this repository's first study: Logistic Regression, Random Forest, XGBoost and a PyTorch MLP.")
    try:
        det = get_detector()
    except FileNotFoundError:
        st.error("No trained models found. Run `python -m ids.train` first.")
        st.stop()

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    task_label = c1.radio("Task", ["Normal vs attack", "Attack family (5 classes)"])
    task = "binary" if task_label.startswith("Normal") else "multiclass"
    model = c2.selectbox("Model", det.available(task), index=min(2, len(det.available(task)) - 1))
    thr = None
    if task == "binary":
        thr = c3.slider("Alert threshold on P(attack)", 0.05, 0.95, 0.50, 0.05,
                        help="Lower = catch more attacks but raise more false alarms.")

    up = st.file_uploader("Upload a CSV of connection records", type=["csv", "txt"])
    use_sample = st.checkbox("…or use the bundled sample (500 KDDTest+ records)", value=up is None)
    source = up if up is not None else (SAMPLE if use_sample and SAMPLE.exists() else None)
    if source is None:
        st.info("Upload a file to begin. Expected: the 41 NSL-KDD feature columns (header optional); a label column is optional.")
        st.stop()

    try:
        df = read_traffic(source)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    res = det.predict(df, task, model)
    if task == "binary":
        res["prediction"] = np.where(res["p_attack"] >= thr, "attack", "normal")
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    n = len(out)
    flagged = int((out["prediction"] != "normal").sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Records", f"{n:,}")
    k2.metric("Flagged as attack", f"{flagged:,}  ({flagged / n:.1%})")
    k3.metric("Model", model)

    left, right = st.columns(2)
    with left:
        st.subheader("Predictions")
        counts = out["prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.bar(counts.index, counts.values, color=[BLUE if i == "normal" else ORANGE for i in counts.index], width=0.55)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color="#e6edf7")
        ax.set_ylabel("Records", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        st.pyplot(fig, transparent=True)
        plt.close(fig)

    with right:
        if "attack" in out.columns:
            st.subheader("Check against the labels you supplied")
            truth = out["attack"].map(C.ATTACK_FAMILY)
            if task == "binary":
                truth = np.where(truth == "normal", "normal", "attack")
            truth = pd.Series(truth)
            acc = float((truth == out["prediction"]).mean())
            st.metric("Accuracy on this file", f"{acc:.1%}")
            st.dataframe(pd.crosstab(truth.rename("actual"), out["prediction"].rename("predicted")))
            if truth.isna().any():
                st.caption("Some labels were not recognised NSL-KDD attack names and were ignored in the table.")
        else:
            st.subheader("No label column")
            st.caption("Add a `label` column (e.g. `neptune`, `normal`) to see accuracy and a confusion table.")

    st.subheader("Most suspicious records")
    sort_col = "p_attack" if task == "binary" else "confidence"
    show_cols = ["prediction", sort_col, "protocol_type", "service", "flag", "src_bytes", "dst_bytes"] + (
        ["attack"] if "attack" in out.columns else [])
    top = out[out["prediction"] != "normal"].sort_values(sort_col, ascending=False)
    st.dataframe(top[show_cols].head(200).round(3), hide_index=True, width="stretch", height=300)
    st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    with st.expander("About this tool"):
        st.markdown(
            """
Trained on **NSL-KDD** (KDDTrain+). Real attacks in the held-out KDDTest+ set include types never seen in training,
so expect lower recall on novel attacks than on known ones. See the README for measured numbers.
This is a research/teaching model and is **not** a production IDS.
"""
        )


def bench():
    ui.section("Bench",
               "Pick a real KDDTest+ connection record and compare five classifiers side by side. The page below runs in your browser "
               "and uses its own models, separate from the Detector page.")
    if not BENCH.exists():
        st.warning("The bench page (app/bench.html) is missing from this copy of the repo.")
        return
    page = BENCH.read_text(encoding="utf-8").replace("<html>", '<html data-theme="dark">', 1)
    components.html(page, height=1500, scrolling=True)
