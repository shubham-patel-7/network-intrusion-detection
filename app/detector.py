"""The Detector page: upload a CSV of connection records and classify them with the models trained in this repo,
and the Bench page: an embedded standalone page that compares five classifiers on one chosen test record."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def detector():
    ui.section("Detector",
               "Upload a CSV of NSL-KDD connection records and classify each one with a model trained in this repository "
               "(classic machine learning against deep learning). These are not the five models of the study: they are four others, "
               "trained on the same data.")
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
