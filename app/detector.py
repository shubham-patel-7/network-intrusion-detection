"""The Detector page: upload a CSV of CICIoT2023 flows and classify every flow with any of the five study models,
or run all five on the same file and see where they disagree."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import study_data as S
import study_models
import ui

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#38bdf8", "#f97316"
SAMPLE = ROOT / "sample_data" / "sample_flows.csv"
SHOW = ["Protocol Type", "Rate", "Header_Length", "Tot size", "IAT"]  # a few readable columns for the alert table


@st.cache_resource(show_spinner="Loading the model…")
def _model(key):
    return study_models.load_model(key)


@st.cache_resource
def _pre():
    return study_models.load_preprocessor()


def _read_source():
    """File upload or the bundled sample; returns (DataFrame of flows, label column name or None)."""
    up = st.file_uploader("Upload a CSV of CICIoT2023 flows", type=["csv", "txt"])
    use_sample = st.checkbox("…or use the bundled sample (500 test flows)", value=up is None)
    source = up if up is not None else (SAMPLE if use_sample and SAMPLE.exists() else None)
    if source is None:
        st.info("Upload a file to begin. Expected: a CSV with the flow feature columns of CICIoT2023 (header required); "
                "a `Label` column is optional.")
        st.stop()
    try:
        return study_models.read_flows(source)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()


def _truth(df, label):
    """True category of every flow when a label column was supplied, else None."""
    if label is None:
        return None
    return df[label].map(study_models.category)


def detector():
    ui.section("Detector",
               f"Upload a CSV of {S.DATASET_NAME} flows and classify every flow with one of the five study models, or run all five on "
               "the same file. Anything that is not benign raises an alert.")
    if not study_models.available():
        st.error("The exported models are missing from this copy of the repo (models/study/).")
        st.stop()
    meas = study_models.meta()
    classes = study_models.classes()
    st.session_state.setdefault("detector_model", S.BEST)  # the model with the best macro-F1; the Models page can preselect another
    c1, c2 = st.columns([1.3, 1])
    name = c1.radio("Model", S.MODELS, horizontal=True, key="detector_model")
    key = study_models.KEY_OF[name]
    c2.markdown(f"<div style='padding-top:1.9rem;color:#94a3b8;font-size:.85rem'>{S.FAMILY[name]} · benchmark macro-F1 "
                f"<b style='color:#e6edf7'>{S.MAIN[name]['mf1'][0]:.3f}</b> · {S.COST[name]['lat']:.2f} ms per record</div>",
                unsafe_allow_html=True)
    df, label = _read_source()
    model, pre = _model(key), _pre()
    res = study_models.classify(key, model, pre, df)
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    n = len(out)
    flagged = int(out["alert"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Flows", f"{n:,}")
    k2.metric("Flagged as attack", f"{flagged:,}  ({flagged / n:.1%})")
    k3.metric("Model", name)

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted classes")
        counts = out["prediction"].value_counts().reindex(classes).fillna(0).reset_index()
        counts.columns = ["Class", "Flows"]
        ch = alt.Chart(counts).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("Class:N", sort=classes, axis=alt.Axis(labelAngle=-30, title=None)), y=alt.Y("Flows:Q", title=None),
            color=alt.condition(alt.datum.Class == classes[0], alt.value(BLUE), alt.value(ORANGE)),
            tooltip=["Class", alt.Tooltip("Flows:Q", format=",")])
        txt = alt.Chart(counts).mark_text(dy=-8, color="#e6edf7", fontWeight=600).encode(
            x=alt.X("Class:N", sort=classes), y="Flows:Q", text=alt.Text("Flows:Q", format=","))
        st.altair_chart(ui.style(ch + txt, 280), width="stretch")
    truth = _truth(df, label)
    with right:
        if truth is not None:
            st.subheader("Check against your labels")
            known = truth.notna()
            acc = float((truth[known].values == out["prediction"][known].values).mean())
            st.metric("Accuracy on this file", f"{acc:.1%}")
            st.dataframe(pd.crosstab(truth[known].rename("actual"), out["prediction"][known].rename("predicted")))
            if (~known).any():
                st.caption("Some labels were not recognised CICIoT2023 attack names and were ignored.")
        else:
            st.subheader("No label column")
            st.caption("Add a `Label` column (for example `DDOS-SYN_FLOOD`, `BENIGN`) to see accuracy and a confusion table.")

    st.subheader("Flows raising an alert")
    cols = ["prediction", "confidence"] + [c for c in SHOW if c in out.columns] + ([label] if label else [])
    top = out[out["alert"]].sort_values("confidence", ascending=False, na_position="last")
    st.dataframe(top[cols].head(200).round(3), hide_index=True, width="stretch", height=300)
    if key == "SVM":
        st.caption("The SVM outputs decision scores, not probabilities, so it has no confidence value.")
    st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    ui.section("Run all five on this file", "Same flows, five different verdicts. Where the models disagree is where the interesting "
               "cases are.")
    if st.toggle("Compare all five models", key="compare_all"):
        rows, verdicts = [], {}
        for m in S.MODELS:
            k = study_models.KEY_OF[m]
            r = study_models.classify(k, _model(k), pre, df)
            verdicts[m] = r["prediction"].values
            row = {"Model": m, "Family": S.FAMILY[m], "Flagged as attack": int(r["alert"].sum()), "Share (%)": round(r["alert"].mean() * 100, 1)}
            if truth is not None:
                kn = truth.notna()
                row["Accuracy (%)"] = round(float((truth[kn].values == r["prediction"][kn].values).mean()) * 100, 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        agree = pd.DataFrame(verdicts)
        same = (agree.nunique(axis=1) == 1).mean()
        st.caption(f"All five models give the same class for {same:.1%} of the flows.")
        extra = [c for c in SHOW[:2] + ([label] if label else []) if c in df.columns]
        disagree = pd.concat([df.reset_index(drop=True)[extra], agree], axis=1)[agree.nunique(axis=1) > 1]
        if len(disagree):
            with st.expander(f"The {len(disagree):,} flows the models disagree on"):
                st.dataframe(disagree.head(200), hide_index=True, width="stretch", height=300)
    with st.expander("About these models"):
        st.markdown(
            "These are the study's five models trained on the full training split with the settings chosen by the benchmark's random "
            "search (one seed; the MLP and LSTM run on a Keras backend). Each was tested once on the held-out test split when it was "
            "trained here:\n\n" +
            "\n".join(f"- **{m}**: accuracy {meas[study_models.KEY_OF[m]]['test']['acc']:.3f}, macro-F1 "
                      f"{meas[study_models.KEY_OF[m]]['test']['macro_f1']:.3f} (benchmark mean over seeds: {S.MAIN[m]['acc'][0]:.3f} and "
                      f"{S.MAIN[m]['mf1'][0]:.3f})" for m in S.MODELS) +
            ("\n\nTo fit free hosting, the Random Forest here is the first 40 of its 146 trees (the full forest needs about 550 MB "
             "of memory); the figures above are for those 40 trees." if "first 40" in meas["RF"]["settings_from"] else "") +
            "\n\nThe classifiers see pre-extracted flow features, are weakest on the rare classes, and are **not** a production "
            "intrusion detection system.")
