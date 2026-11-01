"""Machine learning vs deep learning for network intrusion detection on NSL-KDD.

The study results (Overview, Results, Ablations, About), a working detector for CSV files, and an interactive bench.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="ML vs DL for intrusion detection", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

import detector  # noqa: E402
import ui  # noqa: E402
import views  # noqa: E402

PAGES = {
    "🏠 Overview": views.overview,
    "📊 Results": views.results,
    "🧪 Ablations": views.ablations,
    "🛡️ Detector": detector.detector,
    "🔬 Bench": detector.bench,
    "ℹ️ About": views.about,
}


SLUGS = {name.split(" ", 1)[1].lower(): name for name in PAGES}  # overview, results, ablations, detector, bench, about


def main():
    ui.inject_css()
    if "page" not in st.session_state:  # first load: honour a deep link such as ?page=results
        st.session_state["page"] = SLUGS.get(str(st.query_params.get("page", "")).lower(), "🏠 Overview")
    page = st.radio("Page", list(PAGES), key="page", horizontal=True, label_visibility="collapsed")
    slug = page.split(" ", 1)[1].lower()
    if st.query_params.get("page") != slug:
        st.query_params["page"] = slug  # keep the address bar shareable
    st.write("")
    PAGES[page]()


main()
