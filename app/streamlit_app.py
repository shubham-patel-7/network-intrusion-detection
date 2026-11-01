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


def main():
    ui.inject_css()
    st.session_state.setdefault("page", "🏠 Overview")
    page = st.radio("Page", list(PAGES), key="page", horizontal=True, label_visibility="collapsed")
    st.write("")
    PAGES[page]()


main()
