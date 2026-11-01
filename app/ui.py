"""Look and feel: CSS, small HTML components and a shared chart style."""
import html

import altair as alt
import streamlit as st

INK, MUTE, LINE, PANEL, BG = "#e6edf7", "#94a3b8", "#243044", "#111827", "#0a0f1c"
ACCENT, WARM = "#38bdf8", "#f97316"

CSS = f"""
<style>
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer {{ visibility: hidden; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* hero */
.hero {{ position: relative; overflow: hidden; border: 1px solid {LINE}; border-radius: 18px; padding: 34px 34px 30px;
  background: radial-gradient(1200px 400px at 85% -20%, rgba(56,189,248,.22), transparent 60%),
              radial-gradient(900px 380px at -10% 120%, rgba(249,115,22,.16), transparent 55%), {PANEL}; }}
.hero:before {{ content: ""; position: absolute; inset: 0; opacity: .35;
  background-image: linear-gradient({LINE} 1px, transparent 1px), linear-gradient(90deg, {LINE} 1px, transparent 1px);
  background-size: 34px 34px; mask-image: radial-gradient(circle at 70% 20%, #000, transparent 70%); }}
.hero > * {{ position: relative; }}
.eyebrow {{ color: {ACCENT}; font: 600 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .16em; text-transform: uppercase; }}
.hero h1 {{ margin: 12px 0 10px; font-size: clamp(1.7rem, 4.2vw, 2.7rem); line-height: 1.12; font-weight: 750; text-wrap: balance; }}
.hero h1 em {{ font-style: normal; background: linear-gradient(90deg, {ACCENT}, {WARM}); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ margin: 0; color: {MUTE}; max-width: 70ch; font-size: 1.02rem; line-height: 1.6; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
.chip {{ border: 1px solid {LINE}; background: rgba(17,24,39,.7); border-radius: 999px; padding: 4px 12px; font-size: .8rem; color: {INK}; }}
.chip.c {{ border-color: rgba(129,140,248,.5); color: #c7d2fe; }}
.chip.d {{ border-color: rgba(249,115,22,.55); color: #fed7aa; }}

/* KPI cards */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0 6px; }}
.kpi {{ border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; background: {PANEL}; transition: transform .15s, border-color .15s; }}
.kpi:hover {{ transform: translateY(-2px); border-color: {ACCENT}; }}
.kpi .v {{ font: 700 1.85rem/1.1 ui-sans-serif, system-ui; letter-spacing: -.02em; }}
.kpi .l {{ margin-top: 6px; color: {MUTE}; font-size: .82rem; }}
.kpi .s {{ margin-top: 2px; color: {ACCENT}; font-size: .78rem; }}

/* finding cards */
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(188px, 1fr)); gap: 12px; margin: 8px 0; }}
.card {{ border: 1px solid {LINE}; border-radius: 14px; padding: 16px; background: {PANEL}; }}
.card .n {{ color: {ACCENT}; font: 700 .78rem ui-monospace, Menlo, monospace; letter-spacing: .12em; }}
.card h4 {{ margin: 6px 0 6px; font-size: 1.02rem; line-height: 1.3; }}
.card p {{ margin: 0; color: {MUTE}; font-size: .88rem; line-height: 1.55; }}

/* model cards */
.mcards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(176px, 1fr)); gap: 12px; margin: 8px 0; }}
.mcard {{ border: 1px solid {LINE}; border-top: 3px solid var(--c); border-radius: 14px; padding: 14px 15px 13px; background: {PANEL}; }}
.mcard .fam {{ font: 600 .68rem ui-monospace, Menlo, monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--c); }}
.mcard h4 {{ margin: 6px 0 4px; font-size: 1.12rem; }}
.mcard .tag {{ color: {MUTE}; font-size: .8rem; line-height: 1.4; min-height: 2.3em; }}
.mcard .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 10px; margin-top: 12px; }}
.mcard .stats b {{ display: block; font-size: 1.02rem; }}
.mcard .stats span {{ color: {MUTE}; font-size: .72rem; }}

/* section headings */
.sec {{ margin: 30px 0 4px; font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
.sec:before {{ content: ""; width: 4px; height: 1.1em; border-radius: 2px; background: linear-gradient(180deg, {ACCENT}, {WARM}); }}
.sub {{ color: {MUTE}; margin: 0 0 12px; font-size: .92rem; max-width: 80ch; line-height: 1.55; }}
.note {{ border-left: 3px solid {ACCENT}; background: rgba(56,189,248,.07); padding: 10px 14px; border-radius: 0 10px 10px 0; color: {INK}; font-size: .9rem; margin: 10px 0; }}
.note.warn {{ border-left-color: {WARM}; background: rgba(249,115,22,.08); }}
.src {{ color: {MUTE}; font-size: .78rem; margin-top: -4px; }}

/* page navigation: the radio becomes a pill bar */
div[data-testid="stRadioGroup"], div[data-testid="stRadio"] > div[role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
label[data-testid="stRadioOption"], div[data-testid="stRadio"] label[data-baseweb="radio"] {{ background: {PANEL}; border: 1px solid {LINE};
  border-radius: 999px; padding: 5px 16px 5px 12px; margin: 0; cursor: pointer; transition: border-color .15s, background .15s; }}
label[data-testid="stRadioOption"]:hover, div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{ border-color: {ACCENT}; }}
label[data-testid="stRadioOption"][data-selected="true"], div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
  background: rgba(56,189,248,.16); border-color: {ACCENT}; }}
label[data-testid="stRadioOption"] > div > div:first-child, div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display: none; }}
label[data-testid="stRadioOption"] p, div[data-testid="stRadio"] label[data-baseweb="radio"] p {{ font-size: .92rem; font-weight: 600; }}

/* tables, expanders, buttons */
div[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 12px; background: {PANEL}; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 999px; border: 1px solid {LINE}; background: {PANEL}; }}
.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
div[data-testid="stMetric"] {{ border: 1px solid {LINE}; border-radius: 12px; padding: 10px 14px; background: {PANEL}; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow, title_html, text, chips=()):
    """`title_html` may contain <em> for the gradient accent. `chips` are (label, kind) with kind in '', 'c', 'd'."""
    chips_html = "".join(f'<span class="chip {k}">{html.escape(t)}</span>' for t, k in chips)
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{title_html}</h1>'
                f'<p>{html.escape(text)}</p><div class="chips">{chips_html}</div></div>', unsafe_allow_html=True)


def kpis(items):
    """items: (value, label, sub)."""
    cells = "".join(f'<div class="kpi"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div>'
                    f'<div class="s">{html.escape(s)}</div></div>' for v, l, s in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def cards(items):
    """items: (title, text). Numbered automatically."""
    cells = "".join(f'<div class="card"><div class="n">{i:02d}</div><h4>{html.escape(t)}</h4><p>{html.escape(p)}</p></div>'
                    for i, (t, p) in enumerate(items, 1))
    st.markdown(f'<div class="cards">{cells}</div>', unsafe_allow_html=True)


def model_cards(items):
    """items: dicts with name, family, color, tagline and stats [(value, label), ...]."""
    cells = ""
    for it in items:
        stats = "".join(f"<div><b>{html.escape(v)}</b><span>{html.escape(l)}</span></div>" for v, l in it["stats"])
        cells += (f'<div class="mcard" style="--c:{it["color"]}"><div class="fam">{html.escape(it["family"])}</div>'
                  f'<h4>{html.escape(it["name"])}</h4><div class="tag">{html.escape(it["tagline"])}</div>'
                  f'<div class="stats">{stats}</div></div>')
    st.markdown(f'<div class="mcards">{cells}</div>', unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f'<div class="sec">{html.escape(title)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="sub">{html.escape(sub)}</p>', unsafe_allow_html=True)


def note(text, warn=False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{html.escape(text)}</div>', unsafe_allow_html=True)


def source(text):
    st.markdown(f'<div class="src">{html.escape(text)}</div>', unsafe_allow_html=True)


def style(chart, height=None):
    """Dark, minimal Altair styling that blends with the page."""
    if height:
        chart = chart.properties(height=height)
    return (chart.configure(background="rgba(0,0,0,0)")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTE, titleColor=MUTE, gridColor=LINE, domainColor=LINE, tickColor=LINE,
                            labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK, titleColor=MUTE, labelFontSize=12, orient="bottom")
            .configure_title(color=INK, fontSize=14, anchor="start"))


def color_scale(models, colors):
    return alt.Scale(domain=list(models), range=[colors[m] for m in models])
