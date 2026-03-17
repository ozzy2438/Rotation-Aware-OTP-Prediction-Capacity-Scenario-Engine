"""
Spirit Airlines FLL Hub: OTP Prediction & Demand-Capacity Scenario Engine
Streamlit Dashboard

Five pages:
  1. Overview        — KPI cards + OTP trend
  2. Route Performance — heatmap, delay cause bar, comparison table
  3. OTP Predictor   — input form + probability gauge + feature importance
  4. Scenario Simulator — what-if inputs + Monte Carlo histogram + table
  5. Ask Analytics   — LLM / pattern-matching chat interface

Run:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.llm_query import LLMQueryEngine
from src.models.scenario_simulator import (
    ROUTE_CONFIG,
    SCHEDULE_FACTORS,
    ScenarioInput,
    ScenarioSimulator,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spirit Airlines colour palette
# ---------------------------------------------------------------------------
SPIRIT_YELLOW = "#FFD700"
SPIRIT_GREY = "#4A4A4A"
SPIRIT_LIGHT_GREY = "#F0F0F0"
SPIRIT_DARK = "#1A1A1A"
SPIRIT_ACCENT = "#FFC200"

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SCALE = [[0, "#FF4444"], [0.5, SPIRIT_YELLOW], [1, "#00CC66"]]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spirit Airlines FLL Hub Analytics",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — production-grade dark theme
st.markdown(
    f"""
    <style>
    /* ── Base ─────────────────────────────────────────────── */
    .stApp {{ background-color: {SPIRIT_DARK}; color: #E8E8E8; }}

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #111111 0%, #1E1E1E 100%);
        border-right: 1px solid #2E2E2E;
    }}
    [data-testid="stSidebar"] * {{ color: #E0E0E0 !important; }}
    [data-testid="stSidebar"] .stRadio label {{
        color: #E0E0E0 !important;
        font-size: 0.95rem;
    }}
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        color: #E0E0E0 !important;
    }}
    [data-testid="stSidebarUserContent"] h2,
    [data-testid="stSidebarUserContent"] h3 {{
        color: {SPIRIT_YELLOW} !important;
    }}

    /* ── Main content text ────────────────────────────────── */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: #E0E0E0; }}
    h1, h2, h3, h4, h5 {{ color: #FFFFFF !important; }}

    /* ── Widget labels ────────────────────────────────────── */
    label, .stSelectbox label, .stSlider label,
    .stMultiSelect label, .stCheckbox label,
    .stTextInput label, .stNumberInput label,
    [data-testid="stWidgetLabel"] p,
    .stRadio label div p {{
        color: #D0D0D0 !important;
        font-weight: 500 !important;
    }}

    /* ── Select box / input backgrounds ──────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: #2A2A2A !important;
        border: 1px solid #444444 !important;
        color: #FFFFFF !important;
    }}
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {{
        color: #FFFFFF !important;
    }}

    /* ── Sliders ──────────────────────────────────────────── */
    .stSlider [data-testid="stTickBar"] span {{ color: #AAAAAA !important; }}
    .stSlider p {{ color: #D0D0D0 !important; }}
    [data-testid="stThumbValue"] {{ color: #FFFFFF !important; }}

    /* ── Checkboxes ───────────────────────────────────────── */
    .stCheckbox span {{ color: #D0D0D0 !important; }}

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {{
        background-color: #2A2A2A;
        color: #E0E0E0;
        border: 1px solid #444444;
        border-radius: 6px;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background-color: #3A3A3A;
        border-color: {SPIRIT_YELLOW};
        color: {SPIRIT_YELLOW};
    }}
    .stButton > button[kind="primary"] {{
        background-color: {SPIRIT_YELLOW} !important;
        color: #111111 !important;
        font-weight: bold;
        border: none;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {SPIRIT_ACCENT} !important;
        color: #000000 !important;
    }}

    /* ── Chat ─────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {{
        background-color: #222222;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #333333;
    }}
    [data-testid="stChatMessage"] p {{ color: #E8E8E8 !important; }}
    [data-testid="stChatInput"] textarea {{
        background-color: #2A2A2A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }}

    /* ── Info / Warning / Success boxes ──────────────────── */
    .stInfo {{ background-color: #1A2A3A; border-left: 4px solid #4A9EFF; }}
    .stInfo p, .stAlert p {{ color: #D0E8FF !important; }}
    .stWarning {{ background-color: #2A1F0A; border-left: 4px solid {SPIRIT_YELLOW}; }}
    .stWarning p {{ color: #FFE8A0 !important; }}
    .stSuccess {{ background-color: #0A2A1A; border-left: 4px solid #00CC66; }}
    .stSuccess p {{ color: #A0FFD0 !important; }}
    .stError {{ background-color: #2A0A0A; border-left: 4px solid #FF4444; }}
    .stError p {{ color: #FFD0D0 !important; }}

    /* ── Expander ─────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background-color: #222222 !important;
        color: #D0D0D0 !important;
        border: 1px solid #333333;
        border-radius: 6px;
    }}
    .streamlit-expanderContent {{
        background-color: #1E1E1E !important;
        border: 1px solid #333333;
    }}

    /* ── Tabs ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: #AAAAAA !important;
        border-radius: 6px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SPIRIT_YELLOW} !important;
        color: #111111 !important;
        font-weight: bold;
    }}

    /* ── Dataframe ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #333333;
    }}

    /* ── Code blocks ──────────────────────────────────────── */
    .stCode, code, pre {{
        background-color: #1A1A2E !important;
        color: #E0E0FF !important;
        border: 1px solid #333355;
        border-radius: 6px;
    }}

    /* ── Caption ──────────────────────────────────────────── */
    .stCaption p, [data-testid="caption"] {{ color: #888888 !important; }}

    /* ── Spinner ──────────────────────────────────────────── */
    .stSpinner p {{ color: {SPIRIT_YELLOW} !important; }}

    /* ── Custom components ────────────────────────────────── */
    .metric-card {{
        background: linear-gradient(135deg, #252525 0%, #2F2F2F 100%);
        border: 1px solid #3A3A3A;
        border-top: 3px solid {SPIRIT_YELLOW};
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: transform 0.2s, border-top-color 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-top-color: {SPIRIT_ACCENT};
    }}
    .metric-value {{ font-size: 2.1rem; font-weight: 800; color: {SPIRIT_YELLOW}; letter-spacing: -0.5px; }}
    .metric-label {{ font-size: 0.78rem; color: #999999; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-delta {{ font-size: 0.78rem; margin-top: 6px; font-weight: 500; }}

    .section-header {{
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
        border-bottom: 2px solid {SPIRIT_YELLOW};
        padding-bottom: 8px;
        margin-bottom: 20px;
        letter-spacing: 0.2px;
    }}
    .section-header::before {{
        content: "▍";
        color: {SPIRIT_YELLOW};
        margin-right: 8px;
    }}

    .insight-box {{
        background: linear-gradient(135deg, #1E2A1E, #1A2A2A);
        border: 1px solid #2A4A2A;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #C0E8C0;
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    .suggest-btn button {{
        background-color: #242424 !important;
        color: #D0D0D0 !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        padding: 4px 12px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .suggest-btn button:hover {{
        border-color: {SPIRIT_YELLOW} !important;
        color: {SPIRIT_YELLOW} !important;
    }}

    /* ── Divider ──────────────────────────────────────────── */
    hr {{ border-color: #333333 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_flight_data() -> Optional[pd.DataFrame]:
    """Load processed flight data from Parquet."""
    path = PROJECT_ROOT / "data" / "processed" / "flights_processed.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=600)
def load_capacity_data() -> Optional[pd.DataFrame]:
    """Load processed capacity data from Parquet."""
    path = PROJECT_ROOT / "data" / "processed" / "capacity_processed.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def get_db_engine() -> Optional[LLMQueryEngine]:
    """Return a cached LLMQueryEngine instance."""
    db_path = PROJECT_ROOT / "data" / "spirit_otp.duckdb"
    if not db_path.exists():
        return None
    return LLMQueryEngine(db_path=db_path)


@st.cache_resource
def get_simulator() -> ScenarioSimulator:
    """Return a cached ScenarioSimulator instance."""
    return ScenarioSimulator(rng_seed=42)


@st.cache_data(ttl=3600)
def load_model_meta() -> dict:
    """Load OTP model metadata and best MLflow run info."""
    import json
    meta: dict = {}
    meta_path = PROJECT_ROOT / "data" / "models" / "otp_predictor_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    try:
        from src.mlops.tracking import get_tracker
        best = get_tracker().get_best_run()
        if best:
            meta["best_run"] = best
    except Exception:
        pass
    return meta


@st.cache_data(ttl=3600)
def load_otp_predictor():
    """Load the trained OTP predictor model if available."""
    model_path = PROJECT_ROOT / "data" / "models" / "otp_predictor.pkl"
    if not model_path.exists():
        return None
    try:
        from src.models.otp_predictor import OTPPredictor
        return OTPPredictor.load(PROJECT_ROOT / "data" / "models")
    except Exception as exc:
        logger.warning("Could not load OTP predictor: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helper: check data availability
# ---------------------------------------------------------------------------

def _data_unavailable_banner(msg: str = "") -> None:
    st.warning(
        msg or (
            "Processed data not found. Please run the pipeline first:\n\n"
            "```bash\npython main.py generate-data && python main.py run-etl\n```"
        )
    )


def _metric_card(label: str, value: str, delta: str = "", delta_color: str = "#00CC66") -> str:
    delta_html = (
        f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>'
        if delta else ""
    )
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def sidebar_nav() -> str:
    """Render sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:12px 0 6px 0;">
              <div style="font-size:1.5rem; font-weight:900; color:{SPIRIT_YELLOW};
                          letter-spacing:1px; line-height:1.2;">✈ SPIRIT AIRLINES</div>
              <div style="font-size:0.72rem; color:#888888; text-transform:uppercase;
                          letter-spacing:2px; margin-top:2px;">FLL Hub Analytics Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<hr style="border-color:#2A2A2A; margin:8px 0 14px 0;">',
            unsafe_allow_html=True,
        )

        pages = {
            "Overview": "📊",
            "Route Performance": "🗺️",
            "OTP Predictor": "🔮",
            "Scenario Simulator": "⚙️",
            "Ask Analytics": "💬",
        }
        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            format_func=lambda p: f"{pages[p]}  {p}",
            label_visibility="collapsed",
        )

        st.markdown(
            '<hr style="border-color:#2A2A2A; margin:14px 0 10px 0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.72rem; color:#666666; line-height:1.8;">'
            'OTP & Scenario Engine <span style="color:#444444;">v1.0</span><br>'
            'Routes: FLL Hub (10 routes)<br>'
            'Period: 2022–2024 (synthetic)<br>'
            'Model: XGBoost · AUC 0.817'
            "</div>",
            unsafe_allow_html=True,
        )
    return selected


# ---------------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------------

def page_overview(flights: pd.DataFrame, capacity: pd.DataFrame) -> None:
    """Render the Overview page with KPI cards and OTP trend."""
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1A1A1A,#252530);
                    border:1px solid #2E2E3A; border-left:4px solid {SPIRIT_YELLOW};
                    border-radius:10px; padding:16px 22px; margin-bottom:24px;
                    display:flex; align-items:center; gap:14px;">
          <div style="font-size:2rem;">✈</div>
          <div>
            <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">
              Spirit Airlines — Fort Lauderdale Hub (FLL)
            </div>
            <div style="font-size:0.82rem; color:#888888; margin-top:2px;">
              OTP Prediction &amp; Capacity Scenario Engine &nbsp;·&nbsp;
              2022–2024 &nbsp;·&nbsp; 10 Routes &nbsp;·&nbsp; Synthetic Data
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-header">Network KPIs</div>', unsafe_allow_html=True)

    # -- KPI Computations --
    operated = flights[flights["Cancelled"] == 0]
    otp_pct = (operated["ArrDel15"] == 0).mean() * 100
    avg_delay = operated["ArrDelay"].mean()
    cancel_rate = flights["Cancelled"].mean() * 100
    total_routes = flights["Route"].nunique()

    # Load factor from capacity
    if capacity is not None and len(capacity) > 0:
        avg_lf = capacity["LoadFactor"].mean() * 100
        total_pax = capacity["Passengers"].sum()
    else:
        avg_lf = 0
        total_pax = 0

    # -- KPI Cards --
    cols = st.columns(5)
    kpis = [
        ("OTP Rate", f"{otp_pct:.1f}%", "vs 79% industry avg", SPIRIT_YELLOW if otp_pct > 79 else "#FF4444"),
        ("Avg Arr Delay", f"{avg_delay:.1f} min", "all operated flights", "#AAAAAA"),
        ("Load Factor", f"{avg_lf:.1f}%", "monthly avg", "#00CC66"),
        ("Cancel Rate", f"{cancel_rate:.2f}%", "2022–2024", "#AAAAAA"),
        ("Routes", f"{total_routes}", "FLL hub", SPIRIT_YELLOW),
    ]
    for col, (label, val, delta, color) in zip(cols, kpis):
        col.markdown(_metric_card(label, val, delta, color), unsafe_allow_html=True)

    # -- Model Status Strip --
    meta = load_model_meta()
    if meta:
        auc   = meta.get("eval_results", {}).get("auc_roc",  meta.get("best_run", {}).get("auc_roc",  "—"))
        f1    = meta.get("eval_results", {}).get("f1_score", meta.get("best_run", {}).get("f1_score", "—"))
        feats = len(meta.get("feature_columns", []) or []) or "43"
        run_id = (meta.get("best_run") or {}).get("run_id", "")
        run_badge = f'<span style="color:#555555; font-size:0.72rem;">run {run_id[:8]}</span>' if run_id else ""
        st.markdown(
            f"""
            <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;
                        background:#1A1A2A; border:1px solid #2A2A3A; border-radius:8px;
                        padding:10px 18px; margin:14px 0 6px 0; font-size:0.82rem;">
              <span style="color:#888888;">Active Model:</span>
              <span style="color:#FFFFFF; font-weight:600;">XGBoost OTP Predictor</span>
              <span style="color:#333333;">|</span>
              <span style="color:#888888;">AUC-ROC</span>
              <span style="color:{SPIRIT_YELLOW}; font-weight:700;">{auc if isinstance(auc, str) else f"{auc:.4f}"}</span>
              <span style="color:#333333;">|</span>
              <span style="color:#888888;">F1</span>
              <span style="color:{SPIRIT_YELLOW}; font-weight:700;">{f1 if isinstance(f1, str) else f"{f1:.4f}"}</span>
              <span style="color:#333333;">|</span>
              <span style="color:#888888;">Features</span>
              <span style="color:#CCCCCC; font-weight:600;">{feats}</span>
              <span style="color:#333333;">|</span>
              <span style="color:#888888;">Tracker</span>
              <span style="color:#00AA55; font-weight:600;">MLflow ✓</span>
              {run_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- OTP Trend --
    st.markdown("### Monthly Network OTP Trend")
    monthly = (
        operated.groupby(["Year", "Month"])
        .agg(
            total=("ArrDel15", "count"),
            ontime=("ArrDel15", lambda x: (x == 0).sum()),
            avg_delay=("ArrDelay", "mean"),
        )
        .reset_index()
    )
    monthly["otp_pct"] = monthly["ontime"] / monthly["total"] * 100
    monthly["YearMonth"] = pd.to_datetime(monthly[["Year", "Month"]].assign(day=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["YearMonth"], y=monthly["otp_pct"],
        mode="lines+markers", name="OTP %",
        line=dict(color=SPIRIT_YELLOW, width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>OTP: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=79, line_dash="dash", line_color="#FF4444",
                  annotation_text="79% Industry Avg", annotation_position="right")
    fig.add_hline(y=80, line_dash="dot", line_color="#00CC66",
                  annotation_text="80% Target", annotation_position="right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SPIRIT_DARK,
        plot_bgcolor=SPIRIT_DARK,
        yaxis_title="OTP (%)",
        xaxis_title="Month",
        height=350,
        margin=dict(t=20, b=20, l=20, r=80),
        font=dict(color="#FFFFFF"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Route Summary Table --
    st.markdown("### Route Performance Summary")
    route_summary = (
        operated.groupby("Route")
        .agg(
            total_flights=("ArrDel15", "count"),
            otp_pct=("ArrDel15", lambda x: (x == 0).mean() * 100),
            avg_delay=("ArrDelay", "mean"),
            cancelled=("Cancelled", "sum"),
        )
        .reset_index()
        .sort_values("otp_pct", ascending=False)
    )
    route_summary.columns = ["Route", "Operated Flights", "OTP %", "Avg Delay (min)", "Cancelled"]
    route_summary["OTP %"] = route_summary["OTP %"].round(1)
    route_summary["Avg Delay (min)"] = route_summary["Avg Delay (min)"].round(1)

    st.dataframe(
        route_summary.style.background_gradient(subset=["OTP %"], cmap="RdYlGn", vmin=60, vmax=90),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# PAGE 2: Route Performance
# ---------------------------------------------------------------------------

def page_route_performance(flights: pd.DataFrame, capacity: pd.DataFrame) -> None:
    """Render Route Performance page with heatmap and delay breakdown."""
    st.markdown('<div class="section-header">Route Performance Deep-Dive</div>',
                unsafe_allow_html=True)

    operated = flights[flights["Cancelled"] == 0].copy()

    # -- Route Filter --
    routes = sorted(operated["Route"].unique())
    selected_routes = st.multiselect("Filter Routes", routes, default=routes)
    operated = operated[operated["Route"].isin(selected_routes)]

    # -- OTP Heatmap: Route × Month --
    st.markdown("### OTP Rate Heatmap (Route × Month)")
    heatmap_data = (
        operated.groupby(["Route", "Month"])
        .agg(otp=("ArrDel15", lambda x: (x == 0).mean() * 100))
        .reset_index()
    )
    heatmap_pivot = heatmap_data.pivot(index="Route", columns="Month", values="otp")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[month_labels[i - 1] for i in heatmap_pivot.columns],
        y=heatmap_pivot.index.tolist(),
        colorscale=COLOR_SCALE,
        zmin=60, zmax=90,
        text=np.round(heatmap_pivot.values, 1),
        texttemplate="%{text}%",
        colorbar=dict(title="OTP %", ticksuffix="%"),
        hovertemplate="Route: %{y}<br>Month: %{x}<br>OTP: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SPIRIT_DARK,
        plot_bgcolor=SPIRIT_DARK,
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # -- Delay Cause Breakdown --
    st.markdown("### Delay Cause Decomposition")
    delay_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    delay_labels = ["Carrier", "Weather", "NAS / ATC", "Security", "Late Aircraft"]
    color_map = {
        "Carrier": "#FF6B35",
        "Weather": "#4ECDC4",
        "NAS / ATC": "#45B7D1",
        "Security": "#96CEB4",
        "Late Aircraft": SPIRIT_YELLOW,
    }

    delay_only = operated[operated["ArrDel15"] == 1]
    delay_agg = (
        delay_only.groupby("Route")[delay_cols]
        .mean()
        .reset_index()
    )
    delay_agg.columns = ["Route"] + delay_labels

    fig_bar = go.Figure()
    for label, color in color_map.items():
        if label in delay_agg.columns:
            fig_bar.add_trace(go.Bar(
                x=delay_agg["Route"],
                y=delay_agg[label],
                name=label,
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}} min<extra></extra>",
            ))
    fig_bar.update_layout(
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SPIRIT_DARK,
        plot_bgcolor=SPIRIT_DARK,
        yaxis_title="Avg Delay (min, delayed flights only)",
        height=380,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Route Comparison Table --
    st.markdown("### Detailed Route Comparison")
    if capacity is not None and len(capacity) > 0:
        lf_by_route = capacity.groupby("Route")["LoadFactor"].mean().reset_index()
        lf_by_route.columns = ["Route", "Avg LF"]
    else:
        lf_by_route = pd.DataFrame({"Route": operated["Route"].unique(), "Avg LF": np.nan})

    route_detail = (
        operated.groupby("Route")
        .agg(
            flights=("ArrDel15", "count"),
            otp_pct=("ArrDel15", lambda x: round((x == 0).mean() * 100, 1)),
            avg_delay=("ArrDelay", lambda x: round(x.mean(), 1)),
            p75_delay=("ArrDelay", lambda x: round(x.quantile(0.75), 1)),
            carrier_del=("CarrierDelay", "mean"),
            weather_del=("WeatherDelay", "mean"),
        )
        .reset_index()
        .merge(lf_by_route, on="Route", how="left")
    )
    route_detail["Avg LF"] = (route_detail["Avg LF"] * 100).round(1)
    route_detail.columns = [
        "Route", "Operated", "OTP%", "Avg Delay", "P75 Delay",
        "Carrier Del", "Weather Del", "Load Factor%",
    ]

    st.dataframe(
        route_detail.style.background_gradient(subset=["OTP%"], cmap="RdYlGn", vmin=60, vmax=90),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# PAGE 3: OTP Predictor
# ---------------------------------------------------------------------------

def page_otp_predictor(flights: pd.DataFrame) -> None:
    """Render OTP Predictor page with input form and probability gauge."""
    st.markdown('<div class="section-header">OTP Delay Predictor</div>',
                unsafe_allow_html=True)

    predictor = load_otp_predictor()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### Flight Parameters")
        route = st.selectbox("Route", list(ROUTE_CONFIG.keys()), index=0)
        dep_hour = st.slider("Departure Hour (local)", 5, 23, 14)
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            index=6,
        )
        day_of_week = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d],
            index=4,
        )
        # Weather section with live fetch capability
        st.markdown(
            f'<div style="color:#AAAAAA; font-size:0.82rem; text-transform:uppercase; '
            f'letter-spacing:1px; margin:16px 0 8px 0;">Weather at FLL</div>',
            unsafe_allow_html=True,
        )

        # Live weather fetch button
        from src.integrations.weather_api import get_client as get_weather_client
        wx_client = get_weather_client()

        if wx_client.is_configured():
            fetch_wx = st.button("🌤 Fetch Live FLL Weather", use_container_width=True)
            if fetch_wx:
                try:
                    with st.spinner("Fetching live weather from OpenWeatherMap …"):
                        snap = wx_client.get_fll_weather()
                    st.session_state["wx_severity"]    = snap.severity
                    st.session_state["wx_thunderstorm"] = snap.thunderstorm
                    st.session_state["wx_low_vis"]      = snap.low_visibility
                    st.session_state["wx_wind_gust"]    = snap.wind_gust
                    st.success(
                        f"FLL: {snap.description.title()} | "
                        f"Wind {snap.wind_speed_ms:.1f} m/s | "
                        f"Visibility {snap.visibility_m/1000:.1f} km"
                    )
                except Exception as exc:
                    st.error(f"Weather fetch failed: {exc}")
        else:
            st.markdown(
                '<div style="font-size:0.78rem; color:#666666; padding:4px 0 8px 0;">'
                'Set <code>OPENWEATHERMAP_API_KEY</code> in .env for live weather</div>',
                unsafe_allow_html=True,
            )

        weather_severity = st.slider(
            "Weather Severity (0=Clear, 10=Severe)",
            0.0, 10.0,
            float(st.session_state.get("wx_severity", 1.0)),
            0.5,
        )
        thunderstorm   = st.checkbox("Thunderstorm",                   value=bool(st.session_state.get("wx_thunderstorm", False)))
        low_visibility = st.checkbox("Low Visibility (<3 miles)",      value=bool(st.session_state.get("wx_low_vis", False)))
        wind_gust      = st.checkbox("Strong Wind Gusts (>25 kt)",     value=bool(st.session_state.get("wx_wind_gust", False)))
        prev_delay = st.slider("Previous Flight Delay on This Tail (min)", 0, 120, 0)

        predict_btn = st.button("Predict OTP", type="primary", use_container_width=True)

    with col_result:
        if not predict_btn:
            st.markdown(
                f"""
                <div style="margin-top:40px; padding:28px 24px; background:#222222;
                            border-radius:12px; border:1px solid #333333; text-align:center;">
                  <div style="font-size:3rem; margin-bottom:12px;">🔮</div>
                  <div style="font-size:1.05rem; font-weight:600; color:#E0E0E0;
                              margin-bottom:8px;">Configure & Predict</div>
                  <div style="font-size:0.85rem; color:#888888; line-height:1.6;">
                    Select a route, set flight conditions,<br>
                    and click <b style="color:{SPIRIT_YELLOW};">Predict OTP</b> to get<br>
                    a real-time delay probability estimate.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if predict_btn:
            # Build feature dict
            from src.pipeline.features import ROUTE_DISTANCE, ROUTE_OTP_BASELINE
            import math

            features = {
                "hour_of_day": dep_hour,
                "day_of_week": day_of_week,
                "month": month,
                "quarter": (month - 1) // 3 + 1,
                "is_weekend": int(day_of_week >= 5),
                "is_monday": int(day_of_week == 0),
                "is_friday": int(day_of_week == 4),
                "is_holiday": 0,
                "is_holiday_window": 0,
                "is_early_morning": int(dep_hour < 7),
                "is_peak_morning": int(7 <= dep_hour < 10),
                "is_afternoon": int(13 <= dep_hour < 18),
                "is_evening": int(dep_hour >= 18),
                "is_summer": int(month in (6, 7, 8)),
                "is_hurricane_season": int(month in (8, 9, 10)),
                "is_winter": int(month in (12, 1, 2)),
                "hour_sin": math.sin(2 * math.pi * dep_hour / 24),
                "hour_cos": math.cos(2 * math.pi * dep_hour / 24),
                "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
                "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
                "dow_sin": math.sin(2 * math.pi * day_of_week / 7),
                "dow_cos": math.cos(2 * math.pi * day_of_week / 7),
                "route_distance": ROUTE_DISTANCE.get(route, 1000),
                "route_otp_baseline": ROUTE_OTP_BASELINE.get(route, 0.75),
                "is_high_congestion_dest": int(route.split("-")[1] in ("ATL","ORD","JFK","DFW","LAX")),
                "is_long_haul": int(ROUTE_DISTANCE.get(route, 0) >= 1500),
                "is_short_haul": int(ROUTE_DISTANCE.get(route, 0) < 400),
                "distance_normalised": ROUTE_DISTANCE.get(route, 1000) / 2342,
                "weather_severity": weather_severity + (4 if thunderstorm else 0) + (2.5 if low_visibility else 0),
                "thunderstorm_flag": int(thunderstorm),
                "precipitation_flag": int(weather_severity > 1),
                "low_visibility_flag": int(low_visibility),
                "wind_gust_flag": int(wind_gust),
                "wind_speed": 15.0 + (15 if wind_gust else 0),
                "high_wind_flag": int(wind_gust),
                "rolling_avg_dep_delay": 5.0 + weather_severity * 2,
                "prev_tail_dep_delay": float(prev_delay),
            }

            if predictor is not None:
                result = predictor.predict_single(features)
                prob = result["delay_probability"]
            else:
                # Heuristic fallback
                base_prob = 1 - ROUTE_CONFIG[route]["base_otp"]
                base_prob += weather_severity * 0.025
                base_prob += int(thunderstorm) * 0.12
                base_prob += int(dep_hour >= 16) * 0.06
                base_prob += int(day_of_week >= 4) * 0.04
                base_prob += prev_delay * 0.002
                prob = min(max(base_prob, 0.05), 0.95)
                result = {"delay_probability": prob, "prediction": int(prob >= 0.5),
                          "risk_level": "High" if prob >= 0.55 else "Medium" if prob >= 0.30 else "Low"}

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "Delay Probability", "font": {"color": "#FFFFFF", "size": 16}},
                number={"suffix": "%", "font": {"color": SPIRIT_YELLOW, "size": 36}},
                delta={"reference": 27, "valueformat": ".1f",
                       "suffix": "pp vs avg", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#AAAAAA"},
                    "bar": {"color": SPIRIT_YELLOW, "thickness": 0.25},
                    "bgcolor": "#2A2A2A",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#00CC66"},
                        {"range": [30, 55], "color": "#FFB347"},
                        {"range": [55, 100], "color": "#FF4444"},
                    ],
                    "threshold": {
                        "line": {"color": SPIRIT_YELLOW, "width": 4},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor=SPIRIT_DARK,
                height=320,
                margin=dict(t=30, b=10, l=10, r=10),
                font=dict(color="#FFFFFF"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            risk_colors = {"Low": "#00CC66", "Medium": "#FFB347", "High": "#FF4444"}
            risk = result["risk_level"]
            st.markdown(
                f'<div style="text-align:center; font-size:1.5rem; font-weight:bold; '
                f'color:{risk_colors[risk]};">Risk Level: {risk}</div>',
                unsafe_allow_html=True,
            )

            if predictor is None:
                st.info("Showing heuristic estimate. Run `python main.py train-models` for ML predictions.")
            else:
                st.success("XGBoost model active — AUC-ROC 0.817, 43 rotation-aware features.")

    # -- Feature Importance --
    st.markdown("---")
    st.markdown("### Feature Importance (SHAP-like)")

    if predictor is not None:
        fi = predictor.get_feature_importance().head(15)
        fig_fi = px.bar(
            fi,
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0, SPIRIT_GREY], [1, SPIRIT_YELLOW]],
        )
    else:
        # Synthetic feature importance for display
        fi_demo = pd.DataFrame({
            "feature": [
                "weather_severity", "rolling_avg_dep_delay", "prev_tail_dep_delay",
                "route_otp_baseline", "hour_of_day", "thunderstorm_flag",
                "month", "is_hurricane_season", "is_high_congestion_dest",
                "low_visibility_flag", "is_evening", "is_holiday_window",
                "day_of_week", "route_distance", "wind_gust_flag",
            ],
            "importance": [
                0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04,
                0.03, 0.025, 0.02, 0.018, 0.015, 0.012,
            ],
        })
        fig_fi = px.bar(
            fi_demo,
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0, SPIRIT_GREY], [1, SPIRIT_YELLOW]],
        )

    fig_fi.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SPIRIT_DARK,
        plot_bgcolor=SPIRIT_DARK,
        height=420,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(color="#FFFFFF"),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 4: Scenario Simulator
# ---------------------------------------------------------------------------

def page_scenario_simulator() -> None:
    """Render the Monte Carlo Scenario Simulator page."""
    st.markdown('<div class="section-header">Capacity Scenario Simulator — Monte Carlo</div>',
                unsafe_allow_html=True)

    simulator = get_simulator()

    # -- Sidebar inputs --
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Scenario Inputs")
        route = st.selectbox("Route", list(ROUTE_CONFIG.keys()), key="sim_route")
        additional_flights = st.slider(
            "Additional Daily Flights",
            min_value=-2, max_value=4, value=1, key="sim_flights",
        )
        sim_runs = st.select_slider(
            "Simulation Runs",
            options=[1_000, 5_000, 10_000, 25_000],
            value=10_000,
            key="sim_runs",
        )
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if "scenario_results" not in st.session_state:
        st.session_state["scenario_results"] = None
        st.session_state["scenario_comparison"] = None

    if run_btn:
        with st.spinner("Running Monte Carlo simulation …"):
            comparison, results = simulator.compare_scenarios(
                route=route,
                additional_daily_flights=additional_flights,
                simulation_runs=sim_runs,
            )
            st.session_state["scenario_results"] = results
            st.session_state["scenario_comparison"] = comparison
            st.session_state["last_sim_route"] = route
            st.session_state["last_sim_flights"] = additional_flights

    if st.session_state["scenario_results"] is None:
        st.info(
            "Configure your scenario in the sidebar and click **Run Simulation**.\n\n"
            "**Example:** +1 daily flight on FLL-ATL — what happens to load factor, OTP, and revenue?"
        )
        # Show a demo
        _render_demo_scenario()
        return

    results = st.session_state["scenario_results"]
    comparison = st.session_state["scenario_comparison"]
    sim_route = st.session_state.get("last_sim_route", route)
    sim_flights = st.session_state.get("last_sim_flights", additional_flights)

    # -- Scenario narrative --
    narrative = simulator.generate_narrative(comparison, sim_route, sim_flights)
    st.code(narrative, language=None)

    # -- Comparison table --
    st.markdown("### Scenario Comparison Table")
    display_cols = [
        "Scenario", "Daily_Flights_Before", "Daily_Flights_After",
        "Baseline_LF_%", "Projected_LF_%", "LF_Change_pp",
        "Baseline_OTP_%", "Projected_OTP_%", "OTP_Change_pp",
        "Annual_Revenue_Delta_M$",
    ]
    st.dataframe(
        comparison[display_cols].style.background_gradient(
            subset=["Annual_Revenue_Delta_M$"], cmap="RdYlGn"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # -- Monte Carlo distributions --
    st.markdown("### Monte Carlo Distributions")

    tab_lf, tab_otp, tab_rev = st.tabs(["Load Factor", "OTP", "Revenue Impact"])

    colors = [SPIRIT_YELLOW, "#00CC66", "#FF6B35"]
    labels = ["All-Day", "Peak Only", "Off-Peak"]

    with tab_lf:
        fig_lf = go.Figure()
        for res, color, label in zip(results, colors, labels):
            fig_lf.add_trace(go.Histogram(
                x=res.lf_distribution * 100,
                name=label,
                opacity=0.65,
                marker_color=color,
                nbinsx=60,
                hovertemplate=f"<b>{label}</b><br>LF: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>",
            ))
        fig_lf.add_vline(x=results[0].baseline_lf * 100, line_dash="dash",
                          line_color="#FFFFFF", annotation_text=f"Baseline {results[0].baseline_lf*100:.1f}%")
        fig_lf.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            paper_bgcolor=SPIRIT_DARK, plot_bgcolor=SPIRIT_DARK,
            xaxis_title="Load Factor (%)", yaxis_title="Frequency",
            height=350, margin=dict(t=20, b=20),
            font=dict(color="#FFFFFF"),
        )
        st.plotly_chart(fig_lf, use_container_width=True)

    with tab_otp:
        fig_otp = go.Figure()
        for res, color, label in zip(results, colors, labels):
            fig_otp.add_trace(go.Histogram(
                x=res.otp_distribution * 100,
                name=label,
                opacity=0.65,
                marker_color=color,
                nbinsx=60,
            ))
        fig_otp.add_vline(x=results[0].baseline_otp * 100, line_dash="dash",
                           line_color="#FFFFFF",
                           annotation_text=f"Baseline {results[0].baseline_otp*100:.1f}%")
        fig_otp.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            paper_bgcolor=SPIRIT_DARK, plot_bgcolor=SPIRIT_DARK,
            xaxis_title="OTP (%)", yaxis_title="Frequency",
            height=350, margin=dict(t=20, b=20),
            font=dict(color="#FFFFFF"),
        )
        st.plotly_chart(fig_otp, use_container_width=True)

    with tab_rev:
        fig_rev = go.Figure()
        for res, color, label in zip(results, colors, labels):
            fig_rev.add_trace(go.Histogram(
                x=res.revenue_distribution / 1e6,
                name=label,
                opacity=0.65,
                marker_color=color,
                nbinsx=60,
            ))
        fig_rev.add_vline(x=0, line_dash="dash", line_color="#FF4444",
                           annotation_text="Breakeven")
        fig_rev.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            paper_bgcolor=SPIRIT_DARK, plot_bgcolor=SPIRIT_DARK,
            xaxis_title="Annual Revenue Delta ($M)", yaxis_title="Frequency",
            height=350, margin=dict(t=20, b=20),
            font=dict(color="#FFFFFF"),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # -- Recommendation boxes --
    st.markdown("### Recommendations")
    for res in results:
        verdict = "RECOMMENDED" if "RECOMMENDED" in res.recommendation else \
                  "CAUTION" if "CAUTION" in res.recommendation else "NOT RECOMMENDED"
        color = "#00CC66" if verdict == "RECOMMENDED" else \
                "#FFB347" if verdict == "CAUTION" else "#FF4444"
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding: 10px; '
            f'margin-bottom: 8px; background: #2A2A2A; border-radius: 4px;">'
            f'<b style="color:{color};">{verdict}</b><br>'
            f'<small style="color:#CCCCCC;">{res.recommendation}</small></div>',
            unsafe_allow_html=True,
        )


def _render_demo_scenario() -> None:
    """Render a static demo scenario for FLL-ATL +1 flight."""
    st.markdown("### Demo: FLL-ATL +1 Daily Flight")

    sim = get_simulator()
    comp, results = sim.compare_scenarios("FLL-ATL", 1, 5000)

    fig = go.Figure()
    for res, color, label in zip(results, [SPIRIT_YELLOW, "#00CC66", "#FF6B35"],
                                   ["All-Day", "Peak Only", "Off-Peak"]):
        fig.add_trace(go.Histogram(
            x=res.lf_distribution * 100,
            name=label,
            opacity=0.70,
            marker_color=color,
            nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay", template=PLOTLY_TEMPLATE,
        paper_bgcolor=SPIRIT_DARK, plot_bgcolor=SPIRIT_DARK,
        xaxis_title="Projected Load Factor (%)",
        yaxis_title="Simulated Outcomes",
        title="FLL-ATL: Load Factor Distribution (+1 daily flight)",
        height=320, font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 5: Ask Analytics (LLM)
# ---------------------------------------------------------------------------

def page_ask_analytics() -> None:
    """Render the LLM chat analytics page."""
    st.markdown('<div class="section-header">Ask Analytics — Natural Language Query</div>',
                unsafe_allow_html=True)

    engine = get_db_engine()
    if engine is None:
        _data_unavailable_banner("Database not found. Run `python main.py run-etl` first.")
        return

    # -- Session state for chat history --
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # -- Suggested questions --
    st.markdown(
        '<div style="color:#AAAAAA; font-size:0.82rem; text-transform:uppercase; '
        'letter-spacing:1px; margin-bottom:10px;">Try a question</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "Which route has the worst OTP?",
        "What are the most common delay causes at FLL?",
        "Compare load factors for summer vs winter",
        "Show me the OTP trend over 2022-2024",
        "Which month has the most weather-related delays?",
        "How does FLL-ATL perform compared to FLL-ORD?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_question"] = suggestion
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<hr style="border-color:#2A2A2A; margin:14px 0;">',
        unsafe_allow_html=True,
    )

    # -- Chat input --
    user_question = st.chat_input("Ask anything about Spirit FLL operations …")
    pending = st.session_state.pop("pending_question", None)
    question = user_question or pending

    if question:
        with st.spinner("Analysing …"):
            result = engine.query(question)

        st.session_state["chat_history"].append({
            "question": question,
            "result": result,
        })

    # -- Render conversation --
    for exchange in reversed(st.session_state["chat_history"]):
        q = exchange["question"]
        r = exchange["result"]

        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant", avatar="✈"):
            if r["error"]:
                st.error(f"Query error: {r['error']}")
            else:
                st.markdown(
                    f'<div style="color:#E8E8E8; font-size:1rem; font-weight:600; '
                    f'margin-bottom:10px;">{r["interpretation"]}</div>',
                    unsafe_allow_html=True,
                )

                if r["data"] is not None and not r["data"].empty:
                    st.dataframe(r["data"].head(20), use_container_width=True, hide_index=True)

                with st.expander("SQL Query Used"):
                    st.code(r["sql"], language="sql")

                st.markdown(
                    f'<div style="color:#777777; font-size:0.78rem; margin-top:6px;">'
                    f'Source: <span style="color:#999999;">{r["source"].upper()}</span>'
                    f' &nbsp;·&nbsp; {r["row_count"]} rows returned</div>',
                    unsafe_allow_html=True,
                )

    # -- Clear history --
    if st.session_state["chat_history"]:
        if st.button("Clear Conversation", type="secondary"):
            st.session_state["chat_history"] = []
            if engine:
                engine.reset_conversation()
            st.rerun()


# ---------------------------------------------------------------------------
# Main app router
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    selected_page = sidebar_nav()

    # Load data
    flights = load_flight_data()
    capacity = load_capacity_data()

    if selected_page == "Overview":
        if flights is None:
            _data_unavailable_banner()
        else:
            page_overview(flights, capacity)

    elif selected_page == "Route Performance":
        if flights is None:
            _data_unavailable_banner()
        else:
            page_route_performance(flights, capacity)

    elif selected_page == "OTP Predictor":
        if flights is None:
            _data_unavailable_banner()
        else:
            page_otp_predictor(flights)

    elif selected_page == "Scenario Simulator":
        page_scenario_simulator()

    elif selected_page == "Ask Analytics":
        page_ask_analytics()


if __name__ == "__main__":
    main()
