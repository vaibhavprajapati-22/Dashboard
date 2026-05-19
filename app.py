"""
app.py
------
Main Streamlit entry point for the Trajectory Analysis Dashboard.

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title = "Trajectory Analysis Dashboard",
    page_icon  = "🛰️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from config_loader import (
    load_config, get_path_names, get_kf_names, get_strategies,
    get_gt_filepath, get_kf_node_filepaths, get_kf_combination_filepath, get_reference, get_path_velocity
)
from data_loader import load_ground_truth, load_kf_output, align_and_convert
from metrics     import compute_metrics
from plotting    import (
    plot_overlay, plot_side_by_side,
    plot_error, plot_error_histogram,
    plot_rmse_variation,
    plot_mse_rmse_bar,
    export_config,
)
from ui_components import render_sidebar, render_metrics_summary, render_raw_data_tabs


# ── Load config & Render Sidebar ──────────────────────────────────────────────

try:
    cfg = load_config("config.yaml")
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

path_names = get_path_names(cfg)
ui_state = render_sidebar(cfg, path_names, get_kf_names, get_strategies)

dark_mode = ui_state["dark_mode"]
selected_path = ui_state["path"]
selected_kf = ui_state["kf"]
selected_strategy = ui_state["strategy"]
viz_mode = ui_state["viz_mode"]
selected_estimators = ui_state["selected_estimators"]


# ── Dynamic CSS (theme-aware) ─────────────────────────────────────────────────

if dark_mode:
    BG        = "#0e1117"
    CARD_BG   = "#151c2c"
    BORDER    = "#1e2940"
    TEXT      = "#e2e8f0"
    MUTED     = "#94a3b8"
    ACCENT    = "#38bdf8"
    SIDEBAR   = "#0b0f1a"
    VAL_COLOR = "#38bdf8"
else:
    BG        = "#f8fafc"
    CARD_BG   = "#ffffff"
    BORDER    = "#e2e8f0"
    TEXT      = "#0f172a"
    MUTED     = "#475569"
    ACCENT    = "#0ea5e9"
    SIDEBAR   = "#ffffff"
    VAL_COLOR = "#0ea5e9"

st.markdown(f"""
<style>
/* ── Global ── */
html, body, [class*="css"] {{ font-family: "Inter", sans-serif; }}
.stApp                      {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] > div:first-child {{
    background-color: {SIDEBAR};
}}

/* General text reset to prevent dark mode bleed */
/* Force visibility on all common Streamlit text containers */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown li, .stMarkdown label,
.stMarkdown b, .stMarkdown strong,
.stText, .stText p, .stText span,
[data-testid="stMarkdownContainer"] p, 
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stHeader"] *,
[data-testid="stSidebar"] *,
.stSelectbox label p,
.stMultiSelect label p,
.stSlider label p,
.stRadio label p {{
    color: {TEXT} !important;
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stHeader"] {{
    color: {TEXT} !important;
    font-weight: 700 !important;
}}

/* Sidebar headers */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {{
    color: {TEXT} !important;
}}

/* ── Metric card ── */
.metric-card {{
    background    : {CARD_BG};
    border        : 1px solid {BORDER};
    border-top    : 3px solid {ACCENT};
    border-radius : 12px;
    padding       : 20px 16px 18px;
    text-align    : center;
    transition    : all .2s ease-in-out;
    box-shadow    : 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}}
.metric-card:hover {{ 
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
}}
.metric-label  {{
    font-size    : 0.73rem;
    font-weight  : 600;
    color        : {MUTED} !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 8px;
}}
.metric-value  {{
    font-size  : 2rem;
    font-weight: 800;
    color      : {VAL_COLOR} !important;
    line-height: 1.15;
}}
.metric-unit   {{
    font-size: 0.72rem;
    color    : {MUTED} !important;
    margin-top: 4px;
}}

/* ── Section header ── */
.sect-hdr {{
    font-size    : 1rem;
    font-weight  : 700;
    color        : {ACCENT} !important;
    border-bottom: 2px solid {ACCENT}22;
    padding-bottom: 8px;
    margin       : 32px 0 20px;
    letter-spacing: .02em;
    display: flex;
    align-items: center;
}}

/* ── Info badge ── */
.info-badge {{
    display      : inline-block;
    background   : {CARD_BG};
    border       : 1px solid {BORDER};
    border-radius: 6px;
    padding      : 4px 12px;
    font-size    : 0.8rem;
    color        : {MUTED} !important;
    margin-right : 8px;
}}

/* ── Scrollable stats table ── */
.stats-table {{ max-height: 320px; overflow-y: auto; }}

/* ── Plotly toolbar tweak ── */
.modebar-container {{ opacity: 0.4; transition: opacity .2s; }}
.modebar-container:hover {{ opacity: 1; }}

/* override streamlit default padding */
.block-container {{ padding-top: 1.5rem !important; }}

/* ── Muted text ── */
.muted-text {{ color: {MUTED} !important; }}

/* ── Streamlit Widget Overrides ── */
/* Input Backgrounds */
div[data-baseweb="select"] > div, 
.stMultiSelect div[data-baseweb="select"] > div {{
    background-color: {CARD_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
div[data-baseweb="select"] > div:hover,
.stMultiSelect div[data-baseweb="select"] > div:hover {{
    background-color: {BG} !important;
}}
span[data-baseweb="tag"] {{
    background-color: {ACCENT} !important;
    color: #ffffff !important;
}}
span[data-baseweb="tag"] span {{
    color: #ffffff !important;
}}
div[data-baseweb="popover"] > div {{
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
}}
ul[data-baseweb="menu"] {{ background-color: {CARD_BG} !important; }}
li[data-baseweb="menu-item"] {{ color: {TEXT} !important; background-color: transparent !important; }}
li[data-baseweb="menu-item"]:hover {{ background-color: {BG} !important; }}

/* Labels and Checkbox/Toggle/Radio Text */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
[data-baseweb="radio"] p,
[data-baseweb="radio"] span,
[data-baseweb="radio"] div,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span {{
    color: {TEXT} !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
    background-color: {CARD_BG} !important;
    border-color: {BORDER} !important;
}}
[data-testid="stExpander"] summary {{
    background-color: {CARD_BG} !important;
}}
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {{
    color: {TEXT} !important;
}}
[data-testid="stExpanderDetails"] {{
    background-color: {BG} !important;
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    background-color: transparent !important;
}}
[data-baseweb="tab"] p, [data-baseweb="tab"] span {{
    color: {MUTED} !important;
}}
[aria-selected="true"] p, [aria-selected="true"] span {{
    color: {TEXT} !important;
    font-weight: 600 !important;
}}
[data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
}}

/* Buttons */
[data-testid*="baseButton-secondary"], [data-testid*="stBaseButton-secondary"], button[kind="secondary"] {{
    background-color: {CARD_BG} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}
[data-testid*="baseButton-secondary"] p, [data-testid*="baseButton-secondary"] span,
[data-testid*="stBaseButton-secondary"] p, [data-testid*="stBaseButton-secondary"] span,
button[kind="secondary"] p, button[kind="secondary"] span,
button[kind="secondary"] div {{
    color: {TEXT} !important;
}}
[data-testid*="baseButton-secondary"]:hover, [data-testid*="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background-color: {BG} !important;
}}
[data-testid*="baseButton-secondary"]:hover *, [data-testid*="stBaseButton-secondary"]:hover *, button[kind="secondary"]:hover * {{
    color: {ACCENT} !important;
}}

/* Top Header Bar */
header[data-testid="stHeader"] {{
    background-color: {BG} !important;
}}
header[data-testid="stHeader"] * {{
    color: {TEXT} !important;
    fill: {TEXT} !important;
}}

/* Alerts (Info/Error/Warning) */
[data-testid="stAlert"] {{
    background-color: {CARD_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{
    color: {TEXT} !important;
}}

/* ── Custom HTML Tables ── */
.scrollable-table {{
    max-height: 340px;
    overflow-y: auto;
    display: block;
}}
.custom-table table {{
    width: 100%;
    border-collapse: collapse;
    background-color: {CARD_BG};
    color: {TEXT};
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.9rem;
}}
.custom-table th, .custom-table td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid {BORDER};
}}
.custom-table th {{
    background-color: {BG};
    color: {TEXT} !important;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}}
.custom-table td {{
    color: {TEXT};
}}
</style>
""", unsafe_allow_html=True)


# ── Load & process data ───────────────────────────────────────────────────────

lat_ref, lon_ref = get_reference(cfg)
gt_path  = get_gt_filepath(cfg, selected_path)

def load_strategy_data(strategy_name):
    node_paths = get_kf_node_filepaths(cfg, selected_path, selected_kf, strategy_name)
    comb_path = get_kf_combination_filepath(cfg, selected_path, selected_kf, strategy_name)
    
    df_dict = {}
    metrics_dict = {}
    
    try:
        gt_df = load_ground_truth(gt_path)
        for node_name, path in node_paths.items():
            kf_df = load_kf_output(path)
            merged = align_and_convert(gt_df, kf_df, lat_ref, lon_ref)
            df, mtx = compute_metrics(merged)
            df_dict[node_name] = df
            metrics_dict[node_name] = mtx
            
        if comb_path:
            kf_df = load_kf_output(comb_path)
            merged = align_and_convert(gt_df, kf_df, lat_ref, lon_ref)
            df, mtx = compute_metrics(merged)
            df_dict["Combination"] = df
            metrics_dict["Combination"] = mtx
            
        return df_dict, metrics_dict, True, gt_df, ""
    except (FileNotFoundError, ValueError) as exc:
        return {}, {}, False, None, str(exc)

with st.spinner("Loading data …"):
    df_dict_1, metrics_dict_1, data_ok_1, gt_df_1, err_1 = load_strategy_data(selected_strategy)
    
    compare_mode = ui_state.get("compare_mode", False)
    strategy_2 = ui_state.get("strategy_2")
    
    if compare_mode and strategy_2:
        df_dict_2, metrics_dict_2, data_ok_2, gt_df_2, err_2 = load_strategy_data(strategy_2)
    else:
        data_ok_2 = False

if not data_ok_1:
    st.error(f"**Data error (Strategy 1):** {err_1}")
    st.info("Run `python generate_sample_data.py` in the project folder to create the sample CSV files, then refresh.")

if compare_mode and strategy_2 and not data_ok_2:
    st.error(f"**Data error (Strategy 2):** {err_2}")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:2px'>🛰️  Trajectory Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)


# ── Performance Comparison Section (at the top) ─────────────────────────────

st.markdown(
    "<div class='sect-hdr'>📊  Performance Comparison</div>",
    unsafe_allow_html=True,
)


def _get_representative_metric(cfg, path, kf, strategy, lat_ref, lon_ref, target_source=None):
    """Load data for a strategy and return (TrajectoryMetrics, source_label).

    Priority: target_source → Combination file → first node.
    Returns (None, None) on any error.
    """
    gt_path_local = get_gt_filepath(cfg, path)
    try:
        gt_df_local = load_ground_truth(gt_path_local)

        # 1. Try Target Source (if specified)
        if target_source:
            if target_source == "Combination":
                comb_path = get_kf_combination_filepath(cfg, path, kf, strategy)
                if comb_path:
                    kf_df = load_kf_output(comb_path)
                    merged = align_and_convert(gt_df_local, kf_df, lat_ref, lon_ref)
                    _, mtx = compute_metrics(merged)
                    return mtx, "Combination"
            else:
                node_paths = get_kf_node_filepaths(cfg, path, kf, strategy)
                if target_source in node_paths:
                    kf_df = load_kf_output(node_paths[target_source])
                    merged = align_and_convert(gt_df_local, kf_df, lat_ref, lon_ref)
                    _, mtx = compute_metrics(merged)
                    return mtx, target_source

        # 2. Fallback to Combination
        comb_path = get_kf_combination_filepath(cfg, path, kf, strategy)
        if comb_path:
            kf_df = load_kf_output(comb_path)
            merged = align_and_convert(gt_df_local, kf_df, lat_ref, lon_ref)
            _, mtx = compute_metrics(merged)
            return mtx, "Combination"

        # 3. Fallback to first available node
        node_paths = get_kf_node_filepaths(cfg, path, kf, strategy)
        if node_paths:
            first_name = next(iter(node_paths.keys()))
            first_path = next(iter(node_paths.values()))
            kf_df = load_kf_output(first_path)
            merged = align_and_convert(gt_df_local, kf_df, lat_ref, lon_ref)
            _, mtx = compute_metrics(merged)
            return mtx, first_name
    except Exception:
        return None, None
    return None, None


# --- Source Discovery Logic ---
all_possible_sources = set()

# Check all KF variants for the selected path
all_kfs_for_path = get_kf_names(cfg, selected_path)
for kf in all_kfs_for_path:
    strats = get_strategies(cfg, selected_path, kf)
    for s in strats:
        node_paths = get_kf_node_filepaths(cfg, selected_path, kf, s)
        all_possible_sources.update(node_paths.keys())
        if get_kf_combination_filepath(cfg, selected_path, kf, s):
            all_possible_sources.add("Combination")

source_options = sorted(list(all_possible_sources))
# Ensure "Combination" is first if it exists
if "Combination" in source_options:
    source_options.remove("Combination")
    source_options = ["Combination"] + source_options

c1, c2 = st.columns([1, 2])
with c1:
    rep_source = st.selectbox(
        "Select Representative Source",
        options=source_options,
        index=0,
        help="Choose which node or combination to use for the comparison charts and tables."
    )
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin-top:5px' class='muted-text'>Comparing data from <b>{rep_source}</b> (with automatic fallback if unavailable).</div>", 
        unsafe_allow_html=True
    )


comp_tab1, comp_tab2 = st.tabs([
    "📐 KF Variant Fixed — Compare Strategies",
    "🔀 Strategy Fixed — Compare KF Variants",
])

# ── Tab 1: Fixed KF Variant, compare all strategies ──────────────────────────
with comp_tab1:
    st.markdown(
        f"Showing **MSE** and **RMSE** for every strategy under "
        f"**{selected_path}** · **{selected_kf}**"
    )

    strategies_for_kf = get_strategies(cfg, selected_path, selected_kf)
    strategy_metrics: dict = {}
    strategy_sources: dict = {}

    with st.spinner("Computing strategy metrics …"):
        for strat in strategies_for_kf:
            mtx, src = _get_representative_metric(
                cfg, selected_path, selected_kf, strat, lat_ref, lon_ref,
                target_source=rep_source
            )
            if mtx is not None:
                strategy_metrics[strat] = mtx
                strategy_sources[strat] = src

    if strategy_metrics:
        fig_strat = plot_mse_rmse_bar(
            strategy_metrics,
            title=f"MSE & RMSE · {selected_path} · {selected_kf}",
            x_label="Strategy",
            dark=dark_mode,
        )
        st.plotly_chart(
            fig_strat,
            width='stretch',
            theme=None,
            config=export_config(selected_path, selected_kf, "strategy_comparison"),
        )

        # Numeric summary table
        rows_html = "".join(
            f"<tr>"
            f"<td>{s}</td>"
            f"<td>{strategy_sources[s]}</td>"
            f"<td>{m.mse:.6f}</td>"
            f"<td>{m.rmse:.6f}</td>"
            f"<td>{m.mean_err:.4f}</td>"
            f"<td>{m.n_points}</td>"
            f"</tr>"
            for s, m in strategy_metrics.items()
        )
        st.markdown(
            f"""
            <div class='custom-table scrollable-table'>
            <table>
                <thead><tr>
                    <th>Strategy</th>
                    <th>Source</th>
                    <th>MSE (m²)</th>
                    <th>RMSE (m)</th>
                    <th>Mean Err (m)</th>
                    <th>Points</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No data could be loaded for any strategy under the current selection.")

# ── Tab 2: Fixed Strategy, compare all KF variants ───────────────────────────
with comp_tab2:
    st.markdown(
        f"Showing **MSE** and **RMSE** for every KF variant under "
        f"**{selected_path}** · Strategy **{selected_strategy}**"
    )

    kf_names_for_path = get_kf_names(cfg, selected_path)
    kf_metrics: dict = {}
    kf_sources: dict = {}

    with st.spinner("Computing KF variant metrics …"):
        for kf_name in kf_names_for_path:
            # The chosen strategy may not exist for every KF variant – skip gracefully
            kf_strategies = get_strategies(cfg, selected_path, kf_name)
            strat_to_use = (
                selected_strategy
                if selected_strategy in kf_strategies
                else (kf_strategies[0] if kf_strategies else None)
            )
            if strat_to_use is None:
                continue
            mtx, src = _get_representative_metric(
                cfg, selected_path, kf_name, strat_to_use, lat_ref, lon_ref,
                target_source=rep_source
            )
            if mtx is not None:
                kf_metrics[kf_name] = mtx
                kf_sources[kf_name] = src

    if kf_metrics:
        fig_kf = plot_mse_rmse_bar(
            kf_metrics,
            title=f"MSE & RMSE · {selected_path} · {selected_strategy}",
            x_label="KF Variant",
            dark=dark_mode,
        )
        st.plotly_chart(
            fig_kf,
            width='stretch',
            theme=None,
            config=export_config(selected_path, selected_strategy, "kf_comparison"),
        )

        # Numeric summary table
        rows_html_kf = "".join(
            f"<tr>"
            f"<td>{k}</td>"
            f"<td>{kf_sources[k]}</td>"
            f"<td>{m.mse:.6f}</td>"
            f"<td>{m.rmse:.6f}</td>"
            f"<td>{m.mean_err:.4f}</td>"
            f"<td>{m.n_points}</td>"
            f"</tr>"
            for k, m in kf_metrics.items()
        )
        st.markdown(
            f"""
            <div class='custom-table scrollable-table'>
            <table>
                <thead><tr>
                    <th>KF Variant</th>
                    <th>Source</th>
                    <th>MSE (m²)</th>
                    <th>RMSE (m)</th>
                    <th>Mean Err (m)</th>
                    <th>Points</th>
                </tr></thead>
                <tbody>{rows_html_kf}</tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No data could be loaded for any KF variant under the current selection.")

st.markdown("---")

def render_strategy_view(strategy_name, df_dict, metrics_dict, selected_estimators, gt_df, title_suffix=""):
    first_mtx = next(iter(metrics_dict.values())) if metrics_dict else None
    if first_mtx:
        st.markdown(
            f"<span class='info-badge'>📍 {selected_path}</span>"
            f"<span class='info-badge'>💨 {get_path_velocity(cfg, selected_path)} km/h</span>"
            f"<span class='info-badge'>📡 {selected_kf}</span>"
            f"<span class='info-badge'>🧠 {strategy_name}</span>"
            f"<span class='info-badge'>🔗 {first_mtx.n_points} aligned points</span>"
            f"<span class='info-badge'>🌐 ref ({lat_ref}°, {lon_ref}°)</span>",
            unsafe_allow_html=True,
        )
        
    render_metrics_summary(metrics_dict, dark_mode, title_suffix=title_suffix)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='sect-hdr'>🗺️  Trajectory Visualization</div>", unsafe_allow_html=True)

    if viz_mode == "Plot in same plot":
        if not selected_estimators:
            st.info("👈  Select at least one estimator to plot in the sidebar.")
        else:
            show_gt = "Ground Truth" in selected_estimators
            filtered_df_dict = {k: v for k, v in df_dict.items() if k in selected_estimators}
            plot_gt_df = list(df_dict.values())[0] if df_dict else None
            
            fig = plot_overlay(
                filtered_df_dict, dark=dark_mode, path_name=selected_path, 
                kf_name=selected_kf, show_gt=show_gt, gt_df=plot_gt_df
            )
            st.plotly_chart(fig, width='stretch', theme=None, config=export_config(selected_path, selected_kf, f"overlay_{strategy_name}"))

    elif viz_mode == "Plot all separately":
        if not selected_estimators:
            st.info("👈  Select at least one estimator to plot in the sidebar.")
        else:
            plot_gt_df = list(df_dict.values())[0] if df_dict else None
            for estimator in selected_estimators:
                st.markdown(f"#### 🔹 {estimator}")
                
                if estimator == "Ground Truth":
                    fig = plot_overlay({}, dark=dark_mode, path_name=selected_path, kf_name="N/A", show_gt=True, gt_df=plot_gt_df)
                    st.plotly_chart(fig, width='stretch', theme=None, config=export_config(selected_path, selected_kf, f"sep_gt_{strategy_name}"))
                elif estimator in df_dict:
                    df = df_dict[estimator]
                    fig = plot_overlay(
                        {estimator: df}, dark=dark_mode, path_name=selected_path, 
                        kf_name=f"{selected_kf} ({estimator})", show_gt=True, gt_df=plot_gt_df
                    )
                    st.plotly_chart(fig, width='stretch', theme=None, config=export_config(selected_path, selected_kf, f"sep_overlay_{estimator}_{strategy_name}"))
                
                st.divider()

    st.markdown("<div class='sect-hdr'>📉  RMSE Variation</div>", unsafe_allow_html=True)
    if not selected_estimators:
        st.info("👈  Select at least one estimator to plot in the sidebar.")
    else:
        filtered_df_dict = {k: v for k, v in df_dict.items() if k in selected_estimators}
        fig_rmse = plot_rmse_variation(filtered_df_dict, dark=dark_mode)
        st.plotly_chart(fig_rmse, width='stretch', theme=None, config=export_config(selected_path, selected_kf, f"rmse_{strategy_name}"))

    render_raw_data_tabs(df_dict, title_suffix=title_suffix)


if compare_mode and strategy_2:
    col1, col2 = st.columns(2)
    with col1:
        if data_ok_1:
            render_strategy_view(selected_strategy, df_dict_1, metrics_dict_1, ui_state["selected_estimators"], gt_df_1, title_suffix=f"({selected_strategy})")
    with col2:
        if data_ok_2:
            render_strategy_view(strategy_2, df_dict_2, metrics_dict_2, ui_state["selected_estimators_2"], gt_df_2, title_suffix=f"({strategy_2})")
else:
    if data_ok_1:
        render_strategy_view(selected_strategy, df_dict_1, metrics_dict_1, ui_state["selected_estimators"], gt_df_1)

