"""
plotting.py
-----------
All Plotly visualisation functions for the Trajectory Dashboard.
Each function returns a plotly.graph_objects.Figure ready to be rendered
with st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from metrics import TrajectoryMetrics

# ── Colour palette ────────────────────────────────────────────────────────────

GT_COLOR       = "#38bdf8"   # sky-blue
ERROR_COLOR    = "#fbbf24"   # amber
START_COLOR    = "#4ade80"   # green
END_COLOR      = "#f87171"   # rose
MEAN_ERR_COLOR = "#e879f9"   # fuchsia
GRID_DARK      = "#1e2940"
GRID_LIGHT     = "#e2e8f0"
BG_DARK        = "#0e1117"
PLOT_BG_DARK   = "#151c2c"
BG_LIGHT       = "#ffffff"
PLOT_BG_LIGHT  = "#f8fafc"

# Colors for multiple estimators
NODE_COLORS = ["#fb923c", "#a3e635", "#f472b6", "#818cf8", "#2dd4bf"]
COMB_COLOR  = "#ff0000" # Bright red for combination to stand out

# ── Theme helpers ─────────────────────────────────────────────────────────────

def _theme(dark: bool) -> dict:
    if dark:
        return dict(
            paper_bg  = BG_DARK,
            plot_bg   = PLOT_BG_DARK,
            font_col  = "#e2e8f0",
            grid_col  = GRID_DARK,
            zero_col  = "#2d3a55",
            legend_bg = "rgba(15,20,35,0.85)",
            legend_bc = "#2d3a55",
            sub_col   = "#94a3b8",
        )
    return dict(
        paper_bg  = BG_LIGHT,
        plot_bg   = PLOT_BG_LIGHT,
        font_col  = "#1e293b",
        grid_col  = GRID_LIGHT,
        zero_col  = "#cbd5e1",
        legend_bg = "rgba(255,255,255,0.9)",
        legend_bc = "#e2e8f0",
        sub_col   = "#64748b",
    )


def _base_axis(t: dict, title: str = "", equal_scale: bool = False) -> dict:
    ax = dict(
        title_text      = title,
        title_font      = dict(size=12, color=t["font_col"]),
        tickfont        = dict(size=11, color=t["font_col"]),
        gridcolor       = t["grid_col"],
        zerolinecolor   = t["zero_col"],
        showgrid        = True,
        zeroline        = True,
    )
    if equal_scale:
        ax["scaleanchor"] = "x"
        ax["scaleratio"]  = 1
    return ax


def _export_cfg(filename: str) -> dict:
    return {
        "toImageButtonOptions": {
            "format":   "png",
            "filename": filename,
            "scale":    2,
        },
        "displayModeBar": True,
        "displaylogo":    False,
    }


def _get_estimator_style(name: str, index: int) -> dict:
    if "combination" in name.lower():
        return dict(color=COMB_COLOR, width=3, dash="solid")
    else:
        return dict(color=NODE_COLORS[index % len(NODE_COLORS)], width=2, dash="dot")


# ── Trajectory: Overlay plot ──────────────────────────────────────────────────

def plot_overlay(
    df_dict: dict[str, pd.DataFrame],
    dark: bool = True,
    path_name: str = "",
    kf_name: str = "",
    show_gt: bool = True,
    gt_df: pd.DataFrame = None,
) -> go.Figure:
    """Single overlay plot – Ground Truth and KF on the same axes."""

    t   = _theme(dark)
    fig = go.Figure()

    # ── Ground-truth trajectory ──────────────────────────────────────────────
    if show_gt and gt_df is not None:
        cd_gt = np.stack(
            [gt_df["sequence_no"], gt_df["latitude"], gt_df["longitude"]],
            axis=-1,
        )
        fig.add_trace(go.Scatter(
            x          = gt_df["gt_x"],
            y          = gt_df["gt_y"],
            mode       = "lines",
            name       = "Ground Truth",
            line       = dict(color=GT_COLOR, width=3),
            customdata = cd_gt,
            hovertemplate = (
                "<b>Ground Truth</b><br>"
                "Seq: %{customdata[0]:.0f}<br>"
                "X: %{x:.3f} m | Y: %{y:.3f} m<br>"
                "Lat: %{customdata[1]:.7f}°<br>"
                "Lon: %{customdata[2]:.7f}°<extra></extra>"
            ),
        ))

    # ── KF trajectories ────────────────────────────────────────────────────────
    for idx, (name, df) in enumerate(df_dict.items()):
        has_err = "error" in df.columns
        cd_kf_cols = [df["sequence_no"], df["KF_lat"], df["KF_long"]]
        if has_err:
            cd_kf_cols.append(df["error"])
        cd_kf = np.stack(cd_kf_cols, axis=-1)

        hover_kf = (
            f"<b>{name}</b><br>"
            "Seq: %{customdata[0]:.0f}<br>"
            "X: %{x:.3f} m | Y: %{y:.3f} m<br>"
            "Lat: %{customdata[1]:.7f}°<br>"
            "Lon: %{customdata[2]:.7f}°"
            + ("<br>Err: %{customdata[3]:.3f} m" if has_err else "")
            + "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x          = df["kf_x"],
            y          = df["kf_y"],
            mode       = "lines",
            name       = name,
            line       = _get_estimator_style(name, idx),
            customdata = cd_kf,
            hovertemplate = hover_kf,
        ))

    # ── Start / End markers ──────────────────────────────────────────────────
    # Removed as requested

    fig.update_layout(
        title=dict(
            text=f"<b>Trajectory Overlay</b>  ·  {path_name}  ·  {kf_name}",
            font=dict(size=15, color=t["font_col"]),
            x=0.01,
        ),
        xaxis  = _base_axis(t, "X  (metres)"),
        yaxis  = _base_axis(t, "Y  (metres)", equal_scale=True),
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font=dict(color=t["font_col"], family="Inter, sans-serif"),
        legend=dict(
            bgcolor     = t["legend_bg"],
            bordercolor = t["legend_bc"],
            borderwidth = 1,
            font        = dict(size=12, color=t["font_col"]),
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        hovermode = "closest",
        height    = 520,
        margin    = dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ── Trajectory: Side-by-side ──────────────────────────────────────────────────

def plot_side_by_side(
    df_dict: dict[str, pd.DataFrame],
    dark: bool = True,
    path_name: str = "",
    kf_name: str = "",
) -> go.Figure:
    """Two-panel plot – Ground Truth left, KF estimators right."""

    t   = _theme(dark)
    first_df = next(iter(df_dict.values()))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"<b>Ground Truth</b>  ·  {path_name}",
            f"<b>KF Estimates</b>  ·  {kf_name}",
        ],
        horizontal_spacing=0.08,
    )

    # ── Panel 1: GT ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=first_df["gt_x"], y=first_df["gt_y"],
        mode="lines", name="Ground Truth",
        line=dict(color=GT_COLOR, width=3),
        hovertemplate="Seq: %{text}<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>",
        text=first_df["sequence_no"],
    ), row=1, col=1)

    # Start and End markers removed

    # ── Panel 2: KF ──────────────────────────────────────────────────────────
    for idx, (name, df) in enumerate(df_dict.items()):
        has_err = "error" in df.columns
        hover_kf = (
            "Seq: %{text}<br>X: %{x:.2f} m<br>Y: %{y:.2f} m"
            + ("<br>Err: %{customdata:.3f} m" if has_err else "")
            + "<extra></extra>"
        )
        kf_scatter = go.Scatter(
            x=df["kf_x"], y=df["kf_y"],
            mode="lines", name=name,
            line=_get_estimator_style(name, idx),
            text=df["sequence_no"],
            hovertemplate=hover_kf,
        )
        if has_err:
            kf_scatter.customdata = df["error"].to_numpy()
        fig.add_trace(kf_scatter, row=1, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    ax_style = dict(
        tickfont      = dict(size=11, color=t["font_col"]),
        gridcolor     = t["grid_col"],
        zerolinecolor = t["zero_col"],
        showgrid      = True,
    )
    fig.update_xaxes(title_text="X (metres)", **ax_style)
    fig.update_yaxes(title_text="Y (metres)", **ax_style)

    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color=t["font_col"])

    fig.update_layout(
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font=dict(color=t["font_col"], family="Inter, sans-serif"),
        legend=dict(
            bgcolor=t["legend_bg"], bordercolor=t["legend_bc"],
            borderwidth=1, font=dict(size=12, color=t["font_col"]),
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        hovermode = "closest",
        height    = 520,
        margin    = dict(l=60, r=30, t=70, b=60),
    )
    return fig


# ── Error vs Sequence ─────────────────────────────────────────────────────────

def plot_rmse_variation(
    df_dict: dict[str, pd.DataFrame],
    dark: bool = True,
) -> go.Figure:
    """
    Cumulative RMSE vs sequence number for multiple estimators.
    """
    t   = _theme(dark)
    fig = go.Figure()

    for idx, (name, df) in enumerate(df_dict.items()):
        if "error" not in df.columns:
            continue
        err  = df["error"].to_numpy()
        seqs = df["sequence_no"].to_numpy()
        style = _get_estimator_style(name, idx)

        # Cumulative RMSE
        cum_sq_err = np.cumsum(err**2)
        cum_rmse = np.sqrt(cum_sq_err / np.arange(1, len(err) + 1))

        fig.add_trace(go.Scatter(
            x    = seqs,
            y    = cum_rmse,
            mode = "lines",
            name = name,
            line = style,
            hovertemplate = f"<b>{name}</b><br>Seq %{{x:.0f}}<br>RMSE: %{{y:.4f}} m<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Cumulative RMSE Variation</b>  vs  Sequence Number",
            font=dict(size=15, color=t["font_col"]),
            x=0.01,
        ),
        xaxis  = _base_axis(t, "Sequence Number"),
        yaxis  = _base_axis(t, "RMSE  (metres)"),
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font   = dict(color=t["font_col"], family="Inter, sans-serif"),
        legend = dict(
            bgcolor=t["legend_bg"], bordercolor=t["legend_bc"],
            borderwidth=1, font=dict(size=12, color=t["font_col"]),
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        hovermode = "x unified",
        height    = 380,
        margin    = dict(l=60, r=30, t=60, b=60),
    )
    return fig


def plot_error(
    df_dict: dict[str, pd.DataFrame],
    dark: bool = True,
    metrics: dict = None,
) -> go.Figure:
    """
    Euclidean distance error vs sequence number for multiple estimators.
    """
    t   = _theme(dark)
    fig = go.Figure()

    for idx, (name, df) in enumerate(df_dict.items()):
        err  = df["error"].to_numpy()
        seqs = df["sequence_no"].to_numpy()
        style = _get_estimator_style(name, idx)

        # ── Error Line ────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x    = seqs,
            y    = err,
            mode = "lines",
            name = name,
            line = style,
            hovertemplate = f"<b>{name}</b><br>Seq %{{x:.0f}}<br>Error: %{{y:.4f}} m<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Euclidean Distance Error</b>  vs  Sequence Number",
            font=dict(size=15, color=t["font_col"]),
            x=0.01,
        ),
        xaxis  = _base_axis(t, "Sequence Number"),
        yaxis  = _base_axis(t, "Error  (metres)"),
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font   = dict(color=t["font_col"], family="Inter, sans-serif"),
        legend = dict(
            bgcolor=t["legend_bg"], bordercolor=t["legend_bc"],
            borderwidth=1, font=dict(size=12, color=t["font_col"]),
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        hovermode = "x unified",
        height    = 380,
        margin    = dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ── Error distribution histogram ──────────────────────────────────────────────

def plot_error_histogram(df_dict: dict[str, pd.DataFrame], dark: bool = True) -> go.Figure:
    """Histogram of error distribution for multiple estimators."""

    t   = _theme(dark)
    fig = go.Figure()

    for idx, (name, df) in enumerate(df_dict.items()):
        err = df["error"].to_numpy()
        style = _get_estimator_style(name, idx)
        
        fig.add_trace(go.Histogram(
            x         = err,
            nbinsx    = 40,
            name      = name,
            marker    = dict(color=style["color"], opacity=0.6, line=dict(color=t["paper_bg"], width=0.5)),
            hovertemplate = f"<b>{name}</b><br>Error range: %{{x:.3f}} m<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Error Distribution</b>",
            font=dict(size=15, color=t["font_col"]),
            x=0.01,
        ),
        xaxis  = _base_axis(t, "Error  (metres)"),
        yaxis  = _base_axis(t, "Count"),
        barmode = "overlay",
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font   = dict(color=t["font_col"], family="Inter, sans-serif"),
        legend = dict(
            bgcolor=t["legend_bg"], bordercolor=t["legend_bc"],
            borderwidth=1, font=dict(size=12, color=t["font_col"]),
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        height  = 300,
        margin  = dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ── MSE / RMSE Comparison Bar Chart ──────────────────────────────────────────

def plot_mse_rmse_bar(
    metrics_map: dict,
    title: str = "MSE & RMSE Comparison",
    x_label: str = "Strategy",
    dark: bool = True,
) -> go.Figure:
    """
    Grouped bar chart comparing MSE and RMSE across multiple strategies or KF variants.

    Parameters
    ----------
    metrics_map : dict
        {label -> TrajectoryMetrics}  – one entry per strategy or KF variant.
    title : str
        Chart title.
    x_label : str
        Label for the x-axis categories.
    dark : bool
        Dark/light theme toggle.

    Returns
    -------
    go.Figure
    """
    t = _theme(dark)

    labels = list(metrics_map.keys())
    mse_vals  = [m.mse  for m in metrics_map.values()]
    rmse_vals = [m.rmse for m in metrics_map.values()]

    MSE_COLOR  = "#818cf8"   # indigo
    RMSE_COLOR = "#fb923c"   # orange

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name          = "MSE (m²)",
        x             = labels,
        y             = mse_vals,
        marker_color  = MSE_COLOR,
        marker_opacity= 0.85,
        text          = [f"{v:.4f}" for v in mse_vals],
        textposition  = "outside",
        textfont      = dict(size=11, color=t["font_col"]),
        hovertemplate = "<b>%{x}</b><br>MSE: %{y:.6f} m²<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name          = "RMSE (m)",
        x             = labels,
        y             = rmse_vals,
        marker_color  = RMSE_COLOR,
        marker_opacity= 0.85,
        text          = [f"{v:.4f}" for v in rmse_vals],
        textposition  = "outside",
        textfont      = dict(size=11, color=t["font_col"]),
        hovertemplate = "<b>%{x}</b><br>RMSE: %{y:.6f} m<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text  = f"<b>{title}</b>",
            font  = dict(size=15, color=t["font_col"]),
            x     = 0.01,
        ),
        barmode       = "group",
        bargap        = 0.25,
        bargroupgap   = 0.08,
        xaxis = dict(
            title_text    = x_label,
            title_font    = dict(size=12, color=t["font_col"]),
            tickfont      = dict(size=11, color=t["font_col"]),
            gridcolor     = t["grid_col"],
            zerolinecolor = t["zero_col"],
            showgrid      = False,
        ),
        yaxis = dict(
            title_text    = "Error (m² / m)",
            title_font    = dict(size=12, color=t["font_col"]),
            tickfont      = dict(size=11, color=t["font_col"]),
            gridcolor     = t["grid_col"],
            zerolinecolor = t["zero_col"],
            showgrid      = True,
        ),
        plot_bgcolor  = t["plot_bg"],
        paper_bgcolor = t["paper_bg"],
        font          = dict(color=t["font_col"], family="Inter, sans-serif"),
        legend        = dict(
            bgcolor     = t["legend_bg"],
            bordercolor = t["legend_bc"],
            borderwidth = 1,
            font        = dict(size=12, color=t["font_col"]),
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1,
        ),
        hoverlabel=dict(
            font=dict(color=t["font_col"]),
            bgcolor=t["legend_bg"],
        ),
        hovermode = "x",
        height    = 420,
        margin    = dict(l=60, r=30, t=80, b=100),
    )
    return fig


# ── Export config helper ──────────────────────────────────────────────────────

def export_config(path_name: str, kf_name: str, suffix: str) -> dict:
    safe = lambda s: s.replace(" ", "_").replace("-", "_")
    return _export_cfg(f"traj_{safe(path_name)}_{safe(kf_name)}_{suffix}")
