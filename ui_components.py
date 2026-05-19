import streamlit as st
import pandas as pd

from config_loader import get_kf_node_filepaths, get_kf_combination_filepath

def render_sidebar(cfg: dict, path_names: list[str], get_kf_names, get_strategies):
    """Render the sidebar and return the selected configuration."""
    with st.sidebar:
        st.markdown(
            "<h2 style='margin-bottom:0'>🛰️ Trajectory Dashboard</h2>"
            "<p class='muted-text' style='font-size:0.8rem;margin-top:4px'>"
            "Ground Truth  ·  Kalman Filter  ·  Analysis</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Theme & Compare Mode
        col1, col2 = st.columns(2)
        with col1:
            dark_mode = st.toggle("🌙  Dark Mode", value=False)
        with col2:
            compare_mode = st.toggle("⚖️  Compare", value=False)
        st.divider()

        selected_path = st.selectbox("📍  Select Path", path_names)

        kf_names = get_kf_names(cfg, selected_path)
        selected_kf = st.selectbox("📡  Select KF Variant", kf_names)

        strategies = get_strategies(cfg, selected_path, selected_kf)
        selected_strategy = st.selectbox("🧠  Select Strategy 1", strategies)
        
        selected_strategy_2 = None
        if compare_mode:
            selected_strategy_2 = st.selectbox("🧠  Select Strategy 2", strategies)
        
        # Figure out available estimators for multiselect
        node_paths = get_kf_node_filepaths(cfg, selected_path, selected_kf, selected_strategy)
        comb_path = get_kf_combination_filepath(cfg, selected_path, selected_kf, selected_strategy)
        estimator_options = ["Ground Truth"] + list(node_paths.keys())
        if comb_path:
            estimator_options.append("Combination")

        st.divider()
        st.markdown("### 📊  Visualization Mode")
        
        viz_mode = st.radio(
            "Select Display Style",
            ["Plot in same plot", "Plot all separately"]
        )
        
        selected_estimators = st.multiselect(
            f"Select what to plot ({selected_strategy})",
            options=estimator_options,
            default=["Ground Truth"] + (["Combination"] if comb_path else list(node_paths.keys())),
            key="multiselect_estimators_1"
        )

        selected_estimators_2 = None
        if compare_mode and selected_strategy_2:
            node_paths_2 = get_kf_node_filepaths(cfg, selected_path, selected_kf, selected_strategy_2)
            comb_path_2 = get_kf_combination_filepath(cfg, selected_path, selected_kf, selected_strategy_2)
            estimator_options_2 = ["Ground Truth"] + list(node_paths_2.keys())
            if comb_path_2:
                estimator_options_2.append("Combination")
                
            selected_estimators_2 = st.multiselect(
                f"Select what to plot ({selected_strategy_2})",
                options=estimator_options_2,
                default=["Ground Truth"] + (["Combination"] if comb_path_2 else list(node_paths_2.keys())),
                key="multiselect_estimators_2"
            )

        st.divider()
        st.markdown(
            "<p class='muted-text' style='font-size:0.78rem'>"
            "Coordinate conversion uses Earth-radius tangent-plane approx.<br>"
            "R = 6 371 000 m<br>"
            "Accurate for small–medium trajectories.</p>",
            unsafe_allow_html=True,
        )
        
    return {
        "dark_mode": dark_mode,
        "compare_mode": compare_mode,
        "path": selected_path,
        "kf": selected_kf,
        "strategy": selected_strategy,
        "strategy_2": selected_strategy_2,
        "viz_mode": viz_mode,
        "selected_estimators": selected_estimators,
        "selected_estimators_2": selected_estimators_2
    }


def render_metrics_summary(metrics_dict: dict, dark_mode: bool, title_suffix: str = ""):
    """Render a summary table of MSE and RMSE for all estimators."""
    st.markdown(f"<div class='sect-hdr'>📈  Performance Metrics {title_suffix}</div>", unsafe_allow_html=True)
    
    # Build a DataFrame for the metrics
    data = []
    for name, mtx in metrics_dict.items():
        data.append({
            "Estimator": name,
            "MSE (m²)": mtx.mse,
            "RMSE (m)": mtx.rmse
        })
        
    df_metrics = pd.DataFrame(data)
    
    # Use a more readable colormap for light mode (Green-Blue) vs dark mode (Yellow-Orange-Red)
    cmap = "YlOrRd" if dark_mode else "GnBu"
    
    # Render using pandas styler HTML so we can control the background/text color
    styler = df_metrics.style.format({
        "MSE (m²)": "{:.4f}",
        "RMSE (m)": "{:.4f}"
    }).background_gradient(subset=["MSE (m²)", "RMSE (m)"], cmap=cmap).hide(axis="index")
    
    st.markdown(f"<div class='custom-table'>{styler.to_html()}</div>", unsafe_allow_html=True)


def render_raw_data_tabs(df_dict: dict, title_suffix: str = ""):
    """Render raw data tables in tabs for each estimator."""
    with st.expander(f"📋  Raw Statistics Table {title_suffix}".strip()):
        tabs = st.tabs(list(df_dict.keys()))
        
        for tab, (name, df) in zip(tabs, df_dict.items()):
            with tab:
                cols_display = {
                    "sequence_no" : "Sequence",
                    "gt_x"        : "GT X (m)",
                    "gt_y"        : "GT Y (m)",
                    "kf_x"        : "KF X (m)",
                    "kf_y"        : "KF Y (m)",
                    "error"       : "Error (m)",
                }
                display_df = df[[c for c in cols_display if c in df.columns]].rename(columns=cols_display)
                cmap = "YlOrRd" if st.session_state.get("dark_mode", False) else "GnBu"
                styler = display_df.style.format(precision=5).background_gradient(subset=["Error (m)"], cmap=cmap).hide(axis="index")
                st.markdown(f"<div class='custom-table scrollable-table'>{styler.to_html()}</div>", unsafe_allow_html=True)

                csv_bytes = display_df.to_csv(index=False).encode()
                st.download_button(
                    label     = f"⬇️  Download {name} CSV",
                    data      = csv_bytes,
                    file_name = f"stats_{name}{title_suffix}.csv".replace(" ", "_"),
                    mime      = "text/csv",
                    key       = f"download_{name}_{title_suffix}"
                )
