"""
data_loader.py
--------------
Loads Ground-Truth and Kalman-Filter CSV files, aligns them on sequence_no,
and converts lat/lon → local Cartesian (metres) using a tangent-plane
approximation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Earth radius (metres)
R_EARTH: float = 6_371_000.0

# ── Required columns ─────────────────────────────────────────────────────────

GT_REQUIRED  = {"sequence_no", "latitude", "longitude"}
KF_REQUIRED  = {"sequence_no", "KF_lat", "KF_long"}

GT_OPTIONAL  = {"ax", "ay", "az", "baro_alt"}


# ── Streamlit caching shim ────────────────────────────────────────────────────
# Decorates with @st.cache_data when running inside Streamlit,
# falls back to the bare function when imported from plain Python.

def _cache(fn):
    try:
        import streamlit as st          # noqa: PLC0415
        return st.cache_data(show_spinner=False)(fn)
    except ModuleNotFoundError:
        return fn


# ── Public API ────────────────────────────────────────────────────────────────

@_cache
def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load a ground-truth CSV.  Returns a DataFrame with at minimum:
    sequence_no, latitude, longitude  plus any IMU / baro columns present.
    """
    df = _read_csv(filepath, GT_REQUIRED, label="Ground Truth")
    df["sequence_no"] = df["sequence_no"].astype(int)
    return df.sort_values("sequence_no").reset_index(drop=True)


@_cache
def load_kf_output(filepath: str) -> pd.DataFrame:
    """
    Load a Kalman-Filter output CSV.
    Returns a DataFrame with sequence_no, KF_lat, KF_long.
    """
    df = _read_csv(filepath, KF_REQUIRED, label="KF Output")
    df["sequence_no"] = df["sequence_no"].astype(int)
    df = df.sort_values("sequence_no").reset_index(drop=True)
    
    dlat = df["KF_lat"].diff().abs()
    dlon = df["KF_long"].diff().abs()

    df = df[(dlat < 0.001) & (dlon < 0.001)].copy()
    
    return df


def align_and_convert(
    gt_df: pd.DataFrame,
    kf_df: pd.DataFrame,
    lat_ref: float,
    lon_ref: float,
) -> pd.DataFrame:
    """
    Inner-join GT and KF on sequence_no, then add Cartesian columns:

        gt_x, gt_y  – Ground Truth in metres (local frame)
        kf_x, kf_y  – KF estimate in metres (local frame)

    The local tangent-plane origin is (lat_ref, lon_ref).

    Conversion formulae
    -------------------
        x = (lon - lon_ref) * cos(lat_ref_rad) * (π/180) * R
        y = (lat - lat_ref)                     * (π/180) * R
    """
    merged = pd.merge(gt_df, kf_df, on="sequence_no", how="inner")

    if merged.empty:
        raise ValueError(
            "Inner join on 'sequence_no' produced no rows. "
            "Check that the two files share common sequence numbers."
        )

    merged["gt_x"], merged["gt_y"] = latlon_to_meters(
        merged["latitude"].to_numpy(),
        merged["longitude"].to_numpy(),
        lat_ref, lon_ref,
    )
    merged["kf_x"], merged["kf_y"] = latlon_to_meters(
        merged["KF_lat"].to_numpy(),
        merged["KF_long"].to_numpy(),
        lat_ref, lon_ref,
    )

    return merged.reset_index(drop=True)


def latlon_to_meters(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_ref: float,
    lon_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert arrays of (lat, lon) to local Cartesian (x, y) in metres.

    x = (lon - lon_ref) * cos(lat_ref_rad) * (π/180) * R
    y = (lat - lat_ref)                     * (π/180) * R
    """
    lat_ref_rad = np.radians(lat_ref)
    deg2m_lat   = (np.pi / 180.0) * R_EARTH
    deg2m_lon   = (np.pi / 180.0) * R_EARTH * np.cos(lat_ref_rad)

    x = (np.asarray(lon) - lon_ref) * deg2m_lon
    y = (np.asarray(lat) - lat_ref) * deg2m_lat
    return x, y


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_csv(filepath: str, required: set, label: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path.resolve()}\n"
            "Run  python generate_sample_data.py  to create sample data."
        )
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} CSV '{path.name}' is missing columns: {missing}"
        )
    return df
