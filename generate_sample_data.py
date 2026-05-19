"""
generate_sample_data.py
-----------------------
Creates realistic synthetic ground-truth and Kalman-filter CSV files
inside the  data/  folder so the dashboard works out of the box.

Usage
-----
    python generate_sample_data.py
"""

import os
import yaml
import numpy as np
import pandas as pd

# ── Reference origin (must match config.yaml) ─────────────────────────────────
LAT_REF = 30.3753
LON_REF = 69.3451
R_EARTH = 6_371_000.0

np.random.seed(42)
os.makedirs("data", exist_ok=True)


# ── Conversion helpers ────────────────────────────────────────────────────────

def meters_to_latlon(x, y):
    """Convert local Cartesian (metres) → (lat, lon) degrees."""
    lat_ref_rad = np.radians(LAT_REF)
    lat = y / ((np.pi / 180.0) * R_EARTH) + LAT_REF
    lon = x / ((np.pi / 180.0) * R_EARTH * np.cos(lat_ref_rad)) + LON_REF
    return lat, lon


# ── Path generators ───────────────────────────────────────────────────────────

def circle_path(n=250, radius=500):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.cos(t), radius * np.sin(t)


def spiral_path(n=250):
    t = np.linspace(0, 3 * np.pi, n)
    r = np.linspace(80, 600, n)
    return r * np.cos(t), r * np.sin(t)


def figure8_path(n=250, scale=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    return x, y


def square_path(n=250, side=600):
    """Rounded square with interpolated waypoints."""
    corners_x = [0,    side, side, 0,    0]
    corners_y = [0,    0,    side, side,  0]
    segs = []
    for i in range(4):
        seg_n = n // 4 + (1 if i < n % 4 else 0)
        segs.append(np.column_stack([
            np.linspace(corners_x[i], corners_x[i+1], seg_n, endpoint=False),
            np.linspace(corners_y[i], corners_y[i+1], seg_n, endpoint=False),
        ]))
    pts = np.vstack(segs)
    # Centre at origin
    pts -= pts.mean(axis=0)
    return pts[:n, 0], pts[:n, 1]


PATHS = [circle_path, spiral_path, figure8_path, square_path]
NAMES = ["Circle Path", "Spiral Path", "Figure-8 Path", "Square Path"]

# ── KF noise configurations (noise_std_m, random_walk_std_m_per_step) ─────────
KF_VARIANTS = {
    "KF Standard":  (9.0,  0.06),   # worst  – high noise + drift
    "KF Extended":  (5.5,  0.035),  # medium
    "KF Unscented": (2.8,  0.015),  # good
    "KF Adaptive":  (1.2,  0.006),  # best   – low noise + drift
}

STRATEGIES = ["Incremental", "Consensus", "Adapt Then Combine", "Combine Then Adapt"]


# ── Generate ──────────────────────────────────────────────────────────────────

def add_kf_noise(x, y, noise_std, drift_std, seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    n   = len(x)
    # Gaussian noise + cumulative drift (Brownian)
    dx = rng.normal(0, noise_std, n) + np.cumsum(rng.normal(0, drift_std, n))
    dy = rng.normal(0, noise_std, n) + np.cumsum(rng.normal(0, drift_std, n))
    return x + dx, y + dy


config_dict = {
    "reference": {
        "lat_ref": LAT_REF,
        "lon_ref": LON_REF
    },
    "paths": {}
}

for path_idx, (path_fn, shape_name) in enumerate(zip(PATHS, NAMES), start=1):
    N   = 250
    seq = np.arange(1, N + 1)

    # ── Ground Truth ──────────────────────────────────────────────────────────
    x_gt, y_gt = path_fn(n=N)
    lat, lon   = meters_to_latlon(x_gt, y_gt)

    # Synthetic IMU & baro
    speed = np.sqrt(np.diff(x_gt, prepend=x_gt[0])**2 +
                    np.diff(y_gt, prepend=y_gt[0])**2)
    ax  = np.gradient(np.diff(x_gt, prepend=x_gt[0]))  + np.random.normal(0, 0.05, N)
    ay  = np.gradient(np.diff(y_gt, prepend=y_gt[0]))  + np.random.normal(0, 0.05, N)
    az  = np.random.normal(9.81, 0.05, N)
    baro_alt = 1000 + 50 * np.sin(np.linspace(0, 2 * np.pi, N)) \
               + np.cumsum(np.random.normal(0, 0.02, N))

    gt_df = pd.DataFrame({
        "sequence_no" : seq,
        "latitude"    : lat,
        "longitude"   : lon,
        "ax"          : ax,
        "ay"          : ay,
        "az"          : az,
        "baro_alt"    : baro_alt,
    })
    out_gt = f"data/path{path_idx}_gt.csv"
    gt_df.to_csv(out_gt, index=False, float_format="%.8f")
    print(f"  OK  {out_gt}")
    
    config_dict["paths"][shape_name] = {
        "gt_file": out_gt,
        "kf_variants": {}
    }

    # ── KF variants ───────────────────────────────────────────────────────────
    for v_idx, (kf_name, (noise, drift)) in enumerate(KF_VARIANTS.items()):
        
        config_dict["paths"][shape_name]["kf_variants"][kf_name] = {
            "strategies": {}
        }
        
        for s_idx, strategy in enumerate(STRATEGIES):
            nodes_data = {}
            nodes_config = {}
            
            for node_i in range(1, 4):
                seed_off = v_idx * 100 + s_idx * 10 + node_i
                # Add a bit of strategy-specific noise scaling to make them look different
                s_noise = noise * (1.0 + s_idx * 0.1 + node_i * 0.05)
                s_drift = drift * (1.0 + s_idx * 0.1 + node_i * 0.05)
                
                x_kf, y_kf = add_kf_noise(x_gt, y_gt, s_noise, s_drift, seed_offset=seed_off)
                kf_lat, kf_lon = meters_to_latlon(x_kf, y_kf)
                
                kf_df = pd.DataFrame({
                    "sequence_no" : seq,
                    "KF_lat"      : kf_lat,
                    "KF_long"     : kf_lon,
                })
                
                safe_kfname = kf_name.lower().replace(" ", "_")
                safe_sname = strategy.lower().replace(" ", "_")
                out_kf = f"data/path{path_idx}_{safe_kfname}_{safe_sname}_node{node_i}.csv"
                kf_df.to_csv(out_kf, index=False, float_format="%.8f")
                
                nodes_data[f"Node {node_i}"] = kf_df
                nodes_config[f"Node {node_i}"] = out_kf

            strat_dict = {"nodes": nodes_config}
            
            # ── Combination file (if not Incremental) ─────────────────────────
            if strategy != "Incremental":
                # Average lat and lon of the 3 nodes
                comb_lat = (nodes_data["Node 1"]["KF_lat"] + nodes_data["Node 2"]["KF_lat"] + nodes_data["Node 3"]["KF_lat"]) / 3.0
                comb_lon = (nodes_data["Node 1"]["KF_long"] + nodes_data["Node 2"]["KF_long"] + nodes_data["Node 3"]["KF_long"]) / 3.0
                
                comb_df = pd.DataFrame({
                    "sequence_no" : seq,
                    "KF_lat"      : comb_lat,
                    "KF_long"     : comb_lon,
                })
                
                out_comb = f"data/path{path_idx}_{safe_kfname}_{safe_sname}_combination.csv"
                comb_df.to_csv(out_comb, index=False, float_format="%.8f")
                strat_dict["combination"] = out_comb

            config_dict["paths"][shape_name]["kf_variants"][kf_name]["strategies"][strategy] = strat_dict
            
print("\nDONE  All sample data generated successfully!")

# Write config.yaml
with open("config.yaml", "w") as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
print("DONE  config.yaml updated!")
print("    Run:  streamlit run app.py")
