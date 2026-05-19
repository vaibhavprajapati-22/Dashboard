"""
metrics.py
----------
Computes trajectory-comparison metrics between Ground-Truth and Kalman-Filter
estimates expressed in local Cartesian metres.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrajectoryMetrics:
    mse:       float   # Mean Squared Error           [m²]
    rmse:      float   # Root Mean Squared Error       [m]
    mean_err:  float   # Mean absolute Euclidean error [m]
    max_err:   float   # Max  absolute Euclidean error [m]
    min_err:   float   # Min  absolute Euclidean error [m]
    std_err:   float   # Std-dev of error              [m]
    n_points:  int     # Number of aligned points


def compute_euclidean_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'error' column (Euclidean distance in metres) to *df*.
    Expects columns: gt_x, gt_y, kf_x, kf_y.
    Returns a copy.
    """
    df = df.copy()
    dx = df["gt_x"] - df["kf_x"]
    dy = df["gt_y"] - df["kf_y"]
    df["error"] = np.sqrt(dx**2 + dy**2)
    return df


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, TrajectoryMetrics]:
    """
    Compute all metrics and return (annotated_df, TrajectoryMetrics).

    annotated_df has the extra 'error' column.
    """
    df = compute_euclidean_error(df)
    err = df["error"].to_numpy()

    metrics = TrajectoryMetrics(
        mse      = float(np.mean(err**2)),
        rmse     = float(np.sqrt(np.mean(err**2))),
        mean_err = float(np.mean(err)),
        max_err  = float(np.max(err)),
        min_err  = float(np.min(err)),
        std_err  = float(np.std(err)),
        n_points = len(err),
    )
    return df, metrics
