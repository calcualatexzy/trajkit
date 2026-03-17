"""Derived feature extraction for single trajectories."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0.0:
        return 0.0
    return float(numer / denom)


def trajectory_features(frame: pl.DataFrame) -> dict[str, float]:
    if frame.height < 2:
        return {
            "duration": 0.0,
            "path_length": 0.0,
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "mean_acceleration": 0.0,
            "max_acceleration": 0.0,
            "heading_change": 0.0,
            "curvature_proxy": 0.0,
            "stop_ratio": 0.0,
            "straightness": 0.0,
        }

    ordered = frame.sort("t")
    xyz = ordered.select(["x", "y", "z"]).to_numpy().astype(float)
    t = ordered["t"].to_numpy().astype(float)

    dxyz = np.diff(xyz, axis=0)
    dt = np.diff(t)
    dt = np.where(dt == 0.0, np.nan, dt)

    step_dist = np.linalg.norm(dxyz, axis=1)
    path_length = float(np.nansum(step_dist))
    duration = float(np.nanmax(t) - np.nanmin(t))
    speed = np.divide(step_dist, dt, out=np.zeros_like(step_dist), where=~np.isnan(dt))
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    accel = np.diff(speed)
    dt_mid = dt[1:]
    dt_mid = np.where(dt_mid == 0.0, np.nan, dt_mid)
    accel = np.divide(accel, dt_mid, out=np.zeros_like(accel), where=~np.isnan(dt_mid))
    accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)

    headings = np.arctan2(dxyz[:, 1], dxyz[:, 0])
    heading_diff = np.abs(np.diff(headings))
    heading_change = float(np.nansum(np.minimum(heading_diff, 2 * math.pi - heading_diff)))

    direct_dist = float(np.linalg.norm(xyz[-1] - xyz[0]))
    straightness = _safe_div(direct_dist, path_length)

    stop_ratio = float(np.mean(speed < 1e-6)) if speed.size else 0.0
    curvature_proxy = _safe_div(heading_change, path_length)

    return {
        "duration": duration,
        "path_length": path_length,
        "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
        "max_speed": float(np.max(speed)) if speed.size else 0.0,
        "mean_acceleration": float(np.mean(np.abs(accel))) if accel.size else 0.0,
        "max_acceleration": float(np.max(np.abs(accel))) if accel.size else 0.0,
        "heading_change": heading_change,
        "curvature_proxy": curvature_proxy,
        "stop_ratio": stop_ratio,
        "straightness": straightness,
        **_wrench_stats(ordered),
    }


def _wrench_stats(frame: pl.DataFrame) -> dict[str, float]:
    if "wrench_vec" not in frame.columns:
        return {}
    wrench = np.asarray(frame["wrench_vec"].to_list(), dtype=float)
    if wrench.ndim != 2 or wrench.shape[1] < 6:
        return {}
    force = np.linalg.norm(wrench[:, :3], axis=1)
    torque = np.linalg.norm(wrench[:, 3:6], axis=1)
    return {
        "mean_force_norm": float(np.mean(force)) if force.size else 0.0,
        "max_force_norm": float(np.max(force)) if force.size else 0.0,
        "mean_torque_norm": float(np.mean(torque)) if torque.size else 0.0,
        "max_torque_norm": float(np.max(torque)) if torque.size else 0.0,
    }
