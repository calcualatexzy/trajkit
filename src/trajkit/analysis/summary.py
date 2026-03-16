"""Dataset summary utilities."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def per_trajectory_features(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    if "z" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.lit(0.0).alias("z"))

    steps = (
        lf.select(["trajectory_id", "t", "x", "y", "z"])
        .sort(["trajectory_id", "t"])
        .with_columns(
            pl.col("x").diff().over("trajectory_id").alias("_dx"),
            pl.col("y").diff().over("trajectory_id").alias("_dy"),
            pl.col("z").diff().over("trajectory_id").alias("_dz"),
            pl.col("t").diff().over("trajectory_id").alias("_dt"),
        )
        .with_columns(
            (pl.col("_dx").pow(2) + pl.col("_dy").pow(2) + pl.col("_dz").pow(2)).sqrt().fill_null(0.0).alias("_step_dist")
        )
        .with_columns(
            pl.when(pl.col("_dt") > 0)
            .then(pl.col("_step_dist") / pl.col("_dt"))
            .otherwise(0.0)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_speed"),
        )
        .with_columns(
            pl.col("_speed")
            .diff()
            .over("trajectory_id")
            .fill_null(0.0)
            .alias("_speed_diff"),
            pl.arctan2(pl.col("_dy"), pl.col("_dx")).fill_nan(0.0).fill_null(0.0).alias("_heading"),
        )
        .with_columns(
            pl.when(pl.col("_dt") > 0)
            .then(pl.col("_speed_diff") / pl.col("_dt"))
            .otherwise(0.0)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_accel"),
            (pl.col("_heading").diff().over("trajectory_id").abs()).fill_null(0.0).alias("_heading_delta"),
        )
        .with_columns(
            pl.min_horizontal(pl.col("_heading_delta"), pl.lit(2 * math.pi) - pl.col("_heading_delta")).alias(
                "_heading_delta_wrapped"
            )
        )
    )

    feat = (
        steps.group_by("trajectory_id")
        .agg(
            (pl.col("t").max() - pl.col("t").min()).alias("duration"),
            pl.col("_step_dist").sum().alias("path_length"),
            pl.col("_speed").mean().alias("mean_speed"),
            pl.col("_speed").max().alias("max_speed"),
            pl.col("_accel").abs().mean().alias("mean_acceleration"),
            pl.col("_accel").abs().max().alias("max_acceleration"),
            pl.col("_heading_delta_wrapped").sum().alias("heading_change"),
            (pl.col("_speed") < 1e-6).cast(pl.Float64).mean().alias("stop_ratio"),
            pl.col("x").first().alias("_x0"),
            pl.col("y").first().alias("_y0"),
            pl.col("z").first().alias("_z0"),
            pl.col("x").last().alias("_x1"),
            pl.col("y").last().alias("_y1"),
            pl.col("z").last().alias("_z1"),
        )
        .with_columns(
            (pl.col("_x1") - pl.col("_x0")).pow(2)
            .add((pl.col("_y1") - pl.col("_y0")).pow(2))
            .add((pl.col("_z1") - pl.col("_z0")).pow(2))
            .sqrt()
            .alias("_direct_dist")
        )
        .with_columns(
            pl.when(pl.col("path_length") > 0).then(pl.col("_direct_dist") / pl.col("path_length")).otherwise(0.0).alias(
                "straightness"
            ),
            pl.when(pl.col("path_length") > 0)
            .then(pl.col("heading_change") / pl.col("path_length"))
            .otherwise(0.0)
            .alias("curvature_proxy"),
        )
        .select(
            [
                "trajectory_id",
                "duration",
                "path_length",
                "mean_speed",
                "max_speed",
                "mean_acceleration",
                "max_acceleration",
                "heading_change",
                "curvature_proxy",
                "stop_ratio",
                "straightness",
            ]
        )
        .collect()
    )
    return feat


def summarize_dataset(frame: pl.DataFrame | pl.LazyFrame) -> dict[str, object]:
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema_names = lf.collect_schema().names()
    if "z" not in schema_names:
        lf = lf.with_columns(pl.lit(0.0).alias("z"))
    has_label = "label" in schema_names

    base = (
        lf.select(
            pl.len().alias("num_points"),
            pl.col("trajectory_id").n_unique().alias("num_trajectories"),
            pl.col("t").null_count().alias("missing_timestamps"),
            pl.col("x").min().alias("min_x"),
            pl.col("x").max().alias("max_x"),
            pl.col("y").min().alias("min_y"),
            pl.col("y").max().alias("max_y"),
            pl.col("z").min().alias("min_z"),
            pl.col("z").max().alias("max_z"),
        )
        .collect()
        .row(0, named=True)
    )
    if int(base["num_points"]) == 0:
        return {"num_trajectories": 0}

    feat = per_trajectory_features(lf)
    t_group_aggs: list[pl.Expr] = [
        pl.col("t").min().alias("t_min"),
        pl.col("t").max().alias("t_max"),
        pl.len().alias("n_points"),
        pl.col("t").n_unique().alias("unique_t"),
    ]
    if "frame_id" in schema_names:
        t_group_aggs.append(pl.col("frame_id").n_unique().alias("unique_frame_id_per_traj"))

    t_groups = lf.group_by("trajectory_id").agg(t_group_aggs).collect()
    sample_rate = (
        t_groups.select((pl.col("n_points") / (pl.col("t_max") - pl.col("t_min")).clip(lower_bound=1e-9)).alias("hz"))["hz"]
        .to_numpy()
    )
    duplicates = int((t_groups["n_points"] - t_groups["unique_t"]).sum())
    duplicate_frame_ids = (
        int((t_groups["n_points"] - t_groups["unique_frame_id_per_traj"]).sum())
        if "unique_frame_id_per_traj" in t_groups.columns
        else 0
    )

    label_counts: list[dict[str, object]] = []
    if has_label:
        label_counts = (
            lf.select(pl.col("label").drop_nulls().alias("label"))
            .group_by("label")
            .len()
            .sort("len", descending=True)
            .rename({"len": "count"})
            .collect()
            .to_dicts()
        )

    return {
        "num_trajectories": int(base["num_trajectories"]),
        "num_points": int(base["num_points"]),
        "duration": _describe(feat["duration"].to_numpy()),
        "path_length": _describe(feat["path_length"].to_numpy()),
        "mean_speed": _describe(feat["mean_speed"].to_numpy()),
        "max_speed": _describe(feat["max_speed"].to_numpy()),
        "mean_acceleration": _describe(feat["mean_acceleration"].to_numpy()),
        "sampling_rate_hz": _describe(sample_rate),
        "bounding_box": {
            "min_x": float(base["min_x"]),
            "max_x": float(base["max_x"]),
            "min_y": float(base["min_y"]),
            "max_y": float(base["max_y"]),
            "min_z": float(base["min_z"]),
            "max_z": float(base["max_z"]),
        },
        "label_distribution": label_counts,
        "modalities": _modality_stats(lf, schema_names=schema_names, total_points=int(base["num_points"])),
        "warnings": {
            "missing_timestamps": int(base["missing_timestamps"]),
            "duplicate_timestamps": duplicates,
            "duplicate_frame_ids": duplicate_frame_ids,
        },
    }


def _describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "max": float(np.max(values)),
    }


def _modality_stats(lf: pl.LazyFrame, schema_names: list[str], total_points: int) -> dict[str, object]:
    image_cols = [
        col
        for col in schema_names
        if col.startswith("image_") and (col.endswith("_path") or col.endswith("_bytes"))
    ]
    out: dict[str, object] = {
        "has_states": "state_vec" in schema_names,
        "has_actions": "action_vec" in schema_names,
        "has_images": bool(image_cols),
    }
    if total_points <= 0:
        return out

    coverage_exprs: list[pl.Expr] = []
    for col in ("state_vec", "action_vec", *image_cols):
        if col in schema_names:
            coverage_exprs.append((1.0 - pl.col(col).null_count() / pl.len()).alias(f"{col}_coverage"))
    if coverage_exprs:
        row = lf.select(coverage_exprs).collect().row(0, named=True)
        out.update({k: float(v) for k, v in row.items() if v is not None})
    return out
