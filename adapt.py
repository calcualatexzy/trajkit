"""Load ADAPT trajectories from data/adapt_data with trajkit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from trajkit import TrajectoryDataset


# You can edit these defaults directly, or pass CLI args.
DEFAULT_DATA_DIR = "data/adapt_data"
DEFAULT_FT_KEYWORD = "ft"
DEFAULT_FT_COLS: tuple[str, str, str, str, str, str] | None = None


def _load_adapt_lazy(
    path: str | Path,
    *,
    ft_keyword: str = DEFAULT_FT_KEYWORD,
    ft_cols: tuple[str, str, str, str, str, str] | None = DEFAULT_FT_COLS,
) -> pl.LazyFrame:
    root = Path(path)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no csv files under {root}")

    frames: list[pl.LazyFrame] = []
    for csv in files:
        schema_names = set(pl.scan_csv(csv, n_rows=1).collect_schema().names())
        use_ft_cols = _resolve_ft_cols(schema_names, ft_keyword=ft_keyword, ft_cols=ft_cols)

        traj_id = csv.stem
        lf = pl.scan_csv(csv).with_columns(
            pl.lit(traj_id).alias("trajectory_id"),
            pl.col("elapsed_s").cast(pl.Float64, strict=False).alias("t"),
            pl.col("eef_x").cast(pl.Float64, strict=False).alias("x"),
            pl.col("eef_y").cast(pl.Float64, strict=False).alias("y"),
            pl.col("eef_z").cast(pl.Float64, strict=False).alias("z"),
            pl.col("timestamp_ns").cast(pl.Int64, strict=False).alias("timestamp_ns"),
        )

        lf = lf.with_columns(
            pl.concat_list(
                [
                    pl.col("eef_x").cast(pl.Float64, strict=False),
                    pl.col("eef_y").cast(pl.Float64, strict=False),
                    pl.col("eef_z").cast(pl.Float64, strict=False),
                    pl.col("eef_qx").cast(pl.Float64, strict=False),
                    pl.col("eef_qy").cast(pl.Float64, strict=False),
                    pl.col("eef_qz").cast(pl.Float64, strict=False),
                    pl.col("eef_qw").cast(pl.Float64, strict=False),
                ]
            ).alias("state_vec"),
            pl.concat_list(
                [
                    pl.col("cmd_vx").cast(pl.Float64, strict=False),
                    pl.col("cmd_vy").cast(pl.Float64, strict=False),
                    pl.col("cmd_vz").cast(pl.Float64, strict=False),
                    pl.col("cmd_wx").cast(pl.Float64, strict=False),
                    pl.col("cmd_wy").cast(pl.Float64, strict=False),
                    pl.col("cmd_wz").cast(pl.Float64, strict=False),
                ]
            ).alias("action_vec"),
            pl.concat_list(
                [
                    pl.col(use_ft_cols[0]).cast(pl.Float64, strict=False),
                    pl.col(use_ft_cols[1]).cast(pl.Float64, strict=False),
                    pl.col(use_ft_cols[2]).cast(pl.Float64, strict=False),
                    pl.col(use_ft_cols[3]).cast(pl.Float64, strict=False),
                    pl.col(use_ft_cols[4]).cast(pl.Float64, strict=False),
                    pl.col(use_ft_cols[5]).cast(pl.Float64, strict=False),
                ]
            ).alias("wrench_vec"),
        )

        # Drop invalid samples (ADAPT logs may start with NaN warmup rows).
        lf = lf.filter(
            pl.col("t").is_finite() & pl.col("x").is_finite() & pl.col("y").is_finite() & pl.col("z").is_finite()
        )

        lf = lf.with_columns(pl.int_range(0, pl.len()).alias("frame_id")).select(
            ["trajectory_id", "frame_id", "t", "x", "y", "z", "state_vec", "action_vec", "wrench_vec", "timestamp_ns"]
        )
        frames.append(lf)
    return pl.concat(frames, how="diagonal_relaxed")


def _resolve_ft_cols(
    schema_names: set[str],
    *,
    ft_keyword: str,
    ft_cols: tuple[str, str, str, str, str, str] | None,
) -> tuple[str, str, str, str, str, str]:
    if ft_cols is not None:
        missing = [c for c in ft_cols if c not in schema_names]
        if missing:
            raise ValueError(f"specified ft columns missing: {missing}")
        return ft_cols

    guessed = (
        f"{ft_keyword}_fx",
        f"{ft_keyword}_fy",
        f"{ft_keyword}_fz",
        f"{ft_keyword}_tx",
        f"{ft_keyword}_ty",
        f"{ft_keyword}_tz",
    )
    if all(c in schema_names for c in guessed):
        return guessed

    fallback = ("ft_fx", "ft_fy", "ft_fz", "ft_tx", "ft_ty", "ft_tz")
    if all(c in schema_names for c in fallback):
        return fallback

    raise ValueError(
        "cannot resolve F/T columns; provide --ft-cols "
        "(e.g. --ft-cols fx,fy,fz,tx,ty,tz) or --ft-keyword."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and visualize ADAPT trajectories.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder containing ADAPT csv files.")
    parser.add_argument(
        "--ft-keyword",
        default=DEFAULT_FT_KEYWORD,
        help="Prefix keyword for F/T columns, e.g. 'ft' -> ft_fx..ft_tz, 'wrench' -> wrench_fx..wrench_tz.",
    )
    parser.add_argument(
        "--ft-cols",
        default="",
        help="Explicit F/T columns as comma list: fx,fy,fz,tx,ty,tz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    user_ft_cols = tuple(c.strip() for c in args.ft_cols.split(",") if c.strip())
    ft_cols: tuple[str, str, str, str, str, str] | None = None
    if user_ft_cols:
        if len(user_ft_cols) != 6:
            raise ValueError("--ft-cols must contain exactly 6 columns: fx,fy,fz,tx,ty,tz")
        ft_cols = (
            user_ft_cols[0],
            user_ft_cols[1],
            user_ft_cols[2],
            user_ft_cols[3],
            user_ft_cols[4],
            user_ft_cols[5],
        )

    ds = TrajectoryDataset(_load_adapt_lazy(args.data_dir, ft_keyword=args.ft_keyword, ft_cols=ft_cols))
    print("summary:", ds.summary())
    print("feature_rows:", ds.feature_table().height)
    first = ds[0]
    print("first_trajectory:", first.id, "frames:", first.frame.height)
    print("first_frame:", first.get_frame(idx=0))

    # Dataset-level visualization.
    ds.plot_coverage()
    ds.plot_start_end_distribution()
    ds.plot_feature_distributions()
    ds.plot_outlier_ranking(top_k=min(10, len(ds)))
    ds.plot_cluster_embedding(k=min(4, len(ds)), seed=0)
    ds.plot_modality_coverage()
    ds.plot_prototypes()

    # Trajectory-level visualization for every trajectory.
    traj_plots = 0
    wrench_plots = 0
    for traj_id in ds.trajectory_ids():
        traj = ds.get_trajectory(traj_id)
        traj.plot()
        traj_plots += 1
        if traj.state_vectors() is not None:
            traj.plot_state_channels(max_channels=6)
        if traj.action_vectors() is not None:
            traj.plot_action_channels(max_channels=6)
        if traj.force_torque_vectors() is None:
            continue
        traj.plot_wrench_channels()
        wrench_plots += 1
    print("trajectory_plots:", traj_plots)
    print("wrench_plots:", wrench_plots)
    plt.show()


if __name__ == "__main__":
    main()
