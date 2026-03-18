"""Interactive web UI for trajkit."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st

from trajkit import TrajectoryDataset

SUPPORTED_EXTENSIONS = {".parquet", ".csv", ".json", ".jsonl", ".npy", ".npz", ".hdf5", ".h5", ".rlds", ".zip"}


def _inject_style() -> None:
    st.markdown(
        """
<style>
  .main { max-width: 1200px; margin: 0 auto; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .card {
    background: #ffffff;
    border: 1px solid #e7e9ee;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _write_uploaded_files(files: Iterable[st.runtime.uploaded_file_manager.UploadedFile], dst_dir: Path) -> Path:
    files = list(files)
    if not files:
        raise ValueError("Please upload at least one dataset file.")
    if len(files) == 1 and files[0].name.lower().endswith(".zip"):
        zip_path = dst_dir / files[0].name
        zip_path.write_bytes(files[0].getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst_dir / "dataset")
        return dst_dir / "dataset"
    for f in files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        (dst_dir / Path(f.name).name).write_bytes(f.getbuffer())
    return dst_dir


def _uploaded_size_mb(files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]) -> float:
    return float(sum(int(getattr(f, "size", 0)) for f in files)) / (1024.0 * 1024.0
    )


@st.cache_resource(show_spinner=False)
def _load_local_dataset_cached(path_str: str, include_images: str) -> TrajectoryDataset:
    return TrajectoryDataset.load(Path(path_str), include_images=include_images)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_traj_ids(path_str: str, include_images: str) -> list[str]:
    ds = _load_local_dataset_cached(path_str, include_images)
    return ds.trajectory_ids()


def _light_summary(ds: TrajectoryDataset) -> dict[str, object]:
    row = (
        ds.lazy_frame.select(
            pl.len().alias("num_points"),
            pl.col("trajectory_id").n_unique().alias("num_trajectories"),
            pl.col("t").null_count().alias("missing_timestamps"),
            pl.col("x").min().alias("min_x"),
            pl.col("x").max().alias("max_x"),
            pl.col("y").min().alias("min_y"),
            pl.col("y").max().alias("max_y"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "num_trajectories": int(row["num_trajectories"]),
        "num_points": int(row["num_points"]),
        "bounding_box_xy": {
            "min_x": float(row["min_x"]) if row["min_x"] is not None else 0.0,
            "max_x": float(row["max_x"]) if row["max_x"] is not None else 0.0,
            "min_y": float(row["min_y"]) if row["min_y"] is not None else 0.0,
            "max_y": float(row["max_y"]) if row["max_y"] is not None else 0.0,
        },
        "warnings": {"missing_timestamps": int(row["missing_timestamps"])},
    }


def _sample_xy(ds: TrajectoryDataset, max_points: int) -> pl.DataFrame:
    return (
        ds.lazy_frame.select(
            pl.col("x").cast(pl.Float64, strict=False).alias("x"),
            pl.col("y").cast(pl.Float64, strict=False).alias("y"),
        )
        .filter(pl.col("x").is_finite() & pl.col("y").is_finite())
        .limit(max_points)
        .collect()
    )


def _plot_trajectory_distribution(ds: TrajectoryDataset, max_points: int = 200_000):
    sampled = _sample_xy(ds, max_points=max_points)
    if sampled.is_empty():
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    hb = ax.hexbin(sampled["x"].to_numpy(), sampled["y"].to_numpy(), gridsize=72, mincnt=1, cmap="viridis")
    ax.set_title(f"Trajectory Distribution (sample up to {max_points:,} points)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    return fig


def _start_end_points(ds: TrajectoryDataset, max_traj: int | None = None) -> pl.DataFrame:
    ids = ds.trajectory_ids()
    if max_traj is not None and len(ids) > max_traj:
        ids = ids[:max_traj]
    lf = ds.lazy_frame
    if ids:
        lf = lf.filter(pl.col("trajectory_id").is_in(ids))
    return (
        lf.select(
            pl.col("trajectory_id"),
            pl.col("t").cast(pl.Float64, strict=False).alias("t"),
            pl.col("x").cast(pl.Float64, strict=False).alias("x"),
            pl.col("y").cast(pl.Float64, strict=False).alias("y"),
        )
        .filter(pl.col("x").is_finite() & pl.col("y").is_finite())
        .sort(["trajectory_id", "t"])
        .group_by("trajectory_id", maintain_order=True)
        .agg(
            pl.col("x").first().alias("x_start"),
            pl.col("y").first().alias("y_start"),
            pl.col("x").last().alias("x_end"),
            pl.col("y").last().alias("y_end"),
        )
        .collect()
    )


def _plot_start_end_heatmap(ds: TrajectoryDataset, max_traj: int = 100_000):
    se = _start_end_points(ds, max_traj=max_traj)
    if se.is_empty():
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
    hb1 = ax1.hexbin(se["x_start"].to_numpy(), se["y_start"].to_numpy(), gridsize=55, mincnt=1, cmap="Blues")
    hb2 = ax2.hexbin(se["x_end"].to_numpy(), se["y_end"].to_numpy(), gridsize=55, mincnt=1, cmap="Reds")
    ax1.set_title("Start-point Heatmap")
    ax2.set_title("End-point Heatmap")
    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(hb1, ax=ax1, label="count")
    fig.colorbar(hb2, ax=ax2, label="count")
    fig.tight_layout()
    return fig


def _plot_feature_hist(ds: TrajectoryDataset, feature_name: str):
    ft = ds.feature_table()
    if feature_name not in ft.columns:
        return None
    arr = ft[feature_name].to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.hist(arr, bins=28, edgecolor="white")
    ax.set_title(f"{feature_name} distribution")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def _infer_vector_labels(kind: str, width: int, frame_cols: list[str]) -> list[str]:
    if width <= 0:
        return []
    if kind == "state":
        # Canonical pose + quaternion layout.
        if width >= 7:
            labels = ["x", "y", "z", "eef_qx", "eef_qy", "eef_qz", "eef_qw"]
            labels.extend([f"state_extra_{i}" for i in range(width - 7)])
            return labels
        # Bridge-style pose layout.
        if width == 6:
            return ["x", "y", "z", "roll", "pitch", "yaw"]
        if width == 3:
            return ["x", "y", "z"]
        if width == 2:
            return ["x", "y"]
        return [f"state_{i}" for i in range(width)]

    if kind == "action":
        if width == 7:
            return ["pose_x", "pose_y", "pose_z", "pose_roll", "pose_pitch", "pose_yaw", "grasp"]
        if width == 6:
            # Common mobile-manipulation command layout.
            return ["cmd_vx", "cmd_vy", "cmd_vz", "cmd_wx", "cmd_wy", "cmd_wz"]
        if width == 3:
            return ["action_x", "action_y", "action_z"]
        return [f"action_{i}" for i in range(width)]

    if kind == "wrench":
        base = ["fx", "fy", "fz", "tx", "ty", "tz"]
        if width <= 6:
            return base[:width]
        return base + [f"wrench_extra_{i}" for i in range(width - 6)]

    return [f"{kind}_{i}" for i in range(width)]


def _expand_vector_with_labels(
    frame: pl.DataFrame,
    source_col: str,
    *,
    kind: str,
    max_channels: int = 16,
) -> tuple[pl.DataFrame, dict[int, str]]:
    if source_col not in frame.columns:
        return frame, {}
    vals = frame[source_col].to_list()
    width = 0
    for v in vals:
        if isinstance(v, list):
            width = max(width, len(v))
    width = min(width, max_channels)
    if width <= 0:
        return frame, {}

    labels = _infer_vector_labels(kind, width, frame.columns)
    exprs: list[pl.Expr] = []
    mapping: dict[int, str] = {}
    for i in range(width):
        label = labels[i] if i < len(labels) else f"{kind}_{i}"
        mapping[i] = label
        exprs.append(pl.col(source_col).list.get(i).cast(pl.Float64, strict=False).alias(f"{source_col}_{label}"))
    return frame.with_columns(exprs), mapping


def _render_trajectory_inspector(ds: TrajectoryDataset, include_images: str) -> None:
    st.subheader("Trajectory Inspector")
    ids = ds.trajectory_ids()
    if not ids:
        st.info("No trajectory_id found.")
        return

    idx = st.number_input("Trajectory index", min_value=0, max_value=len(ids) - 1, value=0, step=1)
    traj_id = ids[int(idx)]
    st.caption(f"Selected trajectory_id: `{traj_id}`")

    traj = ds.get_trajectory(traj_id)
    frame = traj.frame.sort("frame_id")
    st.write(f"Frames: {frame.height}")

    state_map: dict[int, str] = {}
    action_map: dict[int, str] = {}
    wrench_map: dict[int, str] = {}
    view = frame
    if "state_vec" in view.columns:
        view, state_map = _expand_vector_with_labels(view, "state_vec", kind="state")
    if "action_vec" in view.columns:
        view, action_map = _expand_vector_with_labels(view, "action_vec", kind="action")
    if "wrench_vec" in view.columns:
        view, wrench_map = _expand_vector_with_labels(view, "wrench_vec", kind="wrench")

    with st.expander("Channel meaning (inferred)", expanded=True):
        if state_map:
            st.markdown("**state_vec channels**")
            st.dataframe(
                pl.DataFrame(
                    {"index": list(state_map.keys()), "name": list(state_map.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )
        if action_map:
            st.markdown("**action_vec channels**")
            st.dataframe(
                pl.DataFrame(
                    {"index": list(action_map.keys()), "name": list(action_map.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )
        if wrench_map:
            st.markdown("**wrench_vec channels**")
            st.dataframe(
                pl.DataFrame(
                    {"index": list(wrench_map.keys()), "name": list(wrench_map.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not (state_map or action_map or wrench_map):
            st.info("No vector channels found for this trajectory.")

    with st.expander("States and Actions Table", expanded=True):
        st.dataframe(view, use_container_width=True, height=380)

    image_cols = [c for c in frame.columns if c.startswith("image_") and (c.endswith("_path") or c.endswith("_bytes"))]
    if not image_cols or include_images == "none":
        st.info("No image columns loaded for this trajectory.")
        return

    with st.expander("Images", expanded=False):
        modality = st.selectbox("Image modality", ["rgb", "depth", "any"], index=0)
        source = st.selectbox("Image source", ["auto", "path", "bytes"], index=0)
        stride = st.number_input("Display every Nth frame", min_value=1, max_value=50, value=10, step=1)
        max_images = st.number_input("Max decoded images", min_value=1, max_value=200, value=20, step=1)
        if st.button("Render images"):
            shown = 0
            for i in range(0, frame.height, int(stride)):
                if shown >= int(max_images):
                    break
                try:
                    img = traj.get_image(idx=i, modality=modality, source=source, decode=True)
                except Exception:
                    continue
                st.image(img, caption=f"{traj_id} | frame_idx={i}", use_container_width=True)
                shown += 1
            if shown == 0:
                st.warning("No decodable images found for selected options.")


def main() -> None:
    st.set_page_config(page_title="trajkit | Trajectory Explorer", page_icon="🧭", layout="wide")
    _inject_style()

    st.title("Trajectory Explorer")
    st.caption("Efficient visualization for large trajectory datasets.")

    with st.sidebar:
        st.subheader("Settings")
        include_images = st.selectbox(
            "Image loading mode",
            options=["none", "paths", "bytes"],
            index=0,
            help="Use 'none' for fastest loading. Use 'paths' for local image references.",
        )
        max_points_dist = st.slider("Max points for distribution", 20_000, 600_000, 200_000, 20_000)
        max_traj_start_end = st.slider("Max trajectories for start/end heatmap", 2_000, 200_000, 40_000, 2_000)
        process_mode = st.selectbox("Processing mode", options=["Auto", "Fast (Large files)", "Full"], index=0)
        local_dataset_path = st.text_input("Local dataset path (recommended for huge datasets)", value="")
        st.markdown("---")
        st.caption("Or upload one/multiple files, or a single .zip folder.")

    uploaded = st.file_uploader(
        "Drop dataset files here",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    use_local_path = bool(local_dataset_path.strip())
    if not uploaded and not use_local_path:
        st.info("Provide a local path or upload files to begin.")
        return

    upload_size_mb = _uploaded_size_mb(uploaded) if uploaded else 0.0
    auto_fast = upload_size_mb >= 200.0 or use_local_path
    fast_mode = True if process_mode == "Fast (Large files)" else False if process_mode == "Full" else auto_fast
    run_advanced = st.checkbox("Compute advanced analytics (feature table + outliers)", value=not fast_mode)

    source_caption = f"Local path: {local_dataset_path.strip()}" if use_local_path else f"Uploaded size: {upload_size_mb:.1f} MB"
    st.caption(f"{source_caption} | Active mode: {'Fast' if fast_mode else 'Full'}")

    try:
        if use_local_path:
            load_target = Path(local_dataset_path.strip()).expanduser().resolve()
            if not load_target.exists():
                st.error(f"Path not found: {load_target}")
                return
            with st.spinner("Loading dataset from local path..."):
                ds = _load_local_dataset_cached(str(load_target), include_images)
                summary = _light_summary(ds) if fast_mode else ds.summary()
                feature_table = ds.feature_table() if run_advanced else None
                _ = _cached_traj_ids(str(load_target), include_images)
        else:
            with tempfile.TemporaryDirectory(prefix="trajkit_ui_") as tmp_dir:
                tmp = Path(tmp_dir)
                load_target = _write_uploaded_files(uploaded, tmp)
                with st.spinner("Loading uploaded dataset..."):
                    # Important: upload files live in a temporary directory.
                    # Materialize immediately so later lazy collects don't reference deleted temp paths.
                    ds_tmp = TrajectoryDataset.load(load_target, include_images=include_images)
                    ds = TrajectoryDataset(ds_tmp.frame)
                    summary = _light_summary(ds) if fast_mode else ds.summary()
                    feature_table = ds.feature_table() if run_advanced else None
    except Exception as exc:
        st.error("Failed to load dataset.")
        st.code(str(exc))
        st.info(
            "Tip: if dataset is huge, use local path mode. If files are partially downloaded/corrupted, "
            "re-download and retry."
        )
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajectories", int(summary.get("num_trajectories", 0)))
    c2.metric("Points", int(summary.get("num_points", 0)))
    c3.metric("Feature rows", int(feature_table.height) if feature_table is not None else 0)
    warnings = summary.get("warnings", {})
    c4.metric("Missing timestamps", int(warnings.get("missing_timestamps", 0)))
    st.markdown("</div>", unsafe_allow_html=True)

    overview_tab, inspector_tab = st.tabs(["Overview", "Trajectory Inspector"])

    with overview_tab:
        st.subheader("Trajectory Distribution")
        fig_dist = _plot_trajectory_distribution(ds, max_points=max_points_dist)
        if fig_dist is not None:
            st.pyplot(fig_dist, clear_figure=True)
        else:
            st.warning("No valid x/y points available.")

        st.subheader("Start-End Point Heatmap")
        fig_se = _plot_start_end_heatmap(ds, max_traj=max_traj_start_end)
        if fig_se is not None:
            st.pyplot(fig_se, clear_figure=True)
        else:
            st.warning("Unable to compute start/end heatmap.")

        left, right = st.columns([1.2, 1.0])
        with left:
            if run_advanced:
                feature_choice = st.selectbox(
                    "Feature distribution",
                    ["duration", "path_length", "mean_speed", "max_speed", "mean_acceleration", "max_acceleration", "straightness"],
                    index=2,
                )
                hist_fig = _plot_feature_hist(ds, feature_choice)
                if hist_fig is not None:
                    st.pyplot(hist_fig, clear_figure=True)
        with right:
            st.subheader("Summary")
            st.json(summary, expanded=False)
            if run_advanced:
                st.subheader("Outliers")
                st.dataframe(ds.find_outliers(top_k=10), use_container_width=True)

    with inspector_tab:
        _render_trajectory_inspector(ds, include_images=include_images)


if __name__ == "__main__":
    main()
