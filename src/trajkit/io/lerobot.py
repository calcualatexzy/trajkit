"""Adapter for loading LeRobotDataset-like objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl

from trajkit.io.loaders import load_dataset_lazy

ImageMode = Literal["none", "paths", "bytes"]


def load_lerobot_dataset(
    dataset: Any,
    include_states: bool = True,
    include_actions: bool = True,
    include_images: ImageMode = "paths",
) -> pl.LazyFrame:
    root = _candidate_root(dataset)
    if root is not None:
        frame = load_dataset_lazy(
            path=root,
            include_states=include_states,
            include_actions=include_actions,
            include_images=include_images,
        )
        if include_images != "none":
            frame = _attach_video_path_refs(frame, dataset)
        return frame

    # Fallback path: chunked row conversion for generic dataset objects.
    batches: list[pl.DataFrame] = []
    n = len(dataset)
    chunk_size = 4096
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        rows: list[dict[str, Any]] = []
        for idx in range(start, end):
            sample = dataset[idx]
            if not isinstance(sample, Mapping):
                raise TypeError("LeRobot dataset samples must be mapping-like")
            row = _sample_to_row(
                sample=sample,
                idx=idx,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            )
            rows.append(row)
        if rows:
            batches.append(pl.DataFrame(rows))
    if not batches:
        raise ValueError("empty LeRobotDataset")

    frame = pl.concat(batches, how="diagonal_relaxed").with_columns(
        pl.col("trajectory_id").cast(pl.String),
        pl.col("frame_id").cast(pl.Int64),
        pl.col("t").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    if "state_vec" in frame.columns:
        frame = frame.with_columns(pl.col("state_vec").cast(pl.List(pl.Float64)))
    if "action_vec" in frame.columns:
        frame = frame.with_columns(pl.col("action_vec").cast(pl.List(pl.Float64)))
    return frame.lazy()


def detect_lerobot_loader_mode(dataset: Any) -> str:
    root = _candidate_root(dataset)
    if root is None:
        return "lerobot_iter_chunked"
    if _video_meta_keys(dataset):
        return "lerobot_root_fast_path_video_meta"
    return "lerobot_root_fast_path"


def _sample_to_row(
    sample: Mapping[str, Any],
    idx: int,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> dict[str, Any]:
    episode_idx = _first_not_none(
        _get_any(sample, "episode_index", "episode_idx", "episode"),
        _nested(sample, "metadata", "episode_index"),
    )
    frame_idx = _first_not_none(
        _get_any(sample, "frame_index", "frame_id"),
        _nested(sample, "metadata", "frame_index"),
        idx,
    )
    timestamp = _first_not_none(
        _get_any(sample, "timestamp", "t"),
        _nested(sample, "metadata", "timestamp"),
        float(idx),
    )

    state_vec = None
    if include_states:
        state_vec = _as_float_list(
            _first_not_none(
                _get_any(sample, "observation.state", "state_vec"),
                _nested(sample, "observation", "state"),
                _nested(sample, "state"),
            )
        )

    action_vec = None
    if include_actions:
        action_vec = _as_float_list(
            _first_not_none(
                _get_any(sample, "action", "action_vec"),
                _nested(sample, "actions"),
            )
        )

    x = 0.0
    y = 0.0
    z = 0.0
    if state_vec is not None and len(state_vec) >= 2:
        x = float(state_vec[0])
        y = float(state_vec[1])
        z = float(state_vec[2]) if len(state_vec) >= 3 else 0.0

    row: dict[str, Any] = {
        "trajectory_id": str(episode_idx if episode_idx is not None else 0),
        "frame_id": int(frame_idx),
        "t": float(timestamp),
        "x": x,
        "y": y,
        "z": z,
        "episode_index": int(episode_idx) if episode_idx is not None else 0,
        "frame_index": int(frame_idx),
    }
    if state_vec is not None:
        row["state_vec"] = state_vec
    if action_vec is not None:
        row["action_vec"] = action_vec

    if include_images != "none":
        image0 = _first_not_none(
            _get_any(sample, "observation.images.image", "observation.image", "image"),
            _nested(sample, "observation", "images", "image"),
            _nested(sample, "observation", "image"),
        )
        image1 = _first_not_none(
            _get_any(sample, "observation.images.image2", "observation.image2"),
            _nested(sample, "observation", "images", "image2"),
            _nested(sample, "observation", "image2"),
        )
        _write_image_fields(row, "image_rgb", image0, include_images=include_images)
        _write_image_fields(row, "image_depth", image1, include_images=include_images)
        # Legacy aliases for backwards compatibility.
        _write_image_fields(row, "image_0", image0, include_images=include_images)
        _write_image_fields(row, "image_1", image1, include_images=include_images)
    return row


def _write_image_fields(row: dict[str, Any], prefix: str, value: Any, include_images: ImageMode) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        path = value.get("path")
        raw = value.get("bytes")
        if path is not None:
            row[f"{prefix}_path"] = str(path)
        if include_images == "bytes" and raw is not None:
            row[f"{prefix}_bytes"] = raw
        return
    if include_images == "bytes":
        raw = _coerce_image_bytes(value)
        if raw is not None:
            row[f"{prefix}_bytes"] = raw


def _coerce_image_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)

    # Torch tensors and numpy arrays both commonly expose these conversion APIs.
    obj = value
    if hasattr(obj, "detach"):
        try:
            obj = obj.detach()
        except Exception:
            pass
    if hasattr(obj, "cpu"):
        try:
            obj = obj.cpu()
        except Exception:
            pass
    if hasattr(obj, "numpy"):
        try:
            arr = obj.numpy()
            if hasattr(arr, "tobytes"):
                return arr.tobytes()
        except Exception:
            pass
    if hasattr(obj, "tobytes"):
        try:
            return obj.tobytes()
        except Exception:
            return None
    return None


def _get_any(sample: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _nested(sample: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = sample
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            arr = value.detach().cpu().numpy()
            if getattr(arr, "ndim", None) == 1:
                return [float(v) for v in arr.tolist()]
        except Exception:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [float(v) for v in value]
    return None


def _candidate_root(dataset: Any) -> Path | None:
    root = getattr(dataset, "root", None)
    if root is None:
        return None
    path = Path(root)
    if not path.exists():
        return None
    # LeRobot datasets usually store trajectory episodes under root/data/**.parquet.
    # Prefer this folder to avoid pulling meta parquet files with incompatible schemas.
    data_dir = path / "data"
    if data_dir.exists() and any(data_dir.rglob("*.parquet")):
        return data_dir
    if any(path.rglob("*.parquet")):
        return path
    return None


def _attach_video_path_refs(frame: pl.LazyFrame, dataset: Any) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    if "episode_index" not in names:
        return frame

    meta = getattr(dataset, "meta", None)
    if meta is None:
        return frame
    video_keys = _video_meta_keys(dataset)
    if not video_keys:
        return frame

    root = Path(getattr(dataset, "root", ""))
    episodes = getattr(meta, "episodes", None)
    if episodes is None:
        return frame

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        if not isinstance(ep, Mapping):
            continue
        episode_index = ep.get("episode_index")
        if episode_index is None:
            continue
        row: dict[str, Any] = {"episode_index": int(episode_index)}
        rgb_idx = 0
        depth_idx = 0
        for cam_idx, video_key in enumerate(video_keys):
            try:
                rel_path = meta.get_video_file_path(int(episode_index), str(video_key))
            except Exception:
                continue
            if rel_path is None:
                continue
            full = str(root / rel_path)
            row[f"image_{cam_idx}_path"] = full
            if _is_depth_key(str(video_key)):
                row[f"image_depth_{depth_idx}_path"] = full
                if depth_idx == 0:
                    row["image_depth_path"] = full
                    row["image_1_path"] = full
                depth_idx += 1
            else:
                row[f"image_rgb_{rgb_idx}_path"] = full
                if rgb_idx == 0:
                    row["image_rgb_path"] = full
                    row["image_0_path"] = full
                rgb_idx += 1
        rows.append(row)

    if not rows:
        return frame

    refs = pl.DataFrame(rows).lazy()
    existing_cols = [c for c in names if c.startswith("image_") and c.endswith("_path")]
    out = frame.join(refs, on="episode_index", how="left", suffix="_meta")
    if not existing_cols:
        return out

    exprs: list[pl.Expr] = []
    drop_cols: list[str] = []
    out_names = set(out.collect_schema().names())
    for col in existing_cols:
        meta_col = f"{col}_meta"
        if meta_col in out_names:
            exprs.append(pl.coalesce(pl.col(col), pl.col(meta_col)).alias(col))
            drop_cols.append(meta_col)
    if exprs:
        out = out.with_columns(exprs)
    return out.drop(drop_cols) if drop_cols else out


def _video_meta_keys(dataset: Any) -> list[str]:
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return []
    keys = getattr(meta, "video_keys", None)
    if not isinstance(keys, Sequence):
        return []
    return [str(k) for k in keys if isinstance(k, str) and k]


def _is_depth_key(name: str) -> bool:
    return "depth" in name.lower()
