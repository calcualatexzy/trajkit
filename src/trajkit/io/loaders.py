"""I/O loaders for trajectory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import polars as pl

from trajkit.utils.typing import PathLike
from trajkit.utils.validation import ensure_required_columns

ImageMode = Literal["none", "paths", "bytes"]
SUPPORTED_EXTENSIONS = (".parquet", ".csv", ".json", ".jsonl", ".npy", ".npz")
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t", "timestamp", "time", "ts"),
    "x": ("x", "tx", "pos_x", "position_x"),
    "y": ("y", "ty", "pos_y", "position_y"),
    "z": ("z", "tz", "pos_z", "position_z"),
    "trajectory_id": ("trajectory_id", "episode_index", "episode_idx"),
}


def load_dataset(
    path: PathLike,
    include_states: bool = True,
    include_actions: bool = True,
    include_images: ImageMode = "paths",
) -> pl.DataFrame:
    return load_dataset_lazy(
        path=path,
        include_states=include_states,
        include_actions=include_actions,
        include_images=include_images,
    ).collect()


def load_dataset_lazy(
    path: PathLike,
    include_states: bool = True,
    include_actions: bool = True,
    include_images: ImageMode = "paths",
) -> pl.LazyFrame:
    path = Path(path)
    files = list(_discover_files(path))
    if not files:
        raise FileNotFoundError(f"no supported files under {path}")

    frames: list[pl.LazyFrame] = []
    for file in files:
        try:
            frame = _normalize_table_lazy(
                frame=_load_single_file_lazy(file),
                default_trajectory_id=file.stem,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            )
        except ValueError:
            # Skip metadata blobs (e.g. task jsonl) that are not step tables.
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"no trajectory tables discovered under {path}")
    frame = pl.concat(frames, how="diagonal_relaxed")
    cast_exprs: list[pl.Expr] = [
        pl.col("trajectory_id").cast(pl.String),
        pl.col("t").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ]
    names = frame.collect_schema().names()
    if "state_vec" in names:
        cast_exprs.append(pl.col("state_vec").cast(pl.List(pl.Float64)))
    if "action_vec" in names:
        cast_exprs.append(pl.col("action_vec").cast(pl.List(pl.Float64)))
    return frame.with_columns(cast_exprs)


def _discover_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return
    for ext in SUPPORTED_EXTENSIONS:
        for f in sorted(path.rglob(f"*{ext}")):
            if f.is_file():
                yield f


def _load_single_file_lazy(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix in (".json", ".jsonl"):
        if suffix == ".jsonl":
            return pl.scan_ndjson(path)
        return pl.read_json(path).lazy()
    if suffix in (".npy", ".npz"):
        return _load_numpy(path).lazy()
    raise ValueError(f"unsupported file extension: {suffix}")


def _load_numpy(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return _nparray_to_frame(arr)
    data = np.load(path)
    if "arr_0" in data:
        return _nparray_to_frame(data["arr_0"])
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"empty npz file: {path}")
    return _nparray_to_frame(data[keys[0]])


def _nparray_to_frame(arr: np.ndarray) -> pl.DataFrame:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected ndarray of shape (n, d>=2)")
    if arr.shape[1] == 2:
        cols = {"x": arr[:, 0], "y": arr[:, 1]}
    elif arr.shape[1] == 3:
        cols = {"t": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2]}
    else:
        cols = {"t": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2], "z": arr[:, 3]}
    return pl.DataFrame(cols)


def _normalize_table_lazy(
    frame: pl.LazyFrame,
    default_trajectory_id: str,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> pl.LazyFrame:
    out = frame
    out = _extract_vla_modalities(out, include_states=include_states, include_actions=include_actions, include_images=include_images)
    out = _rename_aliases_lazy(out)
    names = set(out.collect_schema().names())

    if "trajectory_id" not in names:
        out = out.with_columns(pl.lit(default_trajectory_id).alias("trajectory_id"))
        names.add("trajectory_id")
    if "t" not in names:
        out = out.with_row_count(name="t", offset=0)
        names.add("t")

    # Recover spatial proxies from state vectors if x/y/z are absent.
    if ("x" not in names or "y" not in names) and "state_vec" in names:
        out = out.with_columns(
            pl.col("state_vec").list.get(0).alias("x"),
            pl.col("state_vec").list.get(1).alias("y"),
            pl.col("state_vec").list.get(2).fill_null(0.0).alias("z"),
        )
        names.update({"x", "y", "z"})
    if "z" not in names:
        out = out.with_columns(pl.lit(0.0).alias("z"))
        names.add("z")
    if "x" not in names or "y" not in names:
        raise ValueError("input must provide x/y or recoverable state vectors")

    if "frame_index" in names:
        out = out.with_columns(pl.col("frame_index").cast(pl.Int64).alias("frame_id"))
    else:
        # Canonical frame id for fast frame-level retrieval.
        out = out.sort(["trajectory_id", "t"]).with_columns(pl.int_range(0, pl.len()).over("trajectory_id").alias("frame_id"))

    ensure_required_columns(tuple(out.collect_schema().names()))

    keep = ["trajectory_id", "frame_id", "t", "x", "y", "z"]
    optional_order = [
        "state_vec",
        "action_vec",
        "frame_index",
        "episode_index",
        "task_index",
        "label",
        "image_0_path",
        "image_1_path",
        "image_0_bytes",
        "image_1_bytes",
    ]
    current = set(out.collect_schema().names())
    keep.extend([col for col in optional_order if col in current])
    out = out.select(keep)
    return _canonicalize_types(out)


def _rename_aliases_lazy(frame: pl.LazyFrame) -> pl.LazyFrame:
    out = frame
    current = set(out.collect_schema().names())
    rename_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        if canonical in current:
            continue
        for alias in aliases:
            if alias in current:
                rename_map[alias] = canonical
                current.add(canonical)
                break
    return out.rename(rename_map) if rename_map else out


def _extract_vla_modalities(
    frame: pl.LazyFrame,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    exprs: list[pl.Expr] = []

    # Bridge-like nested struct format.
    if {"state", "episode_idx"}.issubset(names):
        exprs.extend(
            [
                pl.col("episode_idx").cast(pl.String).alias("trajectory_id"),
                pl.coalesce(pl.col("timestamp"), pl.col("step_idx").cast(pl.Float64)).alias("t"),
                pl.col("state").struct.field("end_effector_pose").struct.field("x").alias("x"),
                pl.col("state").struct.field("end_effector_pose").struct.field("y").alias("y"),
                pl.col("state").struct.field("end_effector_pose").struct.field("z").alias("z"),
            ]
        )
        if include_states:
            exprs.append(
                pl.concat_list(
                    [
                        pl.col("state").struct.field("end_effector_pose").struct.field("x"),
                        pl.col("state").struct.field("end_effector_pose").struct.field("y"),
                        pl.col("state").struct.field("end_effector_pose").struct.field("z"),
                        pl.col("state").struct.field("end_effector_pose").struct.field("roll"),
                        pl.col("state").struct.field("end_effector_pose").struct.field("pitch"),
                        pl.col("state").struct.field("end_effector_pose").struct.field("yaw"),
                    ]
                ).alias("state_vec")
            )
        if include_actions and "action" in names:
            exprs.append(
                pl.concat_list(
                    [
                        pl.col("action").struct.field("pose").struct.field("x"),
                        pl.col("action").struct.field("pose").struct.field("y"),
                        pl.col("action").struct.field("pose").struct.field("z"),
                        pl.col("action").struct.field("pose").struct.field("roll"),
                        pl.col("action").struct.field("pose").struct.field("pitch"),
                        pl.col("action").struct.field("pose").struct.field("yaw"),
                        pl.col("action").struct.field("grasp"),
                    ]
                ).alias("action_vec")
            )

    # LeRobot parquet format (libero/community).
    if "episode_index" in names:
        exprs.append(pl.col("episode_index").cast(pl.String).alias("trajectory_id"))
    if "timestamp" in names:
        exprs.append(pl.col("timestamp").cast(pl.Float64).alias("t"))
    if include_states and "observation.state" in names:
        exprs.append(pl.col("observation.state").cast(pl.List(pl.Float64)).alias("state_vec"))
    if include_actions and "action" in names:
        exprs.append(pl.col("action").cast(pl.List(pl.Float64)).alias("action_vec"))

    # Try spatial recovery from observation state vector (x/y/z are first 3 entries for many VLA datasets).
    if "observation.state" in names:
        state_list = pl.col("observation.state").cast(pl.List(pl.Float64))
        exprs.extend(
            [
                state_list.list.get(0).alias("x"),
                state_list.list.get(1).alias("y"),
                state_list.list.get(2).fill_null(0.0).alias("z"),
            ]
        )

    if include_images != "none":
        image_specs = [
            ("observation.images.image", "image_0"),
            ("observation.images.image2", "image_1"),
            ("observation.image", "image_0"),
            ("observation.image2", "image_1"),
            ("image", "image_0"),
        ]
        for source, target in image_specs:
            if source in names:
                exprs.append(pl.col(source).struct.field("path").alias(f"{target}_path"))
                if include_images == "bytes":
                    exprs.append(pl.col(source).struct.field("bytes").alias(f"{target}_bytes"))

    passthrough = [col for col in ("frame_index", "episode_index", "task_index", "label") if col in names]
    exprs.extend(pl.col(col) for col in passthrough)
    return frame.with_columns(exprs) if exprs else frame


def _canonicalize_types(frame: pl.LazyFrame) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    casts: list[pl.Expr] = []

    def maybe_cast(col: str, dtype: pl.DataType) -> None:
        if col in names:
            casts.append(pl.col(col).cast(dtype, strict=False).alias(col))

    maybe_cast("trajectory_id", pl.String)
    maybe_cast("frame_id", pl.Int64)
    maybe_cast("t", pl.Float64)
    maybe_cast("x", pl.Float64)
    maybe_cast("y", pl.Float64)
    maybe_cast("z", pl.Float64)
    maybe_cast("frame_index", pl.Int64)
    maybe_cast("episode_index", pl.Int64)
    maybe_cast("task_index", pl.Int64)
    maybe_cast("label", pl.String)
    maybe_cast("state_vec", pl.List(pl.Float64))
    maybe_cast("action_vec", pl.List(pl.Float64))
    maybe_cast("image_0_path", pl.String)
    maybe_cast("image_1_path", pl.String)
    maybe_cast("image_0_bytes", pl.Binary)
    maybe_cast("image_1_bytes", pl.Binary)

    if not casts:
        return frame
    return frame.with_columns(casts)
