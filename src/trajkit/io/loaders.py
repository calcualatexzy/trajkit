"""I/O loaders for trajectory datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import polars as pl

from trajkit.utils.typing import PathLike
from trajkit.utils.validation import ensure_required_columns

ImageMode = Literal["none", "paths", "bytes"]
SUPPORTED_EXTENSIONS = (".parquet", ".csv", ".json", ".jsonl", ".npy", ".npz", ".hdf5", ".h5", ".rlds")
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
    return load_dataset_lazy(path, include_states=include_states, include_actions=include_actions, include_images=include_images).collect()


def load_dataset_lazy(
    path: PathLike,
    include_states: bool = True,
    include_actions: bool = True,
    include_images: ImageMode = "paths",
) -> pl.LazyFrame:
    files = list(_discover_files(Path(path)))
    if not files:
        raise FileNotFoundError(f"no supported files under {path}")

    frames: list[pl.LazyFrame] = []
    for file in files:
        try:
            loaded = _load_single_file_lazy(
                file,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            )
            frames.append(
                _normalize_table_lazy(
                    loaded,
                    default_trajectory_id=file.stem,
                    include_states=include_states,
                    include_actions=include_actions,
                    include_images=include_images,
                )
            )
        except ValueError:
            continue
    if not frames:
        raise ValueError(f"no trajectory tables discovered under {path}")

    frame = pl.concat(frames, how="diagonal_relaxed")
    names = set(_schema_names(frame))
    casts: list[pl.Expr] = [
        pl.col("trajectory_id").cast(pl.String),
        pl.col("t").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ]
    if "state_vec" in names:
        casts.append(pl.col("state_vec").cast(pl.List(pl.Float64)))
    if "action_vec" in names:
        casts.append(pl.col("action_vec").cast(pl.List(pl.Float64)))
    return frame.with_columns(casts)


def _discover_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return
    for ext in SUPPORTED_EXTENSIONS:
        for f in sorted(path.rglob(f"*{ext}")):
            if f.is_file():
                yield f


def _load_single_file_lazy(
    path: Path,
    *,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix in (".json", ".jsonl"):
        return pl.scan_ndjson(path) if suffix == ".jsonl" else pl.read_json(path).lazy()
    if suffix in (".npy", ".npz"):
        return _load_numpy(path).lazy()
    if suffix in (".hdf5", ".h5"):
        return _load_hdf5(path, include_states=include_states, include_actions=include_actions, include_images=include_images).lazy()
    if suffix == ".rlds":
        return _load_rlds(path, include_states=include_states, include_actions=include_actions, include_images=include_images).lazy()
    raise ValueError(f"unsupported file extension: {suffix}")


def _load_numpy(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".npy":
        return _nparray_to_frame(np.load(path))
    data = np.load(path)
    key = "arr_0" if "arr_0" in data else (list(data.keys())[0] if data.keys() else None)
    if key is None:
        raise ValueError(f"empty npz file: {path}")
    return _nparray_to_frame(data[key])


def _nparray_to_frame(arr: np.ndarray) -> pl.DataFrame:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected ndarray of shape (n, d>=2)")
    if arr.shape[1] == 2:
        return pl.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
    if arr.shape[1] == 3:
        return pl.DataFrame({"t": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2]})
    return pl.DataFrame({"t": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2], "z": arr[:, 3]})


def _load_hdf5(path: Path, *, include_states: bool, include_actions: bool, include_images: ImageMode) -> pl.DataFrame:
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("loading .hdf5/.h5 requires h5py") from exc

    def group_to_frame(group: Any, trajectory_id: str) -> pl.DataFrame | None:
        t = _h5_get_array(group, "t", "timestamp", "time")
        x = _h5_get_array(group, "x", "position/x", "pos/x")
        y = _h5_get_array(group, "y", "position/y", "pos/y")
        z = _h5_get_array(group, "z", "position/z", "pos/z")

        state = _h5_get_array(group, "state_vec", "observation/state", "observation.state", "state") if include_states else None
        action = _h5_get_array(group, "action_vec", "action", "actions") if include_actions else None

        n = _first_nonzero_len(t, x, y, z, state, action)
        if n <= 0:
            return None
        if t is None:
            t = np.arange(n, dtype=np.float64)

        if (x is None or y is None) and state is not None and state.ndim == 2 and state.shape[1] >= 2:
            x = state[:, 0]
            y = state[:, 1]
            z = state[:, 2] if state.shape[1] >= 3 else np.zeros(n, dtype=np.float64)
        if x is None or y is None:
            return None
        if z is None:
            z = np.zeros(n, dtype=np.float64)

        cols: dict[str, object] = {
            "trajectory_id": np.full(n, str(trajectory_id), dtype=object),
            "frame_index": np.arange(n, dtype=np.int64),
            "t": np.asarray(t).reshape(-1)[:n],
            "x": np.asarray(x).reshape(-1)[:n],
            "y": np.asarray(y).reshape(-1)[:n],
            "z": np.asarray(z).reshape(-1)[:n],
        }
        if include_states and state is not None and state.ndim == 2:
            cols["state_vec"] = np.asarray(state[:n], dtype=np.float64).tolist()
        if include_actions and action is not None and action.ndim == 2:
            cols["action_vec"] = np.asarray(action[:n], dtype=np.float64).tolist()

        if include_images != "none":
            rgb_path = _h5_get_array(group, "image_rgb_path", "image/path", "rgb/path")
            depth_path = _h5_get_array(group, "image_depth_path", "depth/path")
            if rgb_path is not None:
                vals = _decode_h5_strings(rgb_path)[:n]
                cols["image_rgb_path"] = vals
                cols["image_0_path"] = vals
            if depth_path is not None:
                vals = _decode_h5_strings(depth_path)[:n]
                cols["image_depth_path"] = vals
                cols["image_1_path"] = vals
            if include_images == "bytes":
                rgb_bytes = _h5_get_array(group, "image_rgb_bytes", "image/bytes", "rgb/bytes")
                depth_bytes = _h5_get_array(group, "image_depth_bytes", "depth/bytes")
                if rgb_bytes is not None:
                    vals = [bytes(v) if v is not None else None for v in rgb_bytes[:n]]
                    cols["image_rgb_bytes"] = vals
                    cols["image_0_bytes"] = vals
                if depth_bytes is not None:
                    vals = [bytes(v) if v is not None else None for v in depth_bytes[:n]]
                    cols["image_depth_bytes"] = vals
                    cols["image_1_bytes"] = vals
        return pl.DataFrame(cols)

    with h5py.File(path, "r") as h5:
        root = group_to_frame(h5, trajectory_id=path.stem)
        if root is not None:
            return root
        parts: list[pl.DataFrame] = []
        for key in sorted(h5.keys()):
            obj = h5.get(key)
            if obj is None or not hasattr(obj, "keys"):
                continue
            part = group_to_frame(obj, trajectory_id=key)
            if part is not None and not part.is_empty():
                parts.append(part)
        if not parts:
            raise ValueError(f"no trajectory table discovered in hdf5 file: {path}")
        return pl.concat(parts, how="diagonal_relaxed")


def _load_rlds(path: Path, *, include_states: bool, include_actions: bool, include_images: ImageMode) -> pl.DataFrame:
    batch_size = 4096
    frames: list[pl.DataFrame] = []
    rows: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal rows
        if rows:
            frames.append(pl.DataFrame(rows))
            rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            ep = str(item.get("episode_id", item.get("episode_idx", item.get("trajectory_id", "0"))))

            if isinstance(item.get("steps"), list):
                for i, step in enumerate(item["steps"]):
                    if not isinstance(step, dict):
                        continue
                    row = _rlds_step_to_row(
                        step,
                        trajectory_id=ep,
                        frame_index=i,
                        include_states=include_states,
                        include_actions=include_actions,
                        include_images=include_images,
                    )
                    if row is not None:
                        rows.append(row)
            else:
                idx = int(item.get("frame_index", item.get("step_idx", 0)))
                row = _rlds_step_to_row(
                    item,
                    trajectory_id=ep,
                    frame_index=idx,
                    include_states=include_states,
                    include_actions=include_actions,
                    include_images=include_images,
                )
                if row is not None:
                    rows.append(row)
            if len(rows) >= batch_size:
                flush()
    flush()
    if not frames:
        raise ValueError(f"no rows parsed from rlds file: {path}")
    return pl.concat(frames, how="diagonal_relaxed")


def _rlds_step_to_row(
    step: dict[str, object],
    *,
    trajectory_id: str,
    frame_index: int,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> dict[str, object] | None:
    state = step.get("state_vec")
    action = step.get("action_vec")
    x = step.get("x")
    y = step.get("y")
    z = step.get("z", 0.0)
    if (x is None or y is None) and isinstance(state, list) and len(state) >= 2:
        x, y = state[0], state[1]
        z = state[2] if len(state) >= 3 else 0.0
    if x is None or y is None:
        return None

    row: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "frame_index": int(frame_index),
        "t": float(step.get("timestamp", step.get("t", float(frame_index)))),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }
    if include_states and isinstance(state, list):
        row["state_vec"] = [float(v) for v in state]
    if include_actions and isinstance(action, list):
        row["action_vec"] = [float(v) for v in action]
    if include_images != "none":
        rgb = step.get("image_rgb_path")
        depth = step.get("image_depth_path")
        if isinstance(rgb, str):
            row["image_rgb_path"] = rgb
            row["image_0_path"] = rgb
        if isinstance(depth, str):
            row["image_depth_path"] = depth
            row["image_1_path"] = depth
    return row


def _h5_get_array(group: Any, *keys: str) -> np.ndarray | None:
    for key in keys:
        cur = group
        ok = True
        for part in key.split("/"):
            if not hasattr(cur, "__contains__") or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok and hasattr(cur, "shape"):
            try:
                return np.asarray(cur)
            except Exception:
                pass
    return None


def _first_nonzero_len(*arrs: np.ndarray | None) -> int:
    for arr in arrs:
        if arr is not None and getattr(arr, "ndim", 0) > 0 and arr.shape[0] > 0:
            return int(arr.shape[0])
    return 0


def _decode_h5_strings(arr: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for v in arr:
        if v is None:
            out.append(None)
        elif isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def _normalize_table_lazy(
    frame: pl.LazyFrame,
    default_trajectory_id: str,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> pl.LazyFrame:
    out = _extract_vla_modalities(frame, include_states=include_states, include_actions=include_actions, include_images=include_images)
    out = _rename_aliases_lazy(out)
    names = set(_schema_names(out))

    if "trajectory_id" not in names:
        out = out.with_columns(pl.lit(default_trajectory_id).alias("trajectory_id"))
        names.add("trajectory_id")
    if "t" not in names:
        out = out.with_row_count(name="t", offset=0)
        names.add("t")

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
        out = out.sort(["trajectory_id", "t"]).with_columns(pl.int_range(0, pl.len()).over("trajectory_id").alias("frame_id"))

    ensure_required_columns(tuple(_schema_names(out)))
    current = set(_schema_names(out))
    keep = ["trajectory_id", "frame_id", "t", "x", "y", "z"]
    prefer = [
        "state_vec",
        "action_vec",
        "frame_index",
        "episode_index",
        "task_index",
        "label",
        "image_rgb_path",
        "image_depth_path",
        "image_rgb_bytes",
        "image_depth_bytes",
        "image_0_path",
        "image_1_path",
        "image_0_bytes",
        "image_1_bytes",
    ]
    keep.extend([c for c in prefer if c in current])
    keep.extend(
        [
            c
            for c in sorted(current)
            if c.startswith("image_") and (c.endswith("_path") or c.endswith("_bytes")) and c not in keep
        ]
    )
    return _canonicalize_types(out.select(keep))


def _rename_aliases_lazy(frame: pl.LazyFrame) -> pl.LazyFrame:
    names = set(_schema_names(frame))
    rename_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        if canonical in names:
            continue
        for alias in aliases:
            if alias in names:
                rename_map[alias] = canonical
                names.add(canonical)
                break
    return frame.rename(rename_map) if rename_map else frame


def _extract_vla_modalities(
    frame: pl.LazyFrame,
    include_states: bool,
    include_actions: bool,
    include_images: ImageMode,
) -> pl.LazyFrame:
    names = set(_schema_names(frame))
    exprs: list[pl.Expr] = []

    is_bridge = {"state", "episode_idx"}.issubset(names)
    if is_bridge:
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
    else:
        if "episode_index" in names:
            exprs.append(pl.col("episode_index").cast(pl.String).alias("trajectory_id"))
        if "timestamp" in names:
            exprs.append(pl.col("timestamp").cast(pl.Float64).alias("t"))
        if include_states and "observation.state" in names:
            exprs.append(pl.col("observation.state").cast(pl.List(pl.Float64)).alias("state_vec"))
        if include_actions and "action" in names:
            exprs.append(pl.col("action").cast(pl.List(pl.Float64)).alias("action_vec"))
        if "observation.state" in names:
            s = pl.col("observation.state").cast(pl.List(pl.Float64))
            exprs.extend([s.list.get(0).alias("x"), s.list.get(1).alias("y"), s.list.get(2).fill_null(0.0).alias("z")])

    if include_images != "none":
        image_specs = [
            ("observation.images.rgb", "rgb"),
            ("observation.rgb", "rgb"),
            ("observation.images.image", "rgb"),
            ("observation.image", "rgb"),
            ("image", "rgb"),
            ("rgb", "rgb"),
            ("observation.images.image2", "rgb"),
            ("observation.image2", "rgb"),
            ("image2", "rgb"),
            ("observation.images.depth", "depth"),
            ("observation.depth", "depth"),
            ("observation.images.image_depth", "depth"),
            ("observation.image_depth", "depth"),
            ("depth", "depth"),
        ]
        rgb_i, depth_i, cam_i = 0, 0, 0
        for source, kind in image_specs:
            if source not in names:
                continue
            path_expr = pl.col(source).struct.field("path")
            exprs.append(path_expr.alias(f"image_{cam_i}_path"))
            if include_images == "bytes":
                bytes_expr = pl.col(source).struct.field("bytes")
                exprs.append(bytes_expr.alias(f"image_{cam_i}_bytes"))

            if kind == "depth":
                exprs.append(path_expr.alias(f"image_depth_{depth_i}_path"))
                if depth_i == 0:
                    exprs.append(path_expr.alias("image_depth_path"))
                if include_images == "bytes":
                    exprs.append(bytes_expr.alias(f"image_depth_{depth_i}_bytes"))
                    if depth_i == 0:
                        exprs.append(bytes_expr.alias("image_depth_bytes"))
                depth_i += 1
            else:
                exprs.append(path_expr.alias(f"image_rgb_{rgb_i}_path"))
                if rgb_i == 0:
                    exprs.append(path_expr.alias("image_rgb_path"))
                if include_images == "bytes":
                    exprs.append(bytes_expr.alias(f"image_rgb_{rgb_i}_bytes"))
                    if rgb_i == 0:
                        exprs.append(bytes_expr.alias("image_rgb_bytes"))
                rgb_i += 1
            cam_i += 1

    exprs.extend(pl.col(c) for c in ("frame_index", "episode_index", "task_index", "label") if c in names)
    return frame.with_columns(exprs) if exprs else frame


def _canonicalize_types(frame: pl.LazyFrame) -> pl.LazyFrame:
    names = set(_schema_names(frame))
    casts: list[pl.Expr] = []

    def maybe(col: str, dtype: pl.DataType) -> None:
        if col in names:
            casts.append(pl.col(col).cast(dtype, strict=False).alias(col))

    maybe("trajectory_id", pl.String)
    maybe("frame_id", pl.Int64)
    maybe("frame_index", pl.Int64)
    maybe("episode_index", pl.Int64)
    maybe("task_index", pl.Int64)
    maybe("label", pl.String)
    maybe("t", pl.Float64)
    maybe("x", pl.Float64)
    maybe("y", pl.Float64)
    maybe("z", pl.Float64)
    maybe("state_vec", pl.List(pl.Float64))
    maybe("action_vec", pl.List(pl.Float64))
    for c in sorted(names):
        if c.startswith("image_") and c.endswith("_path"):
            maybe(c, pl.String)
        if c.startswith("image_") and c.endswith("_bytes"):
            maybe(c, pl.Binary)
    return frame.with_columns(casts) if casts else frame


def _schema_names(frame: pl.LazyFrame) -> list[str]:
    return frame.collect_schema().names()
