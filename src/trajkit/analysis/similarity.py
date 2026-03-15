"""Trajectory similarity search using feature distance."""

from __future__ import annotations

import numpy as np
import polars as pl

from trajkit.analysis.clustering import feature_matrix


def nearest_neighbors(
    source_id: str,
    frame: pl.DataFrame | pl.LazyFrame,
    k: int = 5,
) -> list[dict[str, float | str]]:
    ids, x = feature_matrix(frame)
    if len(ids) == 0:
        return []
    if source_id not in ids:
        raise KeyError(f"trajectory_id {source_id!r} not found")
    center = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0.0] = 1.0
    z = (x - center) / std

    src_idx = ids.index(source_id)
    d = np.linalg.norm(z - z[src_idx], axis=1)
    order = np.argsort(d)

    out: list[dict[str, float | str]] = []
    for idx in order:
        if idx == src_idx:
            continue
        out.append({"trajectory_id": str(ids[idx]), "distance": float(d[idx])})
        if len(out) >= k:
            break
    return out
