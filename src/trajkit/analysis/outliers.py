"""Outlier scoring for trajectories."""

from __future__ import annotations

import numpy as np
import polars as pl

from trajkit.analysis.clustering import feature_matrix


def find_outliers(frame: pl.DataFrame | pl.LazyFrame, top_k: int = 10) -> list[dict[str, float | str]]:
    ids, x = feature_matrix(frame)
    if len(ids) == 0:
        return []

    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center), axis=0)
    mad[mad == 0.0] = 1.0
    z = np.abs((x - center) / mad)
    score = np.mean(z, axis=1)

    order = np.argsort(score)[::-1][:top_k]
    return [{"trajectory_id": str(ids[i]), "outlier_score": float(score[i])} for i in order]
