"""Simple numpy-based clustering for trajectory features."""

from __future__ import annotations

import numpy as np
import polars as pl

from trajkit.analysis.summary import per_trajectory_features
from trajkit.core.schema import ClusterResult

DEFAULT_CLUSTER_FEATURES = [
    "path_length",
    "duration",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "stop_ratio",
    "straightness",
]


def ensure_feature_frame(frame_or_features: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(frame_or_features, pl.LazyFrame):
        names = set(frame_or_features.collect_schema().names())
    else:
        names = set(frame_or_features.columns)
    required = {"trajectory_id", *DEFAULT_CLUSTER_FEATURES}
    if required.issubset(names):
        return frame_or_features.collect() if isinstance(frame_or_features, pl.LazyFrame) else frame_or_features
    return per_trajectory_features(frame_or_features)


def feature_matrix(frame_or_features: pl.DataFrame | pl.LazyFrame) -> tuple[list[str], np.ndarray]:
    feat = ensure_feature_frame(frame_or_features)
    if feat.is_empty():
        return [], np.empty((0, len(DEFAULT_CLUSTER_FEATURES)), dtype=float)
    ids = feat["trajectory_id"].to_list()
    x = feat.select(DEFAULT_CLUSTER_FEATURES).to_numpy().astype(float)
    return ids, x


def cluster_trajectories(frame: pl.DataFrame | pl.LazyFrame, k: int = 4, seed: int = 0, max_iter: int = 50) -> ClusterResult:
    ids, x = feature_matrix(frame)
    if len(ids) == 0:
        return ClusterResult(labels={}, centroids=[], medoid_ids={})

    x = _zscore(x)

    k = max(1, min(k, x.shape[0]))
    centroids = _kmeans(x, k=k, seed=seed, max_iter=max_iter)
    dist = _pairwise_l2(x, centroids)
    label_idx = np.argmin(dist, axis=1)

    labels = {str(tid): int(cid) for tid, cid in zip(ids, label_idx.tolist())}
    medoid_ids = _medoids_for_clusters(ids, x, label_idx, centroids)
    return ClusterResult(
        labels=labels,
        centroids=[row.tolist() for row in centroids],
        medoid_ids=medoid_ids,
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0)
    std[std == 0.0] = 1.0
    return (x - np.mean(x, axis=0)) / std


def _kmeans(x: np.ndarray, k: int, seed: int, max_iter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    for _ in range(max_iter):
        dist = _pairwise_l2(x, centroids)
        labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for idx in range(k):
            members = x[labels == idx]
            if members.size == 0:
                continue
            new_centroids[idx] = np.mean(members, axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))


def _medoids_for_clusters(
    ids: list[str],
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        d = np.linalg.norm(x[idx] - centroids[cid], axis=1)
        out[int(cid)] = str(ids[idx[int(np.argmin(d))]])
    return out
