"""Reusable plotting helpers."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def plot_start_end_distribution(frame: pl.DataFrame, ax=None):
    import matplotlib.pyplot as plt

    use_ax = ax if ax is not None else plt.subplots(figsize=(7, 5))[1]
    starts = frame.sort(["trajectory_id", "t"]).group_by("trajectory_id").first()
    ends = frame.sort(["trajectory_id", "t"]).group_by("trajectory_id").last()
    use_ax.scatter(starts["x"].to_numpy(), starts["y"].to_numpy(), s=12, alpha=0.7, label="start")
    use_ax.scatter(ends["x"].to_numpy(), ends["y"].to_numpy(), s=12, alpha=0.7, label="end")
    use_ax.set_title("Start/end distribution")
    use_ax.set_xlabel("x")
    use_ax.set_ylabel("y")
    use_ax.legend()
    return use_ax


def plot_feature_distributions(feature_frame: pl.DataFrame):
    import matplotlib.pyplot as plt

    cols = [
        c
        for c in feature_frame.columns
        if c != "trajectory_id" and feature_frame[c].dtype.is_numeric()
    ]
    n = len(cols)
    if n == 0:
        raise ValueError("no numeric feature columns to plot")
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes_1d = np.array(axes).reshape(-1)
    for idx, col in enumerate(cols):
        ax = axes_1d[idx]
        vals = feature_frame[col].to_numpy()
        ax.hist(vals, bins=30, alpha=0.8)
        ax.set_title(col)
    for idx in range(n, len(axes_1d)):
        axes_1d[idx].axis("off")
    fig.tight_layout()
    return fig


def plot_outlier_ranking(outliers: list[dict[str, float | str]], ax=None):
    import matplotlib.pyplot as plt

    if not outliers:
        raise ValueError("outliers list is empty")
    use_ax = ax if ax is not None else plt.subplots(figsize=(8, 4))[1]
    ids = [str(x["trajectory_id"]) for x in outliers]
    scores = [float(x["outlier_score"]) for x in outliers]
    y = np.arange(len(ids))
    use_ax.barh(y, scores)
    use_ax.set_yticks(y)
    use_ax.set_yticklabels(ids)
    use_ax.invert_yaxis()
    use_ax.set_xlabel("outlier score")
    use_ax.set_title("Outlier ranking")
    return use_ax


def plot_cluster_embedding(feature_frame: pl.DataFrame, labels: dict[str, int], ax=None):
    import matplotlib.pyplot as plt

    cols = [c for c in feature_frame.columns if c != "trajectory_id" and feature_frame[c].dtype.is_numeric()]
    if not cols:
        raise ValueError("feature frame has no numeric columns")
    ids = feature_frame["trajectory_id"].to_list()
    x = feature_frame.select(cols).to_numpy().astype(float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    emb = u[:, :2] * s[:2]

    use_ax = ax if ax is not None else plt.subplots(figsize=(7, 5))[1]
    label_vec = np.array([labels.get(str(tid), -1) for tid in ids], dtype=int)
    unique = sorted(set(label_vec.tolist()))
    cmap = plt.cm.get_cmap("tab10", max(1, len(unique)))
    for i, cid in enumerate(unique):
        mask = label_vec == cid
        use_ax.scatter(emb[mask, 0], emb[mask, 1], s=24, alpha=0.8, label=f"cluster {cid}", color=cmap(i))
    use_ax.set_xlabel("embedding dim 1")
    use_ax.set_ylabel("embedding dim 2")
    use_ax.set_title("Cluster embedding (SVD)")
    use_ax.legend()
    return use_ax


def plot_modality_coverage(frame: pl.DataFrame, ax=None):
    import matplotlib.pyplot as plt

    cols = [c for c in ("state_vec", "action_vec", "image_0_path", "image_1_path") if c in frame.columns]
    if not cols:
        raise ValueError("no modality columns found")
    use_ax = ax if ax is not None else plt.subplots(figsize=(6, 4))[1]
    cov = [(1.0 - float(frame[c].null_count()) / max(1, frame.height)) for c in cols]
    use_ax.bar(cols, cov)
    use_ax.set_ylim(0.0, 1.05)
    use_ax.set_ylabel("coverage")
    use_ax.set_title("Modality coverage")
    return use_ax
