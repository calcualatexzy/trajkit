"""Trajectory dataset object and public API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl

from trajkit.analysis.clustering import cluster_trajectories
from trajkit.analysis.compare import compare_frames
from trajkit.analysis.outliers import find_outliers as find_outliers_impl
from trajkit.analysis.summary import per_trajectory_features, summarize_dataset
from trajkit.core.schema import ClusterResult, FrameBatchView, FrameRecord
from trajkit.core.trajectory import Trajectory
from trajkit.io.lerobot import detect_lerobot_loader_mode, load_lerobot_dataset
from trajkit.io.loaders import load_dataset_lazy
from trajkit.utils.typing import PathLike
from trajkit.viz.plots import (
    plot_cluster_embedding,
    plot_feature_distributions,
    plot_modality_coverage,
    plot_outlier_ranking,
    plot_start_end_distribution,
)


class TrajectoryDataset:
    def __init__(self, frame: pl.DataFrame | pl.LazyFrame, loader_mode: str | None = None) -> None:
        self._lazy_frame = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        self._loader_mode = loader_mode
        self._frame_cache: pl.DataFrame | None = None
        self._trajectory_ids_cache: list[str] | None = None
        self._trajectory_cache: dict[str, Trajectory] = {}
        self._summary_cache: dict[str, object] | None = None
        self._feature_cache: pl.DataFrame | None = None
        self._cluster_result: ClusterResult | None = None

    @classmethod
    def load(
        cls,
        path: PathLike | object,
        include_states: bool = True,
        include_actions: bool = True,
        include_images: Literal["none", "paths", "bytes"] = "paths",
    ) -> "TrajectoryDataset":
        if _looks_like_lerobot_dataset(path):
            return cls.from_lerobot(
                dataset=path,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            )
        return cls(
            load_dataset_lazy(
                path=path,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            ),
            loader_mode="filesystem_lazy",
        )

    @classmethod
    def from_lerobot(
        cls,
        dataset: object,
        include_states: bool = True,
        include_actions: bool = True,
        include_images: Literal["none", "paths", "bytes"] = "paths",
    ) -> "TrajectoryDataset":
        return cls(
            load_lerobot_dataset(
                dataset=dataset,
                include_states=include_states,
                include_actions=include_actions,
                include_images=include_images,
            ),
            loader_mode=detect_lerobot_loader_mode(dataset),
        )

    @property
    def frame(self) -> pl.DataFrame:
        if self._frame_cache is None:
            self._frame_cache = self._lazy_frame.collect()
        return self._frame_cache

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        return self._lazy_frame

    def __len__(self) -> int:
        out = self._lazy_frame.select(pl.col("trajectory_id").n_unique().alias("n")).collect()
        return int(out["n"][0])

    def __getitem__(self, idx: int) -> Trajectory:
        ids = self.trajectory_ids()
        traj_id = ids[idx]
        return self.get_trajectory(str(traj_id))

    def trajectory_ids(self, refresh: bool = False) -> list[str]:
        if self._trajectory_ids_cache is None or refresh:
            self._trajectory_ids_cache = (
                self._lazy_frame.select(pl.col("trajectory_id").unique().sort().alias("trajectory_id"))
                .collect()["trajectory_id"]
                .to_list()
            )
        return self._trajectory_ids_cache

    def get_trajectory(self, trajectory_id: str) -> Trajectory:
        if trajectory_id in self._trajectory_cache:
            return self._trajectory_cache[trajectory_id]
        cols = self._lazy_frame.collect_schema().names()
        part = (
            self._lazy_frame.filter(pl.col("trajectory_id") == trajectory_id)
            .select(cols)
            .collect()
        )
        if part.is_empty():
            raise KeyError(f"trajectory_id {trajectory_id!r} not found")
        traj = Trajectory(trajectory_id=trajectory_id, frame=part)
        self._trajectory_cache[trajectory_id] = traj
        return traj

    def get_frame(
        self,
        trajectory_id: str,
        *,
        idx: int | None = None,
        frame_id: int | None = None,
        t: float | None = None,
    ) -> FrameRecord:
        return self.get_trajectory(trajectory_id).get_frame(idx=idx, frame_id=frame_id, t=t)

    def get_frame_view(
        self,
        trajectory_id: str,
        *,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        indices: list[int] | None = None,
        frame_ids: list[int] | None = None,
    ) -> FrameBatchView:
        return self.get_trajectory(trajectory_id).frame_view(
            start=start,
            stop=stop,
            step=step,
            indices=indices,
            frame_ids=frame_ids,
        )

    def get_image_channels(
        self,
        trajectory_id: str,
        *,
        kind: Literal["path", "bytes"] = "path",
        keep_null: bool = False,
    ) -> dict[str, dict[str, list[Any]]]:
        return self.get_trajectory(trajectory_id).image_channels(kind=kind, keep_null=keep_null)

    def get_image_ref(
        self,
        trajectory_id: str,
        *,
        idx: int | None = None,
        frame_id: int | None = None,
        t: float | None = None,
        camera: int = 0,
        kind: Literal["auto", "path", "bytes"] = "auto",
        modality: Literal["rgb", "depth", "other", "any"] = "any",
    ) -> Any:
        return self.get_trajectory(trajectory_id).get_image_ref(
            idx=idx,
            frame_id=frame_id,
            t=t,
            camera=camera,
            kind=kind,
            modality=modality,
        )

    def get_image(
        self,
        trajectory_id: str,
        *,
        idx: int | None = None,
        frame_id: int | None = None,
        t: float | None = None,
        camera: int = 0,
        modality: Literal["rgb", "depth", "other", "any"] = "any",
        source: Literal["auto", "bytes", "path"] = "auto",
        decode: bool = True,
    ) -> Any:
        return self.get_trajectory(trajectory_id).get_image(
            idx=idx,
            frame_id=frame_id,
            t=t,
            camera=camera,
            modality=modality,
            source=source,
            decode=decode,
        )

    def summary(self, refresh: bool = False) -> dict[str, object]:
        if self._summary_cache is None or refresh:
            self._summary_cache = summarize_dataset(self._lazy_frame)
            if self._loader_mode is not None:
                self._summary_cache["loader_mode"] = self._loader_mode
        return self._summary_cache

    def feature_table(self, refresh: bool = False) -> pl.DataFrame:
        if self._feature_cache is None or refresh:
            self._feature_cache = per_trajectory_features(self._lazy_frame)
        return self._feature_cache

    def cluster(self, k: int = 4, seed: int = 0) -> ClusterResult:
        self._cluster_result = cluster_trajectories(self.feature_table(), k=k, seed=seed)
        return self._cluster_result

    def find_outliers(self, top_k: int = 10) -> list[dict[str, float | str]]:
        return find_outliers_impl(self.feature_table(), top_k=top_k)

    def compare(self, other: "TrajectoryDataset") -> dict[str, object]:
        return compare_frames(self._lazy_frame, other.lazy_frame)

    def plot_coverage(self, bins: int = 120, ax=None):
        import matplotlib.pyplot as plt

        use_ax = ax if ax is not None else plt.subplots(figsize=(7, 5))[1]
        xy = self._lazy_frame.select(["x", "y"]).collect()
        x = xy["x"].to_numpy()
        y = xy["y"].to_numpy()
        hb = use_ax.hexbin(x, y, gridsize=bins, cmap="viridis", mincnt=1)
        use_ax.set_title("Spatial coverage")
        use_ax.set_xlabel("x")
        use_ax.set_ylabel("y")
        plt.colorbar(hb, ax=use_ax, label="density")
        return use_ax

    def plot_prototypes(self, ax=None):
        import matplotlib.pyplot as plt

        if self._cluster_result is None:
            self.cluster()
        assert self._cluster_result is not None

        use_ax = ax if ax is not None else plt.subplots(figsize=(7, 5))[1]
        for cid, traj_id in sorted(self._cluster_result.medoid_ids.items()):
            traj = self.get_trajectory(traj_id)
            part = traj.frame
            use_ax.plot(part["x"].to_numpy(), part["y"].to_numpy(), linewidth=1.3, label=f"cluster {cid}")
        use_ax.set_title("Prototype trajectories")
        use_ax.set_xlabel("x")
        use_ax.set_ylabel("y")
        use_ax.legend()
        return use_ax

    def plot_start_end_distribution(self, ax=None):
        frame = self._lazy_frame.select(["trajectory_id", "t", "x", "y"]).collect()
        return plot_start_end_distribution(frame, ax=ax)

    def plot_feature_distributions(self):
        return plot_feature_distributions(self.feature_table())

    def plot_outlier_ranking(self, top_k: int = 20, ax=None):
        return plot_outlier_ranking(self.find_outliers(top_k=top_k), ax=ax)

    def plot_cluster_embedding(self, k: int = 4, seed: int = 0, ax=None):
        if self._cluster_result is None:
            self.cluster(k=k, seed=seed)
        assert self._cluster_result is not None
        return plot_cluster_embedding(self.feature_table(), self._cluster_result.labels, ax=ax)

    def plot_modality_coverage(self, ax=None):
        schema_names = self._lazy_frame.collect_schema().names()
        image_cols = [c for c in schema_names if c.startswith("image_") and (c.endswith("_path") or c.endswith("_bytes"))]
        keep = [c for c in ("state_vec", "action_vec") if c in schema_names] + sorted(image_cols)
        frame = self._lazy_frame.select(keep).collect()
        return plot_modality_coverage(frame, ax=ax)


def _looks_like_lerobot_dataset(obj: object) -> bool:
    if isinstance(obj, (str, bytes, Path)) or hasattr(obj, "__fspath__"):
        return False
    mod = type(obj).__module__.lower()
    name = type(obj).__name__.lower()
    if "lerobot" in mod or "lerobot" in name:
        return True
    # Generic dataset-like fallback if it also exposes a LeRobot-style root path.
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and hasattr(obj, "root")
