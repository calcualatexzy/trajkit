"""Single trajectory object."""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl

from trajkit.analysis.similarity import nearest_neighbors
from trajkit.core.schema import FrameBatchView, FrameRecord
from trajkit.preprocess.features import trajectory_features


class Trajectory:
    def __init__(self, trajectory_id: str, frame: pl.DataFrame) -> None:
        self.id = str(trajectory_id)
        if "frame_id" in frame.columns:
            self._frame = frame.sort("frame_id")
        else:
            self._frame = frame.sort("t").with_row_index(name="frame_id")
        self._frame_id_to_row = {
            int(fid): idx for idx, fid in enumerate(self._frame["frame_id"].to_list())
        }
        self._times = self._frame["t"].to_list()
        self._core_arrays: dict[str, np.ndarray] | None = None
        self._extra_arrays: dict[str, np.ndarray] | None = None

    @property
    def frame(self) -> pl.DataFrame:
        return self._frame

    def state_vectors(self):
        if "state_vec" not in self._frame.columns:
            return None
        return self._frame["state_vec"]

    def action_vectors(self):
        if "action_vec" not in self._frame.columns:
            return None
        return self._frame["action_vec"]

    def image_paths(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for col in ("image_0_path", "image_1_path"):
            if col in self._frame.columns:
                out[col] = [v for v in self._frame[col].to_list() if v is not None]
        return out

    def alignment(self) -> dict[str, object]:
        out: dict[str, object] = {
            "num_frames": int(self._frame.height),
            "has_state_vec": "state_vec" in self._frame.columns,
            "has_action_vec": "action_vec" in self._frame.columns,
            "has_image_0_path": "image_0_path" in self._frame.columns,
            "has_image_1_path": "image_1_path" in self._frame.columns,
            "missing_timestamps": int(self._frame["t"].null_count()),
            "duplicate_frame_ids": int(self._frame.height - self._frame["frame_id"].n_unique()),
            "is_monotonic_t": bool(np.all(np.diff(np.array(self._times, dtype=float)) >= 0.0)) if len(self._times) > 1 else True,
        }
        if "state_vec" in self._frame.columns:
            out["missing_state_rows"] = int(self._frame["state_vec"].null_count())
        if "action_vec" in self._frame.columns:
            out["missing_action_rows"] = int(self._frame["action_vec"].null_count())
        for col in ("image_0_path", "image_1_path"):
            if col in self._frame.columns:
                out[f"missing_{col}_rows"] = int(self._frame[col].null_count())
        return out

    def get_frame(
        self,
        *,
        idx: int | None = None,
        frame_id: int | None = None,
        t: float | None = None,
    ) -> FrameRecord:
        provided = [idx is not None, frame_id is not None, t is not None]
        if sum(provided) != 1:
            raise ValueError("provide exactly one of idx, frame_id, or t")
        if idx is not None:
            if idx < 0 or idx >= self._frame.height:
                raise IndexError(f"idx {idx} out of range for trajectory {self.id}")
            row_idx = idx
        elif frame_id is not None:
            if frame_id not in self._frame_id_to_row:
                raise KeyError(f"frame_id {frame_id} not found in trajectory {self.id}")
            row_idx = self._frame_id_to_row[frame_id]
        else:
            assert t is not None
            row_idx = bisect.bisect_left(self._times, float(t))
            if row_idx >= len(self._times):
                row_idx = len(self._times) - 1
            elif row_idx > 0:
                left = abs(self._times[row_idx - 1] - float(t))
                right = abs(self._times[row_idx] - float(t))
                if left <= right:
                    row_idx = row_idx - 1

        row = self._frame.row(row_idx, named=True)
        return FrameRecord(
            trajectory_id=str(row["trajectory_id"]),
            frame_id=int(row["frame_id"]),
            t=float(row["t"]),
            values=row,
        )

    def frame_view(
        self,
        *,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
        indices: list[int] | np.ndarray | None = None,
        frame_ids: list[int] | np.ndarray | None = None,
    ) -> FrameBatchView:
        if step <= 0:
            raise ValueError("step must be > 0")
        chosen = [indices is not None, frame_ids is not None]
        if sum(chosen) > 1:
            raise ValueError("use only one selector: indices or frame_ids")

        if indices is not None:
            selector = np.asarray(indices, dtype=np.int64)
        elif frame_ids is not None:
            fid = np.asarray(frame_ids, dtype=np.int64)
            selector = np.asarray([self._frame_id_to_row[int(v)] for v in fid], dtype=np.int64)
        else:
            selector = slice(start, stop, step)
        return self._build_frame_batch(selector)

    def _build_frame_batch(self, selector: slice | np.ndarray) -> FrameBatchView:
        core = {name: arr[selector] for name, arr in self._get_core_arrays().items()}
        extras = {name: arr[selector] for name, arr in self._get_extra_arrays().items()}
        return FrameBatchView(
            trajectory_id=self.id,
            frame_id=core["frame_id"],
            t=core["t"],
            x=core["x"],
            y=core["y"],
            z=core["z"],
            extras=extras,
        )

    def _get_core_arrays(self) -> dict[str, np.ndarray]:
        if self._core_arrays is None:
            self._core_arrays = {
                "frame_id": self._frame["frame_id"].to_numpy(),
                "t": self._frame["t"].to_numpy(),
                "x": self._frame["x"].to_numpy(),
                "y": self._frame["y"].to_numpy(),
                "z": self._frame["z"].to_numpy(),
            }
        return self._core_arrays

    def _get_extra_arrays(self) -> dict[str, np.ndarray]:
        if self._extra_arrays is None:
            core_cols = {"frame_id", "t", "x", "y", "z", "trajectory_id"}
            self._extra_arrays = {col: self._frame[col].to_numpy() for col in self._frame.columns if col not in core_cols}
        return self._extra_arrays

    def features(self) -> dict[str, float]:
        return trajectory_features(self._frame)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        use_ax = ax if ax is not None else plt.subplots(figsize=(6, 4))[1]
        x = self._frame["x"].to_numpy()
        y = self._frame["y"].to_numpy()
        use_ax.plot(x, y, linewidth=1.5)
        use_ax.scatter([x[0]], [y[0]], marker="o", s=20, label="start")
        use_ax.scatter([x[-1]], [y[-1]], marker="x", s=30, label="end")
        use_ax.set_title(f"Trajectory {self.id}")
        use_ax.set_xlabel("x")
        use_ax.set_ylabel("y")
        use_ax.legend()
        return use_ax

    def plot_state_channels(self, max_channels: int = 6):
        import matplotlib.pyplot as plt

        if "state_vec" not in self._frame.columns:
            raise ValueError("state_vec is not available for this trajectory")
        state = self._frame["state_vec"].to_list()
        if not state:
            raise ValueError("empty trajectory")
        arr = np.asarray(state, dtype=float)
        n = min(max_channels, arr.shape[1])
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 1.8 * n), sharex=True)
        axes_1d = np.array(axes).reshape(-1)
        t = self._frame["t"].to_numpy()
        for i in range(n):
            axes_1d[i].plot(t, arr[:, i], linewidth=1.1)
            axes_1d[i].set_ylabel(f"s[{i}]")
        axes_1d[-1].set_xlabel("t")
        fig.suptitle(f"Trajectory {self.id} state channels")
        fig.tight_layout()
        return fig

    def plot_action_channels(self, max_channels: int = 6):
        import matplotlib.pyplot as plt

        if "action_vec" not in self._frame.columns:
            raise ValueError("action_vec is not available for this trajectory")
        action = self._frame["action_vec"].to_list()
        if not action:
            raise ValueError("empty trajectory")
        arr = np.asarray(action, dtype=float)
        n = min(max_channels, arr.shape[1])
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 1.8 * n), sharex=True)
        axes_1d = np.array(axes).reshape(-1)
        t = self._frame["t"].to_numpy()
        for i in range(n):
            axes_1d[i].plot(t, arr[:, i], linewidth=1.1)
            axes_1d[i].set_ylabel(f"a[{i}]")
        axes_1d[-1].set_xlabel("t")
        fig.suptitle(f"Trajectory {self.id} action channels")
        fig.tight_layout()
        return fig

    def nearest_neighbors(self, dataset: "TrajectoryDataset", k: int = 5):
        return nearest_neighbors(source_id=self.id, frame=dataset.feature_table(), k=k)
