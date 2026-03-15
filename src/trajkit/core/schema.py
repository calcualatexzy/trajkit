"""Core dataclasses for analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClusterResult:
    labels: dict[str, int]
    centroids: list[list[float]]
    medoid_ids: dict[int, str]


@dataclass(slots=True)
class FrameRecord:
    trajectory_id: str
    frame_id: int
    t: float
    values: dict[str, Any]


@dataclass(slots=True)
class FrameBatchView:
    trajectory_id: str
    frame_id: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    extras: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.frame_id.shape[0])

    def column(self, name: str) -> np.ndarray:
        if name == "frame_id":
            return self.frame_id
        if name == "t":
            return self.t
        if name == "x":
            return self.x
        if name == "y":
            return self.y
        if name == "z":
            return self.z
        if name in self.extras:
            return self.extras[name]
        raise KeyError(f"column {name!r} not found")
