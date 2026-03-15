"""Dataset comparison helpers."""

from __future__ import annotations

from typing import Any

from trajkit.analysis.summary import summarize_dataset


def compare_frames(left_frame: Any, right_frame: Any) -> dict[str, object]:
    left = summarize_dataset(left_frame)
    right = summarize_dataset(right_frame)
    return {
        "left": left,
        "right": right,
        "delta_num_trajectories": int(left.get("num_trajectories", 0)) - int(right.get("num_trajectories", 0)),
        "delta_num_points": int(left.get("num_points", 0)) - int(right.get("num_points", 0)),
    }
