"""Local smoke + speed check for trajkit."""

from __future__ import annotations

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from trajkit import TrajectoryDataset


def run() -> None:
    lerobot_dataset = LeRobotDataset("lerobot/libero")

    for image_mode in ("none", "paths"):
        t0 = time.perf_counter()
        ds = TrajectoryDataset.load(lerobot_dataset, include_images=image_mode)
        load_dt = time.perf_counter() - t0

        t1 = time.perf_counter()
        summary = ds.summary()
        summary_dt = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = ds[0].frame_view(start=0, stop=128, step=2)
        frame_view_dt = time.perf_counter() - t2

        print(
            {
                "image_mode": image_mode,
                "load_sec": round(load_dt, 3),
                "summary_sec": round(summary_dt, 3),
                "frame_view_sec": round(frame_view_dt, 3),
                "loader_mode": summary.get("loader_mode"),
                "has_images": summary.get("modalities", {}).get("has_images"),
                "num_trajectories": summary.get("num_trajectories"),
                "num_points": summary.get("num_points"),
            }
        )

    file_ds = TrajectoryDataset.load("data/hf_vla/libero/file-000.parquet")
    print({"file_summary": file_ds.summary()})


if __name__ == "__main__":
    run()