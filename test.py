"""Minimal demo: load, analyze, show one image."""

import matplotlib.pyplot as plt

from trajkit import TrajectoryDataset


demo_path = "data/hf_vla/openx_bridge_v2_subset.parquet"
ds = TrajectoryDataset.load(demo_path, include_images="bytes")

summary = ds.summary()
print("num_trajectories:", summary.get("num_trajectories"))
print("num_points:", summary.get("num_points"))
print("feature rows:", ds.feature_table().height)
print("top outlier:", ds.find_outliers(top_k=1))

traj = ds[0]
f0 = traj.get_frame(idx=0)
print("frame0:", {"trajectory_id": f0.trajectory_id, "frame_id": f0.frame_id, "t": f0.t})

channels = traj.image_channels(kind="bytes")
print("image channel columns:", {k: list(v.keys()) for k, v in channels.items()})

try:
    img = traj.get_image(idx=0, modality="rgb", decode=True)  # source auto: bytes -> path fallback
except Exception:
    img = None

if img is None:
    print("no image bytes found")
else:
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title("First trajectory image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()