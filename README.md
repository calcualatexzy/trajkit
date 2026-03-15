# trajkit

`trajkit` is a parquet-first Python toolkit for loading, analyzing, and visualizing robot trajectories.

It is designed for:
- fast first result on large datasets
- low-memory lazy workflows
- aligned frame-level access (`time/id + state + action + image`)

---

## 1) Install

```bash
python -m pip install -e .
```

---

## 2) Data and schema support

### Supported file formats
- `.parquet` (primary)
- `.csv`
- `.json` / `.jsonl`
- `.npy` / `.npz`

### Input targets
- single file path
- directory (recursive discovery)

### Canonical internal columns
- `trajectory_id`
- `frame_id`
- `t`
- `x`, `y`, `z`

### Optional modality columns
- `state_vec`
- `action_vec`
- `image_0_path`, `image_1_path`
- `image_0_bytes`, `image_1_bytes` (only when `include_images="bytes"`)

### VLA/LeRobot compatibility
`trajkit` auto-extracts common VLA layouts (including HuggingFaceVLA datasets such as `libero` and `community_dataset_v2`) into the canonical schema.

---

## 3) Load datasets

```python
from trajkit import TrajectoryDataset

# Default: include states/actions and image paths
ds = TrajectoryDataset.load("data/hf_vla/libero/file-000.parquet")
```

Loading options:

```python
ds = TrajectoryDataset.load(
    "data/",
    include_states=True,
    include_actions=True,
    include_images="paths",   # "none" | "paths" | "bytes"
)
```

LeRobotDataset object support:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from trajkit import TrajectoryDataset

lerobot_ds = LeRobotDataset(
    repo_id="local",
    root="./community_dataset_v2/contributor/dataset_name",
)

# Works directly
ds = TrajectoryDataset.load(lerobot_ds)

# Or explicit constructor
ds2 = TrajectoryDataset.from_lerobot(lerobot_ds, include_images="paths")
```

When loading LeRobot datasets, `summary()["loader_mode"]` reports which path was used:
- `lerobot_root_fast_path`
- `lerobot_root_fast_path_video_meta`
- `lerobot_iter_chunked`

Use `include_images="none"` when you want maximum throughput and minimum memory.

---

## 4) Core objects

### `TrajectoryDataset`
Main collection object.

Common methods:
- `len(ds)` -> number of trajectories
- `ds.summary()` -> global statistics + modality coverage + alignment warnings + `loader_mode`
- `ds.feature_table()` -> per-trajectory derived feature table
- `ds.cluster(k=4, seed=0)` -> cluster trajectories
- `ds.find_outliers(top_k=10)` -> robust outlier ranking
- `ds.compare(other_ds)` -> dataset comparison

Trajectory access:
- `traj = ds[0]`
- `traj = ds.get_trajectory("episode_001")`

Frame access through dataset:
- `ds.get_frame("episode_001", idx=10)`
- `ds.get_frame("episode_001", frame_id=10)`
- `ds.get_frame("episode_001", t=2.5)`
- `ds.get_frame_view("episode_001", start=0, stop=256, step=2)`

### `Trajectory`
Single aligned trajectory object.

Common methods:
- `traj.features()`
- `traj.alignment()`
- `traj.state_vectors()`
- `traj.action_vectors()`
- `traj.image_paths()`
- `traj.nearest_neighbors(ds, k=5)`

Single-frame retrieval:
- `traj.get_frame(idx=...)`
- `traj.get_frame(frame_id=...)`
- `traj.get_frame(t=...)` (nearest)

Batched frame retrieval:
- `traj.frame_view(start=..., stop=..., step=...)`
- `traj.frame_view(indices=[...])`
- `traj.frame_view(frame_ids=[...])`

---

## 5) Analysis and visualization

### Dataset-level visualization
- `ds.plot_coverage()` - spatial density / coverage
- `ds.plot_start_end_distribution()` - start/end location map
- `ds.plot_prototypes()` - medoid trajectories per cluster
- `ds.plot_feature_distributions()` - derived feature histograms
- `ds.plot_outlier_ranking(top_k=20)` - ranked outlier bars
- `ds.plot_cluster_embedding(k=4, seed=0)` - feature embedding colored by cluster
- `ds.plot_modality_coverage()` - state/action/image availability

### Trajectory-level visualization
- `traj.plot()` - 2D path with start/end markers
- `traj.plot_state_channels(max_channels=6)` - state traces over time
- `traj.plot_action_channels(max_channels=6)` - action traces over time

---

## 6) Fast frame slicing with `FrameBatchView`

`FrameBatchView` is optimized for batched operations:
- direct numpy arrays for `frame_id`, `t`, `x`, `y`, `z`
- extra modality arrays in `extras`
- column access with `batch.column("state_vec")`

Example:

```python
traj = ds[0]
batch = traj.frame_view(start=0, stop=128, step=2)

print(len(batch))
print(batch.x.shape, batch.t.shape)
print(batch.column("state_vec").shape)
```

---

## 7) End-to-end example

```python
from trajkit import TrajectoryDataset

ds = TrajectoryDataset.load("data/hf_vla/libero/file-000.parquet")
print(ds.summary())

# Analysis
clusters = ds.cluster(k=4, seed=0)
outliers = ds.find_outliers(top_k=10)
print(clusters.medoid_ids)
print(outliers[:3])

# Visualization
ds.plot_coverage()
ds.plot_feature_distributions()
ds.plot_cluster_embedding(k=4, seed=0)
ds.plot_outlier_ranking(top_k=15)

# Frame-aligned access
traj = ds[0]
f = traj.get_frame(t=2.0)
batch = traj.frame_view(frame_ids=[0, 1, 2, 3, 4])
```

---

## 8) Performance tips

- Prefer parquet inputs.
- Keep `include_images="none"` for heavy analysis workloads.
- Use `frame_view(...)` for batched frame computation instead of many single-frame calls.
- Avoid `ds.frame` unless you intentionally want full materialization.
