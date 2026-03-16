# trajkit tutorial

`trajkit` is a small toolkit for loading, analyzing, and visualizing robot trajectory datasets.

## 1) Install

```bash
python -m pip install -e .
```

## 2) Run the demo test

```bash
python test.py
```

This prints:
- dataset summary
- feature table size
- top outlier
- one trajectory/frame example
- image channel keys
- and pops up one decoded image with `plt.show()`

## 3) Load your own dataset

```python
from trajkit import TrajectoryDataset

ds = TrajectoryDataset.load("data/your_dataset", include_images="paths")
print(ds.summary())
```

`include_images`:
- `"none"`: fastest
- `"paths"`: load image paths
- `"bytes"`: load image bytes

## 4) Basic analysis

```python
print(ds.feature_table().head())
print(ds.cluster(k=4).medoid_ids)
print(ds.find_outliers(top_k=5))
```

## 5) Frame-level access

```python
traj = ds[0]
print(traj.get_frame(idx=0))
print(traj.frame_view(start=0, stop=64, step=2).x.shape)
print(traj.image_channels(kind="path").keys())  # rgb/depth/other
img = traj.get_image(idx=0, modality="rgb", decode=True)
# img is a numpy array, ready for plt.imshow(img)
```

`source` supports:
- `"auto"` (default): try bytes first, then path
- `"bytes"`: only bytes columns
- `"path"`: only path columns

## 6) Plot

```python
ax = ds.plot_coverage()  # matplotlib axis
ax.figure.show()
```

## Supported formats

- `.parquet` / `.csv` / `.json` / `.jsonl`
- `.npy` / `.npz`
- `.hdf5` / `.h5`
- `.rlds`
- `LeRobotDataset`
