"""
I/O utilities for reading, writing, and manipulating point clouds with CloudComPy.
"""

import numpy as np
import cloudComPy as cc
from pathlib import Path
from sklearn.neighbors import KDTree


def read_cloud(path: str):
    """Loads a point cloud from disk using CloudComPy."""
    ent = cc.loadPointCloud(path)
    if ent is None:
        raise RuntimeError(f"Failed to load cloud: {path}")
    return ent


def cloud_to_numpy(cloud) -> np.ndarray:
    """Extract Nx3 numpy array from a ccPointCloud (robust but simple).
    NOTE: This is O(N); for huge clouds consider downsampling or batching."""
    n = cloud.size()
    arr = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        P = cloud.getPoint(i)  # returns (x, y, z)
        arr[i, 0], arr[i, 1], arr[i, 2] = P[0], P[1], P[2]
    return arr


def add_or_replace_scalar_field(cloud, name: str, values: np.ndarray) -> int:
    """Adds or updates a scalar field on a cloud with the provided values."""
    if values.ndim != 1 or values.shape[0] != cloud.size():
        raise ValueError("values must be 1D and match cloud size")

    sf_dic = cloud.getScalarFieldDic()
    if name in sf_dic:
        idx = sf_dic[name]
        sf = cloud.getScalarField(idx)
        if sf.size() != cloud.size():
            cloud.deleteScalarField(idx)
            idx = cloud.addScalarField(name)            # should return int
            if not isinstance(idx, int) or idx < 0:
                raise RuntimeError("addScalarField failed")
            sf = cloud.getScalarField(idx)
    else:
        idx = cloud.addScalarField(name)
        if not isinstance(idx, int) or idx < 0:
            raise RuntimeError("addScalarField failed")
        sf = cloud.getScalarField(idx)

    for i, v in enumerate(values):
        sf.setValue(i, float(v))
    sf.computeMinAndMax()
    return idx


def get_scalar_array(cloud, name: str) -> np.ndarray:
    """Extracts a scalar field as a numpy array."""
    sf_dic = cloud.getScalarFieldDic()
    if name not in sf_dic:
        raise RuntimeError(f"Scalar field '{name}' not found")
    idx = sf_dic[name]
    sf = cloud.getScalarField(idx)
    n = sf.size()
    arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        arr[i] = sf.getValue(i)
    return arr


def ensure_parent_dir(path_str: str) -> None:
    """Create the parent folder for a file path if it doesn't exist."""
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def majority_smooth_labels(X: np.ndarray, labels: np.ndarray, radius: float | None = None, k: int | None = None, min_neighbors: int = 3) -> np.ndarray:
    """Return labels smoothed by neighborhood majority vote.

    - X: (N x 3) point coordinates
    - labels: (N,) int labels (use -1 for rejected points)
    - radius: use radius neighbors if provided; else use k nearest neighbors
    - k: if provided use k-NN (ignores point itself when k>1)
    - min_neighbors: don't change labels if neighborhood too small
    """
    N = X.shape[0]
    if radius is None and k is None:
        k = 8

    tree = KDTree(X)
    out = labels.copy()

    if radius is not None:
        idxs = tree.query_radius(X, r=radius, return_distance=False)
        for i in range(N):
            neigh = idxs[i]
            # exclude the point itself if present
            neigh = neigh[neigh != i]
            if neigh.size < min_neighbors:
                continue
            vals = labels[neigh]
            # ignore rejected neighbors
            vals = vals[vals != -1]
            if vals.size == 0:
                continue
            # majority vote
            vals_uni, counts = np.unique(vals, return_counts=True)
            out[i] = vals_uni[np.argmax(counts)]
    else:
        # k-NN (including the point itself by default) -> use k+1 to ensure excluding itself
        kk = k + 1 if k is not None and k > 0 else 9
        d, idxs = tree.query(X, k=kk)
        for i in range(N):
            neigh = idxs[i]
            # exclude itself (first entry normally)
            neigh = neigh[neigh != i]
            if neigh.size < min_neighbors:
                continue
            vals = labels[neigh]
            vals = vals[vals != -1]
            if vals.size == 0:
                continue
            vals_uni, counts = np.unique(vals, return_counts=True)
            out[i] = vals_uni[np.argmax(counts)]

    return out
