"""
I/O utilities for reading, writing, and manipulating point clouds with CloudComPy.
"""

import numpy as np
import cloudComPy as cc
from pathlib import Path


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
