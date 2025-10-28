"""
I/O utilities for reading, writing, and manipulating point clouds with CloudComPy.
"""

import numpy as np
import cloudComPy as cc
from pathlib import Path


def read_cloud(path: str):
    """Loads a point cloud from disk using CloudComPy."""
    # TODO: Implement loading and error handling.
    raise NotImplementedError


def cloud_to_numpy(cloud) -> np.ndarray:
    """Converts a ccPointCloud to an Nx3 numpy array."""
    # TODO: Implement extraction of (x, y, z) coordinates.
    raise NotImplementedError


def add_or_replace_scalar_field(cloud, name: str, values: np.ndarray) -> int:
    """Adds or updates a scalar field on a cloud with the provided values."""
    # TODO: Implement scalar field management logic.
    raise NotImplementedError


def ensure_parent_dir(path_str: str) -> None:
    """Ensures the parent directory of the given path exists."""
    # TODO: Create directories as needed.
    raise NotImplementedError
