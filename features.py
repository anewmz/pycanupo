"""
Feature extraction utilities for multi-scale geometric descriptors.
"""

import numpy as np


def eigen_features(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Computes eigenvalue-based geometric features for one pointâ€™s neighborhood."""
    # TODO: Compute covariance and derive 6 descriptors (linearity, planarity, etc.)
    raise NotImplementedError


def multiscale_features(X: np.ndarray, radii: list[float], leaf_size: int = 40) -> np.ndarray:
    """Computes concatenated features across multiple scales."""
    # TODO: Use KDTree to find neighbors and aggregate features.
    raise NotImplementedError


def estimate_radii(X: np.ndarray, k: int = 16, n_levels: int = 4) -> list[float]:
    """Automatically estimates radii ladder from nearest-neighbor distances."""
    # TODO: Compute base scale and geometric progression.
    raise NotImplementedError
