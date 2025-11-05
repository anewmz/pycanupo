"""
Feature extraction utilities for multi-scale geometric descriptors.
"""

import numpy as np
from sklearn.neighbors import KDTree


def eigen_features(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Compute per-point eigenvalue-based features.
    idxs: list/array of neighbor indices for ONE point (1D array).
    Returns a small feature vector for that point at ONE scale."""
    if idxs.size < 3:
        # too few neighbors → return NaNs to be handled later
        return np.full(6, np.nan)

    P = X[idxs, :]               # k x 3
    C = np.cov(P.T, bias=True)   # 3x3 covariance (biased = divide by N)
    # Numerical guard
    if not np.all(np.isfinite(C)):
        return np.full(6, np.nan)

    w = np.linalg.eigvalsh(C)    # sorted ascending
    # Ensure strictly positive, small eps to avoid divide-by-zero
    w = np.clip(w, 1e-15, None)
    w.sort()                     # λ1 ≤ λ2 ≤ λ3
    lam1, lam2, lam3 = w[2], w[1], w[0]  # reorder to lam1≥lam2≥lam3

    sumlam = lam1 + lam2 + lam3
    # Standard geometric descriptors (scale-sensitive on purpose)
    linearity   = (lam1 - lam2) / lam1
    planarity   = (lam2 - lam3) / lam1
    sphericity  = lam3 / lam1
    anisotropy  = (lam1 - lam3) / lam1
    curvature   = lam3 / sumlam
    omnivariance = (lam1 * lam2 * lam3) ** (1/3)

    return np.array([linearity, planarity, sphericity, anisotropy, curvature, omnivariance], dtype=np.float64)


def multiscale_features(X: np.ndarray, radii: list[float], leaf_size: int = 40) -> np.ndarray:
    """Compute features for all points across all radii and concatenate."""
    tree = KDTree(X, leaf_size=leaf_size)
    feats = []

    for r in radii:
        # neighbors for each point within radius r
        # Returns list of arrays; variable length per point
        idx_lists = tree.query_radius(X, r=r, return_distance=False)
        F = np.empty((X.shape[0], 6), dtype=np.float64)
        for i, idxs in enumerate(idx_lists):
            F[i, :] = eigen_features(X, idxs)
        feats.append(F)

    # Concatenate over scales: [N x (6 * len(radii))]
    Z = np.concatenate(feats, axis=1)

    # NaN handling: replace NaNs with per-column nanmedian; fallback to zeros
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if np.any(~np.isfinite(col)):
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 0.0
            col[~np.isfinite(col)] = med
            Z[:, j] = col
    return Z


def estimate_radii(X: np.ndarray, k: int = 16, n_levels: int = 4) -> list[float]:
    """
    Auto-pick a radii ladder: base ~= median k-NN spacing, then geometric ladder.
    """
    tree = KDTree(X)
    d, _ = tree.query(X, k=k)
    base = float(np.median(d[:, -1]))
    return [base * (2 ** i) for i in range(n_levels)]
