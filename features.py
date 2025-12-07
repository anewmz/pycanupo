"""
Feature extraction utilities for multi-scale geometric descriptors.
"""

import numpy as np
from sklearn.neighbors import KDTree


def eigen_features(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Compute per-point eigenvalue-based features.
    idxs: list/array of neighbor indices for ONE point (1D array).
    Returns a feature vector for that point at ONE scale (9 features total)."""
    if idxs.size < 3:
        # too few neighbors → return NaNs to be handled later
        # we return 9 features per scale (existing 6 + 3 extras)
        return np.full(9, np.nan)

    P = X[idxs, :]               # k x 3
    C = np.cov(P.T, bias=True)   # 3x3 covariance (biased = divide by N)
    # Numerical guard
    if not np.all(np.isfinite(C)):
        return np.full(9, np.nan)

    # eigen decomposition for values + vectors
    w, v = np.linalg.eigh(C)    # w sorted ascending, v columns are eigenvectors
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

    # Additional per-scale descriptors (small but high-ROI):
    # verticality: absolute Z component of the principal eigenvector
    principal_vec = v[:, 2] if v.shape[1] >= 3 else v[:, -1]
    verticality = float(abs(principal_vec[2]))

    # neighbor count and mean distance-to-centroid (scale-aware)
    neighbor_count = float(idxs.size)
    centroid = P.mean(axis=0)
    mean_dist = float(np.mean(np.linalg.norm(P - centroid, axis=1)))

    return np.array([
        linearity, planarity, sphericity, anisotropy, curvature, omnivariance,
        verticality, neighbor_count, mean_dist
    ], dtype=np.float64)


def multiscale_features(X: np.ndarray, radii: list[float], leaf_size: int = 40) -> np.ndarray:
    """Compute features for all points across all radii and concatenate."""
    tree = KDTree(X, leaf_size=leaf_size)
    feats = []

    for r in radii:
        # neighbors for each point within radius r
        # Returns list of arrays; variable length per point
        idx_lists = tree.query_radius(X, r=r, return_distance=False)
        # adjust per-scale feature width to new eigen_features() output (9 features)
        F = np.empty((X.shape[0], 9), dtype=np.float64)
        for i, idxs in enumerate(idx_lists):
            F[i, :] = eigen_features(X, idxs)
        feats.append(F)

    # Concatenate over scales: [N x (9 * len(radii))]
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
