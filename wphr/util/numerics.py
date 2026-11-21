"""Numerical guards.

The cognitive core must never silently produce NaN/Inf. Every state transition
goes through :func:`check_finite` / :func:`sanitize_array`, and the caller counts
faults so that a long run can report them instead of crashing.
"""

from __future__ import annotations

import numpy as np


class NumericFault(RuntimeError):
    """Raised when a non-finite value reaches a guarded array."""


def check_finite(array: np.ndarray, where: str = "array") -> np.ndarray:
    """Return ``array`` unchanged, or raise :class:`NumericFault`."""
    arr = np.asarray(array)
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise NumericFault(f"non-finite values in {where}: {bad} of {arr.size} entries")
    return arr


def sanitize_array(
    array: np.ndarray, clip: float, where: str = "array"
) -> tuple[np.ndarray, int]:
    """Replace NaN by 0 and clamp +/-Inf, returning ``(clean, fault_count)``."""
    arr = np.array(array, dtype=np.float64, copy=True)
    finite = np.isfinite(arr)
    faults = int(arr.size - finite.sum())
    if faults:
        arr[~finite] = 0.0
    np.clip(arr, -clip, clip, out=arr)
    return arr, faults


def clip_array(array: np.ndarray, bound: float) -> np.ndarray:
    """Clip in place and return the same array (bound must be positive)."""
    if bound <= 0:
        raise ValueError("clip bound must be positive")
    np.clip(array, -bound, bound, out=array)
    return array


def spectral_radius(matrix: np.ndarray, power_iters: int = 400, tol: float = 1e-9) -> float:
    """Largest absolute eigenvalue magnitude of a square real matrix.

    Uses an exact eigendecomposition for modest sizes and power iteration for
    large matrices, so stability diagnostics stay cheap enough to run often.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"spectral_radius requires a square matrix, got {mat.shape}")
    n = mat.shape[0]
    if n == 0:
        return 0.0
    if n <= 512:
        values = np.linalg.eigvals(mat)
        return float(np.max(np.abs(values)))
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v) or 1.0
    rho = 0.0
    for _ in range(power_iters):
        nxt = mat @ v
        norm = float(np.linalg.norm(nxt))
        if norm == 0.0:
            return 0.0
        v = nxt / norm
        if abs(norm - rho) <= tol * max(1.0, rho):
            rho = norm
            break
        rho = norm
    return float(rho)
