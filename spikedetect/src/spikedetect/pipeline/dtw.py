"""Dynamic Time Warping with squared-Euclidean local cost.

Ports the MATLAB function ``dtw_WarpingDistance.m`` (Parvez Ahammad, 2009).
A custom implementation is used because standard DTW libraries typically
default to absolute-difference local cost, whereas the spike-detection
pipeline requires squared-Euclidean cost to match the original MATLAB
behaviour.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except (ImportError, AttributeError):  # pragma: no cover
    _HAS_NUMBA = False

    def _njit(func=None, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is not installed."""
        if func is not None:
            return func
        return lambda f: f


@_njit(cache=True)
def _dtw_cost_matrix(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the accumulated cost matrix (inner loop, optionally JIT)."""
    M = r.shape[0]
    N = t.shape[0]
    D = np.empty((M, N), dtype=np.float64)
    D[0, 0] = (r[0] - t[0]) ** 2
    for m in range(1, M):
        D[m, 0] = (r[m] - t[0]) ** 2 + D[m - 1, 0]
    for n in range(1, N):
        D[0, n] = (r[0] - t[n]) ** 2 + D[0, n - 1]
    for m in range(1, M):
        for n in range(1, N):
            cost = (r[m] - t[n]) ** 2
            D[m, n] = cost + min(D[m - 1, n], min(D[m - 1, n - 1], D[m, n - 1]))
    return D


class DTW:
    """Dynamic Time Warping with squared-Euclidean local cost.

    Ports MATLAB ``dtw_WarpingDistance.m``. Uses a custom implementation
    because standard DTW libraries use the wrong cost metric.

    Examples
    --------
    >>> dist, rw, tw = DTW.warping_distance(signal_a, signal_b)
    """

    @staticmethod
    def warping_distance(
        r: np.ndarray,
        t: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute DTW distance with squared-Euclidean local cost.

        Parameters
        ----------
        r : np.ndarray
            Reference signal (1-D).
        t : np.ndarray
            Test signal (1-D).

        Returns
        -------
        distance : float
            Accumulated DTW distance (unnormalised).
        warped_r : np.ndarray
            Reference signal values along the optimal warping path.
        warped_t : np.ndarray
            Test signal values along the optimal warping path.

        Notes
        -----
        Original MATLAB function: ``dtw_WarpingDistance.m``
        """
        r = np.asarray(r, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()

        M = r.shape[0]
        N = t.shape[0]

        D = _dtw_cost_matrix(r, t)
        distance = float(D[M - 1, N - 1])

        # Backtrace for optimal warping path
        m = M - 1
        n = N - 1
        path = [(m, n)]

        while m > 0 or n > 0:
            if n == 0:
                m -= 1
            elif m == 0:
                n -= 1
            else:
                candidates = (D[m - 1, n], D[m, n - 1], D[m - 1, n - 1])
                argmin = int(np.argmin(candidates))
                if argmin == 0:
                    m -= 1
                elif argmin == 1:
                    n -= 1
                else:
                    m -= 1
                    n -= 1
            path.append((m, n))

        path.reverse()
        path_arr = np.array(path, dtype=np.intp)

        warped_r = r[path_arr[:, 0]]
        warped_t = t[path_arr[:, 1]]

        return distance, warped_r, warped_t

    @staticmethod
    def cost_matrix(r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute the accumulated cost matrix.

        Parameters
        ----------
        r, t : np.ndarray
            1-D signals.

        Returns
        -------
        np.ndarray
            Accumulated cost matrix of shape ``(len(r), len(t))``.
        """
        return _dtw_cost_matrix(
            np.asarray(r, dtype=np.float64).ravel(),
            np.asarray(t, dtype=np.float64).ravel(),
        )


# Backwards-compatible alias
dtw_warping_distance = DTW.warping_distance
