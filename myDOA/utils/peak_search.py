"""Shared peak-search helpers for grid-based DOA spectra."""

import math

import numpy as np


def min_sep_to_bins(min_sep: float, grid_step: float) -> int:
    """Convert an angular minimum separation into inclusive NMS bins."""
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if min_sep <= 0:
        return 0
    return int(math.ceil(float(min_sep) / float(grid_step)))


def select_peaks_with_suppression(
    spectrum: np.ndarray,
    k: int,
    min_sep_bins: int,
) -> np.ndarray:
    """Select top-k spectrum bins with an inclusive suppression window."""
    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 1:
        raise ValueError("spectrum must be one-dimensional")
    if spectrum.size == 0 and k > 0:
        raise ValueError("spectrum must not be empty when k > 0")
    if k < 0:
        raise ValueError("k must be non-negative")
    if min_sep_bins < 0:
        raise ValueError("min_sep_bins must be non-negative")

    spectrum_work = spectrum.astype(np.float64, copy=True)
    selected = []
    min_sep_bins = int(min_sep_bins)

    for _ in range(int(k)):
        peak_idx = int(np.argmax(spectrum_work))
        selected.append(peak_idx)

        left = max(0, peak_idx - min_sep_bins)
        right = min(spectrum_work.size, peak_idx + min_sep_bins + 1)
        spectrum_work[left:right] = -np.inf

    return np.asarray(sorted(selected), dtype=np.int64)


def refine_peak_parabolic(
    spectrum: np.ndarray,
    grid: np.ndarray,
    peak_idx: int,
) -> float:
    """Refine a peak angle with three-point parabolic interpolation."""
    spectrum = np.asarray(spectrum)
    grid = np.asarray(grid)
    if spectrum.ndim != 1 or grid.ndim != 1:
        raise ValueError("spectrum and grid must be one-dimensional")
    if spectrum.shape[0] != grid.shape[0]:
        raise ValueError("spectrum and grid must have the same length")

    peak_idx = int(peak_idx)
    if peak_idx <= 0 or peak_idx >= spectrum.shape[0] - 1:
        return float(grid[peak_idx])

    y_left = float(spectrum[peak_idx - 1])
    y_center = float(spectrum[peak_idx])
    y_right = float(spectrum[peak_idx + 1])
    denominator = 2.0 * (y_left - 2.0 * y_center + y_right)

    if abs(denominator) < 1e-10:
        return float(grid[peak_idx])

    delta = (y_left - y_right) / denominator
    delta = float(np.clip(delta, -0.5, 0.5))
    grid_step = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    return float(grid[peak_idx] + delta * grid_step)


def refine_peak_centroid(
    spectrum: np.ndarray,
    grid: np.ndarray,
    peak_idx: int,
    window: int = 2,
) -> float:
    """Refine a peak angle with a non-negative weighted centroid."""
    spectrum = np.asarray(spectrum)
    grid = np.asarray(grid)
    if spectrum.ndim != 1 or grid.ndim != 1:
        raise ValueError("spectrum and grid must be one-dimensional")
    if spectrum.shape[0] != grid.shape[0]:
        raise ValueError("spectrum and grid must have the same length")

    peak_idx = int(peak_idx)
    if peak_idx <= window or peak_idx >= spectrum.shape[0] - window - 1:
        return float(grid[peak_idx])

    left = peak_idx - window
    right = peak_idx + window + 1
    values = np.maximum(spectrum[left:right], 0.0)
    total = float(np.sum(values))
    if total < 1e-10:
        return float(grid[peak_idx])

    weights = values / total
    return float(np.sum(weights * grid[left:right]))
