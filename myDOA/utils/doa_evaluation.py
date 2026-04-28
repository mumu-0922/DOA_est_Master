"""
Unified DOA evaluation utilities.

All multi-source DOA metrics in thesis experiments should use optimal
assignment before computing errors.
"""

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def _as_2d_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    return array


def evaluate_doa_sample(doa_est, doa_true, tol: float = 2.0) -> dict:
    """Evaluate one multi-source DOA estimate with Hungarian matching."""
    est = np.asarray(doa_est, dtype=np.float64).reshape(-1)
    true = np.asarray(doa_true, dtype=np.float64).reshape(-1)

    if est.shape != true.shape or est.size == 0 or np.any(~np.isfinite(est)):
        return {
            "rmse": float("nan"),
            "mse": float("nan"),
            "success": False,
            "matched_errors": np.full(true.shape, np.nan, dtype=np.float64),
        }

    cost_matrix = (est[:, None] - true[None, :]) ** 2
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    squared_errors = cost_matrix[row_ind, col_ind]
    matched_errors = np.sqrt(squared_errors)

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))

    return {
        "rmse": rmse,
        "mse": mse,
        "success": bool(np.all(matched_errors < tol)),
        "matched_errors": matched_errors,
    }


def evaluate_doa_batch(
    doa_est,
    doa_true,
    tol: float = 2.0,
    peak_success: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate a batch of DOA estimates with a shared success definition."""
    est = _as_2d_array(doa_est)
    true = _as_2d_array(doa_true)
    if est.shape != true.shape:
        raise ValueError(f"doa_est shape {est.shape} does not match doa_true shape {true.shape}")

    num_total = true.shape[0]
    if peak_success is None:
        peak_success = np.ones(num_total, dtype=bool)
    else:
        peak_success = np.asarray(peak_success, dtype=bool).reshape(-1)
        if peak_success.shape[0] != num_total:
            raise ValueError("peak_success length must match batch size")

    sample_rmse = np.full(num_total, np.nan, dtype=np.float64)
    sample_success = np.zeros(num_total, dtype=bool)
    mse_values = []

    for i in range(num_total):
        if not peak_success[i]:
            continue

        result = evaluate_doa_sample(est[i], true[i], tol=tol)
        sample_rmse[i] = result["rmse"]
        sample_success[i] = result["success"]
        if np.isfinite(result["mse"]):
            mse_values.append(result["mse"])

    rmse = float(np.sqrt(np.mean(mse_values))) if mse_values else float("nan")

    return {
        "rmse": rmse,
        "success_rate": float(np.mean(sample_success)) if num_total else 0.0,
        "num_success": int(np.sum(sample_success)),
        "num_valid": int(len(mse_values)),
        "num_total": int(num_total),
        "sample_rmse": sample_rmse,
        "sample_success": sample_success,
    }
