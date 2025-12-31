"""
评估指标
"""

import torch
import numpy as np
from typing import Tuple


def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    计算RMSE（均方根误差）

    Args:
        pred: 预测DOA (batch, k)
        target: 真实DOA (batch, k)
        mask: 有效样本掩码 (batch,)

    Returns:
        RMSE值（度）
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    if pred.numel() == 0:
        return float('nan')

    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)

    return rmse.item()


def compute_success_rate(success: torch.Tensor) -> float:
    """
    计算成功率

    Args:
        success: 成功标记 (batch,)

    Returns:
        成功率
    """
    return success.float().mean().item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """计算MAE（平均绝对误差）"""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    if pred.numel() == 0:
        return float('nan')

    mae = torch.mean(torch.abs(pred - target))
    return mae.item()


def compute_resolution_probability(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    计算分辨概率

    如果所有预测角度与真实角度的误差都小于阈值，则认为成功分辨

    Args:
        pred: 预测DOA (batch, k)，已排序
        target: 真实DOA (batch, k)，已排序
        threshold: 误差阈值（度）

    Returns:
        分辨成功率
    """
    errors = torch.abs(pred - target)
    max_errors = errors.max(dim=-1).values
    success = max_errors < threshold

    return success.float().mean().item()


class MetricTracker:
    """指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rmse_sum = 0.0
        self.mae_sum = 0.0
        self.success_count = 0
        self.total_count = 0
        self.resolution_count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        success: torch.Tensor = None,
        threshold: float = 2.0
    ):
        batch_size = pred.shape[0]

        if success is not None:
            mask = success
            self.success_count += success.sum().item()
        else:
            mask = torch.ones(batch_size, dtype=torch.bool, device=pred.device)

        self.total_count += batch_size

        # RMSE
        if mask.any():
            pred_valid = pred[mask]
            target_valid = target[mask]
            self.rmse_sum += ((pred_valid - target_valid) ** 2).sum().item()
            self.mae_sum += torch.abs(pred_valid - target_valid).sum().item()

            # 分辨率
            errors = torch.abs(pred_valid - target_valid)
            max_errors = errors.max(dim=-1).values
            self.resolution_count += (max_errors < threshold).sum().item()

    def compute(self) -> dict:
        if self.total_count == 0:
            return {'rmse': float('nan'), 'mae': float('nan'), 'success_rate': 0.0, 'resolution_prob': 0.0}

        valid_count = self.success_count if self.success_count > 0 else self.total_count

        return {
            'rmse': np.sqrt(self.rmse_sum / (valid_count * 3 + 1e-8)),  # 假设k=3
            'mae': self.mae_sum / (valid_count * 3 + 1e-8),
            'success_rate': self.success_count / self.total_count,
            'resolution_prob': self.resolution_count / self.total_count
        }
