"""
基础网络类：Grid-Based DOA 估计网络基类
提供空间谱到DOA角度的转换功能
"""

import torch
import torch.nn as nn
import math


class GridBasedNetwork(nn.Module):
    """
    基于网格的DOA估计网络基类

    功能：
    - 管理角度搜索网格
    - 从空间谱中提取DOA角度（峰值搜索）
    """

    def __init__(self, start_angle=-60, end_angle=60, step=1, peak_threshold=0.0):
        super().__init__()
        self.register_buffer('grid', torch.arange(start_angle, end_angle + 0.0001, step))
        self.out_dim = self.grid.shape[0]
        self.peak_threshold = peak_threshold

    def spectrum_to_doa(self, spectrum: torch.Tensor, k: int, min_sep: float = 0.0):
        """
        从空间谱中提取k个DOA角度

        Args:
            spectrum: 空间谱 (batch, grid_size)
            k: 信源数量
            min_sep: 最小角度间隔（度）

        Returns:
            success: 是否成功找到k个峰 (batch,)
            theta: DOA角度估计 (batch, k)
        """
        B = spectrum.shape[0]
        device = spectrum.device

        if min_sep > 0:
            # NMS方式提取峰值
            return self._nms_peak_search(spectrum, k, min_sep)
        else:
            # 简单峰值搜索
            return self._simple_peak_search(spectrum, k)

    def _simple_peak_search(self, spectrum: torch.Tensor, k: int):
        """简单峰值搜索：找局部极大值"""
        B = spectrum.shape[0]
        device = spectrum.device

        # 计算一阶差分，找局部极大
        sp_diff = torch.diff(spectrum, dim=-1)
        is_peak = (sp_diff[:, :-1] >= 0) & (sp_diff[:, 1:] <= 0)
        is_peak = torch.cat([
            torch.zeros(B, 1, device=device, dtype=torch.bool),
            is_peak,
            torch.zeros(B, 1, device=device, dtype=torch.bool)
        ], dim=-1)

        # 只保留峰值位置的谱值
        peak_vals = torch.where(is_peak, spectrum, torch.full_like(spectrum, -1e9))

        # 取top-k
        _, indices = torch.topk(peak_vals, k, dim=-1)
        indices, _ = torch.sort(indices, dim=-1)

        theta = self.grid[indices]
        success = (peak_vals.gather(1, indices)[:, -1] > self.peak_threshold)

        return success, theta

    def _nms_peak_search(self, spectrum: torch.Tensor, k: int, min_sep: float):
        """NMS峰值搜索：保证最小角度间隔"""
        B = spectrum.shape[0]
        device = spectrum.device
        step = float(self.grid[1] - self.grid[0]) if len(self.grid) > 1 else 1.0
        sep_bins = int(math.ceil(min_sep / step))

        sp_work = spectrum.clone()
        selected_idx = torch.zeros(B, k, dtype=torch.long, device=device)
        selected_val = torch.zeros(B, k, device=device)

        for t in range(k):
            val, idx = torch.max(sp_work, dim=-1)
            selected_idx[:, t] = idx
            selected_val[:, t] = val

            # 抑制周围区域
            for b in range(B):
                center = idx[b].item()
                left = max(0, center - sep_bins)
                right = min(self.out_dim, center + sep_bins + 1)
                sp_work[b, left:right] = -1e9

        # 排序
        selected_idx, sort_order = torch.sort(selected_idx, dim=-1)
        theta = self.grid[selected_idx]
        success = (selected_val.min(dim=-1).values > self.peak_threshold)

        return success, theta
