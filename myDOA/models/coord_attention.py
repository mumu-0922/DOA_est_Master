"""
Coordinate Attention (CA) 模块
论文: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)

核心思想:
- 将通道注意力分解为两个1D特征编码过程
- 分别沿X和Y方向聚合特征，保留精确的位置信息
- 通过坐标信息增强特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAttention(nn.Module):
    """
    坐标注意力模块

    Args:
        in_channels: 输入通道数
        reduction: 通道压缩比例
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn(self.conv_reduce(y)))

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        attn_h = self.sigmoid(self.conv_h(x_h))
        attn_w = self.sigmoid(self.conv_w(x_w))

        return x * attn_h * attn_w


class DualPathCoordAttention(nn.Module):
    """
    双路径坐标注意力 = 坐标注意力 + 通道注意力 + 自适应融合
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.coord_attn = CoordAttention(in_channels, reduction)

        # 通道注意力分支（类似SE）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coord_out = self.coord_attn(x)
        channel_out = x * self.channel_attn(x)

        alpha = torch.sigmoid(self.alpha)
        out = alpha * coord_out + (1 - alpha) * channel_out

        return out + x  # 残差连接
