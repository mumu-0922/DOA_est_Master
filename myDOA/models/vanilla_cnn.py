"""
Vanilla CNN (标准CNN) - 用于对比实验的基线模型

来源: models/dl_model/CNN/literature_CNN.py 中的 std_CNN
特点:
- 无注意力机制
- 无残差连接
- 2通道输入 (实部 + 虚部)
"""

import torch
import torch.nn as nn
from .base_network import GridBasedNetwork


class VanillaCNN(GridBasedNetwork):
    """
    标准CNN基线模型

    网络结构:
    - 4层卷积 (Conv + BN + ReLU)
    - 3层全连接 (FC + ReLU + Dropout)
    - 输出层

    Args:
        in_channels: 输入通道数 (默认2: 实部+虚部)
        M: 阵元数量
        num_classes: 输出空间谱的网格数
        start_angle, end_angle, step: 角度网格参数
    """

    def __init__(
        self,
        in_channels: int = 2,
        M: int = 8,
        num_classes: int = 121,
        start_angle: float = -60,
        end_angle: float = 60,
        step: float = 1,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(start_angle, end_angle, step)

        assert self.out_dim == num_classes, f"网格大小 {self.out_dim} != num_classes {num_classes}"

        # 4层卷积 (与原始 std_CNN 一致)
        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_seq3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_seq4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 计算展平后的尺寸: M -> M-2 -> M-3 -> M-4 -> M-5
        # M=8 时: 8 -> 6 -> 5 -> 4 -> 3, 所以 len_i = 3*3*256 = 2304
        feat_size = M - 2 - 1 - 1 - 1  # = M - 5
        self.len_i = (feat_size ** 2) * 256

        # 3层全连接
        self.fc_seq1 = nn.Sequential(
            nn.Linear(self.len_i, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc_seq2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc_seq3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 输出层
        self.out_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, 2, M, M) - 实部+虚部

        Returns:
            spectrum: 空间谱 (B, num_classes)
        """
        # 卷积层
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)

        # 展平
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_seq3(x)
        x = self.out_layer(x)

        return x

    def predict_doa(self, x: torch.Tensor, k: int, min_sep: float = 0.0):
        """
        端到端DOA预测
        """
        spectrum = self.forward(x)
        success, theta = self.spectrum_to_doa(spectrum, k, min_sep)
        return success, theta, spectrum


if __name__ == '__main__':
    print("测试 VanillaCNN...")

    # 模拟输入：batch=4, 2通道(实部+虚部), 8x8阵元
    x = torch.randn(4, 2, 8, 8)

    model = VanillaCNN(in_channels=2, M=8, num_classes=121)
    out = model(x)
    print(f"VanillaCNN: {x.shape} -> {out.shape}")

    # 参数量统计
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params/1e6:.2f}M")
