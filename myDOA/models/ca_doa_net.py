"""
CA-DOA-Net: 融合坐标注意力的DOA估计网络

创新点：
1. 四通道输入：实部 + 虚部 + sin(相位) + cos(相位)
2. 坐标注意力：捕获阵元间的空间相关性
3. 残差连接：改善梯度流动
"""

import torch
import torch.nn as nn

from .base_network import GridBasedNetwork
from .coord_attention import CoordAttention, DualPathCoordAttention


class ConvBlock(nn.Module):
    """卷积块：Conv + BN + ReLU"""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """残差块 + 可选注意力"""

    def __init__(self, channels, use_attention=True, attention_type='coord'):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'coord':
                self.attn = CoordAttention(channels)
            elif attention_type == 'dual':
                self.attn = DualPathCoordAttention(channels)
            else:
                self.attn = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.use_attention:
            out = self.attn(out)

        out = out + identity
        return self.relu(out)


class CA_DOA_Net(GridBasedNetwork):
    """
    融合坐标注意力的DOA估计网络

    Args:
        in_channels: 输入通道数（4通道：实部+虚部+sin相位+cos相位）
        M: 阵元数量
        num_classes: 输出空间谱的网格数
        base_channels: 基础通道数
        num_blocks: 残差块数量
        attention_type: 注意力类型 ('coord', 'dual', 'none')
        start_angle, end_angle, step: 角度网格参数
    """

    def __init__(
        self,
        in_channels: int = 4,
        M: int = 8,
        num_classes: int = 121,
        base_channels: int = 64,
        num_blocks: int = 4,
        attention_type: str = 'coord',
        start_angle: float = -60,
        end_angle: float = 60,
        step: float = 1,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(start_angle, end_angle, step)

        self.M = M
        assert self.out_dim == num_classes, f"网格大小 {self.out_dim} != num_classes {num_classes}"

        # 输入投影层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # 残差块（带注意力）
        channels = base_channels * 2
        self.res_blocks = nn.Sequential(*[
            ResBlock(channels, use_attention=True, attention_type=attention_type)
            for _ in range(num_blocks)
        ])

        # 特征聚合
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 分类头（输出空间谱）
        self.classifier = nn.Sequential(
            nn.Linear(channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, 4, M, M) - 四通道协方差矩阵

        Returns:
            spectrum: 空间谱 (B, num_classes)
        """
        # 特征提取
        x = self.stem(x)
        x = self.res_blocks(x)

        # 全局池化
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 输出空间谱
        spectrum = self.classifier(x)

        return spectrum

    def predict_doa(self, x: torch.Tensor, k: int, min_sep: float = 0.0):
        """
        端到端DOA预测

        Args:
            x: 输入特征
            k: 信源数量
            min_sep: 最小角度间隔

        Returns:
            success: 是否成功
            theta: DOA估计值
            spectrum: 空间谱
        """
        spectrum = self.forward(x)
        success, theta = self.spectrum_to_doa(spectrum, k, min_sep)
        return success, theta, spectrum


class CA_DOA_Net_Light(GridBasedNetwork):
    """轻量版CA-DOA-Net，适用于快速实验"""

    def __init__(
        self,
        in_channels: int = 4,
        M: int = 8,
        num_classes: int = 121,
        start_angle: float = -60,
        end_angle: float = 60,
        step: float = 1,
        **kwargs
    ):
        super().__init__(start_angle, end_angle, step)

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 3, 1, 1),
            CoordAttention(64),
            ConvBlock(64, 128, 3, 1, 1),
            CoordAttention(128),
            ConvBlock(128, 256, 3, 1, 1),
            CoordAttention(256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == '__main__':
    # 测试
    print("测试 CA-DOA-Net...")

    # 模拟输入：batch=4, 4通道, 8x8阵元
    x = torch.randn(4, 4, 8, 8)

    # 测试完整版
    model = CA_DOA_Net(in_channels=4, M=8, num_classes=121)
    out = model(x)
    print(f"CA_DOA_Net: {x.shape} -> {out.shape}")

    # 测试DOA预测
    success, theta, spectrum = model.predict_doa(x, k=3, min_sep=5)
    print(f"DOA预测: success={success}, theta={theta.shape}")

    # 测试轻量版
    model_light = CA_DOA_Net_Light(in_channels=4, M=8, num_classes=121)
    out = model_light(x)
    print(f"CA_DOA_Net_Light: {x.shape} -> {out.shape}")

    # 参数量统计
    params = sum(p.numel() for p in model.parameters())
    params_light = sum(p.numel() for p in model_light.parameters())
    print(f"参数量: 完整版={params/1e6:.2f}M, 轻量版={params_light/1e6:.2f}M")
