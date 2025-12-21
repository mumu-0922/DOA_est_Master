import torch
import torch.nn as nn

from .literature_CNN import std_CNN


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 通道注意力模块。
    作用：根据全局平均池化得到的通道描述，为每个通道生成权重，实现“抑噪/增强有效特征”。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class std_CNN_SE(std_CNN):
    """
    在仓库原始 std_CNN 的基础上，插入 SE-Block（通道注意力）。

    用法（与 std_CNN 基本一致）：
        from models.dl_model.CNN.std_cnn_se import std_CNN_SE
        model = std_CNN_SE(3, M=8, out_dims=121, sp_mode=True)
    """

    def __init__(self, in_c, M, out_dims, sp_mode=True, se_reduction: int = 16, **kwargs):
        super().__init__(in_c, M, out_dims, sp_mode=sp_mode, **kwargs)
        self.se = SEBlock(256, reduction=se_reduction)

    def forward(self, x):
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)
        x = self.se(x)
        x = torch.flatten(x, 1)

        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_seq3(x)
        x = self.out_layer(x)

        return x

