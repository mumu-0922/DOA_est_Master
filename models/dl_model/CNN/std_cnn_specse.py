import torch
import torch.nn as nn

from .literature_CNN import std_CNN


class SpectralSE(nn.Module):
    """
    频谱维度的 SE（对输出空间谱的每个 grid bin 做“注意力加权”）。

    设计动机（开题可讲）：
    - 输入端注意力（SE/CBAM）作用在特征图上，可能在极低 SNR 下不稳定；
    - 这里直接对“空间谱 logits”做轻量重标定，让网络更倾向于输出“尖峰 + 低旁瓣”，提升多源分辨成功率（PoR）。
    """

    def __init__(self, out_dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, out_dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, out_dim)
        w = self.mlp(logits)
        # residual-style gating，避免 attention 过强导致整体塌缩
        return logits * (1.0 + w)


class std_CNN_SpecSE(std_CNN):
    """
    std_CNN + SpectralSE（对输出空间谱做注意力）。
    """

    def __init__(self, in_c, M, out_dims, sp_mode=True, specse_reduction: int = 16, **kwargs):
        super().__init__(in_c, M, out_dims, sp_mode=sp_mode, **kwargs)
        self.spec_se = SpectralSE(self.out_dim, reduction=specse_reduction)

    def forward(self, x):
        logits = super().forward(x)
        logits = self.spec_se(logits)
        return logits

