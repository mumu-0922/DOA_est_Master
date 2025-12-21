import torch
import torch.nn as nn

from .literature_CNN import std_CNN


class ChannelAttention(nn.Module):
    """
    CBAM: Channel Attention.
    Using both global average pooling and global max pooling to generate channel weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        w = self.act(avg_out + max_out)
        return x * w


class SpatialAttention(nn.Module):
    """
    CBAM: Spatial Attention.
    Aggregate across channels (avg/max) and learn a spatial mask.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class CBAM(nn.Module):
    """
    CBAM block = Channel Attention + Spatial Attention.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class std_CNN_CBAM(std_CNN):
    """
    std_CNN + CBAM (channel + spatial attention).

    Usage:
        from models.dl_model.CNN.std_cnn_cbam import std_CNN_CBAM
        model = std_CNN_CBAM(in_c=3, M=8, out_dims=121, sp_mode=True)
    """

    def __init__(
        self,
        in_c,
        M,
        out_dims,
        sp_mode=True,
        cbam_reduction: int = 16,
        spatial_kernel_size: int = 3,
        cbam_each_stage: bool = True,
        **kwargs,
    ):
        super().__init__(in_c, M, out_dims, sp_mode=sp_mode, **kwargs)
        self.cbam_each_stage = cbam_each_stage
        if cbam_each_stage:
            self.cbam1 = CBAM(256, reduction=cbam_reduction, spatial_kernel_size=spatial_kernel_size)
            self.cbam2 = CBAM(256, reduction=cbam_reduction, spatial_kernel_size=spatial_kernel_size)
            self.cbam3 = CBAM(256, reduction=cbam_reduction, spatial_kernel_size=spatial_kernel_size)
            self.cbam4 = CBAM(256, reduction=cbam_reduction, spatial_kernel_size=spatial_kernel_size)
        else:
            self.cbam = CBAM(256, reduction=cbam_reduction, spatial_kernel_size=spatial_kernel_size)

    def forward(self, x):
        if self.cbam_each_stage:
            x = self.cbam1(self.conv_seq1(x))
            x = self.cbam2(self.conv_seq2(x))
            x = self.cbam3(self.conv_seq3(x))
            x = self.cbam4(self.conv_seq4(x))
        else:
            x = self.conv_seq1(x)
            x = self.conv_seq2(x)
            x = self.conv_seq3(x)
            x = self.conv_seq4(x)
            x = self.cbam(x)
        x = torch.flatten(x, 1)

        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_seq3(x)
        x = self.out_layer(x)
        return x
