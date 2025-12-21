import torch.nn as nn
import torch
import math


class Grid_Based_network(nn.Module):
    def __init__(self, start_angle=-60, end_angle=60, step=1, threshold=0, peak_min_sep_deg: float = 0.0):
        nn.Module.__init__(self)
        # threshold 是寻峰函数中使用的阈值
        # super(Grid_Based_network, self).__init__()
        # ABC.__init__()
        # grid 存在gpu上
        # self._grid = np.arange(start_angle, end_angle + 0.0001, step=step)
        self.register_buffer('_grid', torch.arange(start_angle, end_angle + 0.0001, step=step))
        self._out_dim = self._grid.shape[0]
        self.threshold = threshold
        # peak_min_sep_deg：从空间谱提取 k 个 DOA 时的最小角间隔（度）。
        # 设为 0 表示禁用该约束，保持原先“找局部极大值再取 top-k”的逻辑。
        self.peak_min_sep_deg = float(peak_min_sep_deg)

    @property
    def grid(self):
        return self._grid

    @property
    def out_dim(self):
        return self._out_dim

    def grid_to_theta(self, sp_batch: torch.Tensor, k, return_sp=False):
        # sp_batch (batch, grid+1)
        assert sp_batch.ndim == 2, ''
        b = sp_batch.shape[0]
        device = sp_batch.device

        # ===== NMS-topk（更稳健）：直接从谱值中选择 k 个峰，并强制最小角间隔 =====
        # 低 SNR 下模型输出的谱可能存在多个相邻假峰，导致 top-k 全落在同一簇；
        # 如果真值 DOA 本身满足最小间隔（例如 min_delta_theta=8°），加入该先验可提升多源分辨成功率。
        if self.peak_min_sep_deg and self.peak_min_sep_deg > 0:
            if self._grid.numel() < 2:
                step_deg = 1.0
            else:
                step_deg = float((self._grid[1] - self._grid[0]).detach().cpu().item())
            sep_bins = int(math.ceil(self.peak_min_sep_deg / max(step_deg, 1e-9)))

            sp_work = sp_batch.clone()
            selected_idx = torch.full((b, k), -1, device=device, dtype=torch.long)
            selected_score = torch.full((b, k), float("-inf"), device=device, dtype=sp_batch.dtype)

            for t in range(k):
                score, idx = torch.max(sp_work, dim=-1)  # (b,)
                selected_idx[:, t] = idx
                selected_score[:, t] = score

                # suppress around selected peaks (per-sample slicing)
                for bi in range(b):
                    center = int(idx[bi].item())
                    left = max(0, center - sep_bins)
                    right = min(self.out_dim, center + sep_bins + 1)
                    sp_work[bi, left:right] = float("-inf")

            succ = (selected_score[:, k - 1] > self.threshold).bool().to(device)
            theta = self.grid[selected_idx]
            theta, _ = theta.sort(dim=-1)

            if return_sp:
                return succ, theta, sp_batch
            else:
                return succ, theta
        sp_diff = torch.diff(sp_batch, n=1, dim=-1)
        sp_peak = (sp_diff[:, :-1] >= 0) & (sp_diff[:, 1:] <= 0)  # bool (batch, grid-1)
        sp_peak = torch.cat([torch.zeros((b, 1), device=device, dtype=torch.bool), sp_peak,
                             torch.zeros((b, 1), device=device, dtype=torch.bool)], dim=-1)  # bool (batch, grid+1)
        sp_peak = torch.where(sp_peak, sp_batch, torch.zeros_like(sp_batch, device=device) - self.threshold)
        # sp_peak = sp_batch[sp_peak]

        sp_peak, idx = sp_peak.sort(dim=-1, descending=True)
        succ = sp_peak[:, k - 1].bool().to(sp_peak.device)
        theta = self.grid[idx[:, :k]]
        theta, _ = theta.sort(dim=-1)

        if return_sp:
            return succ, theta, sp_batch
        else:
            return succ, theta


class std_CNN(Grid_Based_network):
    def __init__(self, in_c, M, out_dims, sp_mode=True, **kwargs):
        if sp_mode:
            Grid_Based_network.__init__(self, **kwargs)
            assert self._grid.shape[0] == out_dims, 'error grid_size or model output dims'
            self.sp_to_doa = self.grid_to_theta  # 重命名 空间谱估计->角度的函数
        else:
            # super(VisionTransformer, self).__init__()
            nn.Module.__init__(self)
        # 巻积不变大小,原文是改变大小的
        # self.conv_seq1 = nn.Sequential(nn.Conv2d(in_c, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.conv_seq2 = nn.Sequential(nn.Conv2d(256, 256, 2, 1, 'same'), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.conv_seq3 = nn.Sequential(nn.Conv2d(256, 256, 2, 1, 'same'), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 256, 2, 1, 'same'), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq1 = nn.Sequential(nn.Conv2d(in_c, 256, 3, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(256, 256, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(256, 256, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 256, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # self.len_i = (M ** 2) * 256
        self.len_i = ((M-2-1-1-1) ** 2) * 256
        # M=8时,len_i = 16384
        # fc 可能导致nan嘛
        self.fc_seq1 = nn.Sequential(nn.Linear(self.len_i, 4096), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.fc_seq2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.fc_seq3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2))
        # self.fc_seq1 = nn.Sequential(nn.Linear(self.len_i, 4096), nn.ReLU(inplace=True))
        # self.fc_seq2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(inplace=True))
        # self.fc_seq3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(inplace=True))

        self.out_layer = nn.Sequential(nn.Linear(1024, self.out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)
        x = torch.flatten(x, 1)

        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_seq3(x)
        x = self.out_layer(x)

        return x


# using average pool
class modified_CNN(Grid_Based_network):
    def __init__(self, in_c, M, out_dims, sp_mode=True, **kwargs):
        if sp_mode:
            Grid_Based_network.__init__(self, **kwargs)
            assert self._grid.shape[0] == out_dims, 'error grid_size or model output dims'
            self.sp_to_doa = self.grid_to_theta  # 重命名 空间谱估计->角度的函数
        else:
            # super(VisionTransformer, self).__init__()
            nn.Module.__init__(self)
        self.conv_seq1 = nn.Sequential(nn.Conv2d(in_c, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # self.len_i = (M ** 2) * 256
        self.len_i = ((M-2-1-1-1) ** 2) * 256
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_seq1 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.out_layer = nn.Sequential(nn.Linear(1024, self.out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc_seq1(x)
        x = self.out_layer(x)

        return x


# class std_CNN_2(nn.Module, Grid_Based_network):
#     def __init__(self, in_c, device='cuda', start_angle=-60, end_angle=60, step=1):
#         super(std_CNN_2, self).__init__()
#         Grid_Based_network.__init__(self, device, start_angle, end_angle, step)
#         # 巻积不变大小,原文是改变大小的
#         self.conv_seq1 = nn.Sequential(nn.Conv2d(in_c, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
#         self.conv_seq2 = nn.Sequential(nn.Conv2d(256, 512, 2, 1, 'same'), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
#         self.conv_seq3 = nn.Sequential(nn.Conv2d(512, 512, 2, 1, 'same'), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
#         self.conv_seq4 = nn.Sequential(nn.Conv2d(512, 1024, 2, 1, 'same'), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
#
#         # M=8时,len_i = 16384
#         # fc 可能导致nan嘛
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc_seq1 = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(inplace=True), nn.Dropout(0.2))
#         self.fc_seq2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(inplace=True), nn.Dropout(0.2))
#         self.fc_seq3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2))
#
#         self.out_layer = nn.Sequential(nn.Linear(1024, self.out_dim))
#         self.sigmoid = nn.Sigmoid()
#
#     # k 写在前向传播中,可以通过调整k让前向传播函数取得不同的谱峰
#     # k 作为类的变量的话更简单,但是难以调整
#     def forward(self, x, k=3, pre_mode=False):
#         x = self.conv_seq1(x)
#         x = self.conv_seq2(x)
#         x = self.conv_seq3(x)
#         x = self.conv_seq4(x)
#         x = torch.flatten(self.avg_pool(x), 1)
#
#         x = self.fc_seq1(x)
#         x = self.fc_seq2(x)
#         x = self.fc_seq3(x)
#         x = self.out_layer(x)
#         if pre_mode is True:
#             x = self.sigmoid(x)
#             x = self.grid_to_thetas_method2(x, k)
#
#         return x



