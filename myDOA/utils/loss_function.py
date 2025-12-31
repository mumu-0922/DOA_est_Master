"""
DOA估计损失函数

包含：
1. MSE损失：基础回归损失
2. 稀疏损失：空间谱稀疏约束
3. 联合损失：自适应加权组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DOALoss(nn.Module):
    """
    DOA角度回归损失

    支持多种损失类型：
    - mse: 均方误差
    - mae: 平均绝对误差
    - huber: Huber损失（对异常值鲁棒）
    """

    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"未知损失类型: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测DOA (batch, k)
            target: 真实DOA (batch, k)
        """
        return self.loss_fn(pred, target)


class SpectrumLoss(nn.Module):
    """
    空间谱损失

    用于监督网络输出的空间谱与目标谱的匹配程度
    """

    def __init__(self, loss_type: str = 'bce'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_spectrum: 预测空间谱 (batch, num_classes)
            target_spectrum: 目标空间谱 (batch, num_classes)
        """
        if self.loss_type == 'bce':
            # 归一化到 [0, 1]
            pred_norm = torch.sigmoid(pred_spectrum)
            return F.binary_cross_entropy(pred_norm, target_spectrum)
        elif self.loss_type == 'mse':
            return F.mse_loss(pred_spectrum, target_spectrum)
        elif self.loss_type == 'kl':
            # KL散度
            pred_log = F.log_softmax(pred_spectrum, dim=-1)
            target_prob = F.softmax(target_spectrum, dim=-1)
            return F.kl_div(pred_log, target_prob, reduction='batchmean')
        else:
            return F.mse_loss(pred_spectrum, target_spectrum)


class ScaleInvariantSpectrumLoss(nn.Module):
    """
    尺度不变空间谱损失 (SI-SDR 形式)

    基于论文: Chen & Rao, "A Comparative Study of Invariance-Aware Loss
    Functions for Deep Learning-based Gridless DoA Estimation", ICASSP 2025

    核心思想：
    - 传统 MSE 损失对缩放敏感：αR 和 R 有相同子空间，但 MSE(αR, R) 很大
    - SI-SDR 损失通过最优缩放因子消除这种敏感性，扩大解空间

    数学公式：
        α* = argmin_α ||αR - R_hat||_F = <pred, target> / ||target||^2
        L_SI = -log(||α*·target||_F / (ε + ||α*·target - pred||_F))

    优势：
    - 解空间从单点扩展为直线（论文 Fig.3 显示 SI-Cov 优于 Cov）
    - 在 k=3,4,5 源时性能显著提升
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: 数值稳定性常数，论文中设为 0，实际实现建议 1e-8
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
        """
        计算 SI-SDR 损失

        Args:
            pred_spectrum: 预测空间谱 (batch, num_classes)
            target_spectrum: 目标空间谱 (batch, num_classes)

        Returns:
            SI-SDR 损失值（标量）
        """
        # 计算最优缩放因子 α* = <pred, target> / ||target||^2
        # 这是 argmin_α ||α·target - pred||_F 的闭式解
        dot_product = torch.sum(pred_spectrum * target_spectrum, dim=-1, keepdim=True)
        target_energy = torch.sum(target_spectrum ** 2, dim=-1, keepdim=True) + self.epsilon
        alpha_star = dot_product / target_energy

        # 缩放后的目标信号
        scaled_target = alpha_star * target_spectrum

        # 计算 SI-SDR
        # SI-SDR = 10 * log10(||α*·target||^2 / ||α*·target - pred||^2)
        signal_power = torch.sum(scaled_target ** 2, dim=-1)
        error_power = torch.sum((scaled_target - pred_spectrum) ** 2, dim=-1) + self.epsilon

        # 使用 log 形式（与论文公式一致）
        # L = -log(||α*·target|| / (ε + ||α*·target - pred||))
        # 等价于最大化 SI-SDR
        si_sdr = 10.0 * torch.log10(signal_power / error_power + self.epsilon)

        # 返回负 SI-SDR 作为损失（因为要最大化 SI-SDR）
        return -si_sdr.mean()


class SparsityLoss(nn.Module):
    """
    稀疏约束损失

    鼓励空间谱只在少数位置有显著响应（峰值稀疏）

    原理：DOA估计中，真实信源数量k远小于网格数，
         因此空间谱应该是稀疏的（只有k个峰）
    """

    def __init__(self, sparsity_type: str = 'l1', temperature: float = 1.0):
        super().__init__()
        self.sparsity_type = sparsity_type
        self.temperature = temperature

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum: 空间谱 (batch, num_classes)
        """
        # 先做softmax归一化
        prob = F.softmax(spectrum / self.temperature, dim=-1)

        if self.sparsity_type == 'l1':
            # L1范数：直接惩罚非零值
            return torch.mean(torch.abs(prob))

        elif self.sparsity_type == 'entropy':
            # 负熵：熵越小越稀疏
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
            return torch.mean(entropy)

        elif self.sparsity_type == 'gini':
            # Gini系数：衡量不均匀程度
            sorted_prob, _ = torch.sort(prob, dim=-1)
            n = prob.shape[-1]
            indices = torch.arange(1, n + 1, device=prob.device, dtype=prob.dtype)
            gini = 1 - 2 * torch.sum(sorted_prob * (n - indices + 0.5) / n, dim=-1) / (torch.sum(sorted_prob, dim=-1) + 1e-8)
            return torch.mean(1 - gini)  # 1 - gini，因为我们希望稀疏（gini大）

        elif self.sparsity_type == 'peak_ratio':
            # 峰值比：top-k值占总和的比例应该高
            k = 3  # 假设3个源
            topk_vals, _ = torch.topk(prob, k, dim=-1)
            peak_sum = torch.sum(topk_vals, dim=-1)
            total_sum = torch.sum(prob, dim=-1)
            ratio = peak_sum / (total_sum + 1e-8)
            return torch.mean(1 - ratio)  # 比例越高越好

        else:
            return torch.mean(torch.abs(prob))


class SeparationLoss(nn.Module):
    """
    角度分离损失

    惩罚预测的DOA角度过于接近，保证多源分辨能力
    """

    def __init__(self, min_sep: float = 5.0):
        super().__init__()
        self.min_sep = min_sep

    def forward(self, pred_doa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_doa: 预测DOA (batch, k)，已排序
        """
        if pred_doa.shape[1] < 2:
            return torch.tensor(0.0, device=pred_doa.device)

        # 计算相邻角度差
        diffs = torch.diff(pred_doa, dim=-1)  # (batch, k-1)

        # 惩罚小于min_sep的差值
        violations = F.relu(self.min_sep - diffs)

        return torch.mean(violations)


class CombinedLoss(nn.Module):
    """
    联合损失函数

    L_total = λ1 * L_spectrum + λ2 * L_sparse + λ3 * L_separation

    Args:
        lambda_spectrum: 空间谱损失权重
        lambda_sparse: 稀疏损失权重
        lambda_sep: 分离损失权重
        spectrum_loss_type: 空间谱损失类型
            - 'mse': 均方误差（传统方法）
            - 'bce': 二元交叉熵
            - 'kl': KL散度
            - 'si_sdr': 尺度不变 SI-SDR（推荐，基于 Chen & Rao 2025）
        sparsity_type: 稀疏损失类型 ('entropy', 'l1', 'gini', 'peak_ratio')
    """

    def __init__(
        self,
        lambda_spectrum: float = 1.0,
        lambda_sparse: float = 0.1,
        lambda_sep: float = 0.1,
        spectrum_loss_type: str = 'mse',
        sparsity_type: str = 'entropy'
    ):
        super().__init__()

        self.lambda_spectrum = lambda_spectrum
        self.lambda_sparse = lambda_sparse
        self.lambda_sep = lambda_sep
        self.spectrum_loss_type = spectrum_loss_type

        # 根据类型选择空间谱损失函数
        if spectrum_loss_type == 'si_sdr':
            self.spectrum_loss = ScaleInvariantSpectrumLoss()
        else:
            self.spectrum_loss = SpectrumLoss(spectrum_loss_type)

        self.sparsity_loss = SparsityLoss(sparsity_type)
        self.separation_loss = SeparationLoss(min_sep=5.0)

    def forward(
        self,
        pred_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
        pred_doa: torch.Tensor = None,
        target_doa: torch.Tensor = None
    ) -> dict:
        """
        计算联合损失

        Returns:
            dict: 包含各项损失和总损失
        """
        losses = {}

        # 空间谱损失
        loss_spectrum = self.spectrum_loss(pred_spectrum, target_spectrum)
        losses['spectrum'] = loss_spectrum

        # 稀疏损失
        loss_sparse = self.sparsity_loss(pred_spectrum)
        losses['sparse'] = loss_sparse

        # 分离损失（如果提供了DOA预测）
        if pred_doa is not None:
            loss_sep = self.separation_loss(pred_doa)
            losses['separation'] = loss_sep
        else:
            loss_sep = torch.tensor(0.0, device=pred_spectrum.device)
            losses['separation'] = loss_sep

        # 总损失
        total = (
            self.lambda_spectrum * loss_spectrum +
            self.lambda_sparse * loss_sparse +
            self.lambda_sep * loss_sep
        )
        losses['total'] = total

        return losses


if __name__ == '__main__':
    print("测试损失函数...")

    batch_size = 4
    num_classes = 121
    k = 3

    # 模拟数据
    pred_spectrum = torch.randn(batch_size, num_classes)
    target_spectrum = torch.rand(batch_size, num_classes)
    pred_doa = torch.sort(torch.rand(batch_size, k) * 120 - 60, dim=-1)[0]

    # 测试各损失
    print("\n--- 传统损失函数 ---")
    spectrum_loss = SpectrumLoss('mse')
    print(f"MSE 空间谱损失: {spectrum_loss(pred_spectrum, target_spectrum).item():.4f}")

    # 测试 SI-SDR 损失
    print("\n--- SI-SDR 损失函数 (Chen & Rao 2025) ---")
    si_sdr_loss = ScaleInvariantSpectrumLoss()
    print(f"SI-SDR 空间谱损失: {si_sdr_loss(pred_spectrum, target_spectrum).item():.4f}")

    # 验证尺度不变性：缩放后的损失应该接近
    scaled_pred = 2.0 * pred_spectrum
    print(f"MSE(2x pred, target): {spectrum_loss(scaled_pred, target_spectrum).item():.4f}")
    print(f"SI-SDR(2x pred, target): {si_sdr_loss(scaled_pred, target_spectrum).item():.4f}")

    sparse_loss = SparsityLoss('entropy')
    print(f"\n稀疏损失: {sparse_loss(pred_spectrum).item():.4f}")

    sep_loss = SeparationLoss(min_sep=5.0)
    print(f"分离损失: {sep_loss(pred_doa).item():.4f}")

    # 测试联合损失 (MSE)
    print("\n--- 联合损失对比 ---")
    combined_mse = CombinedLoss(lambda_spectrum=1.0, lambda_sparse=0.1, lambda_sep=0.1,
                                spectrum_loss_type='mse')
    losses_mse = combined_mse(pred_spectrum, target_spectrum, pred_doa)
    print(f"联合损失 (MSE): total={losses_mse['total'].item():.4f}")

    # 测试联合损失 (SI-SDR)
    combined_si = CombinedLoss(lambda_spectrum=1.0, lambda_sparse=0.1, lambda_sep=0.1,
                               spectrum_loss_type='si_sdr')
    losses_si = combined_si(pred_spectrum, target_spectrum, pred_doa)
    print(f"联合损失 (SI-SDR): total={losses_si['total'].item():.4f}")

    print("\n测试通过!")
