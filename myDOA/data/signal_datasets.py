"""
DOA数据集模块

特点：
1. 四通道输入：实部 + 虚部 + sin(相位) + cos(相位)
2. 批量数据生成（张量化加速）
3. 支持多种SNR和快拍数配置
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List


class DOADataset(Dataset):
    """
    DOA估计数据集

    Args:
        M: 阵元数量
        k: 信源数量
        snr_range: SNR范围 (min, max) dB
        snap: 快拍数
        num_samples: 样本数量
        angle_range: 角度范围 (min, max) 度
        min_sep: 最小角度间隔（度）
        grid_step: 角度网格步长
        rho: 阵列误差程度 [0, 1]
        seed: 随机种子
        input_channels: 输入通道数 (2 或 4)
            - 2: 实部 + 虚部 (基线CNN用)
            - 4: 实部 + 虚部 + sin(相位) + cos(相位) (CA-DOA-Net用)
        low_snr_oversample: 低SNR过采样比例，如0.6表示60%样本来自低SNR区间
        low_snr_threshold: 低SNR阈值，低于此值的SNR被视为"低SNR"
    """

    def __init__(
        self,
        M: int = 8,
        k: int = 3,
        snr_range: Tuple[float, float] = (-10, 10),
        snap: int = 200,
        num_samples: int = 10000,
        angle_range: Tuple[float, float] = (-60, 60),
        min_sep: float = 5.0,
        grid_step: float = 1.0,
        sigma: Optional[float] = None,  # 高斯标签宽度，默认为 grid_step
        rho: float = 0.0,
        seed: Optional[int] = None,
        input_channels: int = 4,
        low_snr_oversample: float = 0.0,
        low_snr_threshold: float = -10.0
    ):
        self.M = M
        self.k = k
        self.snr_range = snr_range
        self.snap = snap
        self.num_samples = num_samples
        self.angle_range = angle_range
        self.min_sep = min_sep
        if input_channels not in (2, 4):
            raise ValueError(f"input_channels must be 2 or 4, got {input_channels}")

        self.grid_step = grid_step
        self.sigma = sigma if sigma is not None else grid_step
        self.rho = rho
        self.input_channels = input_channels
        self.low_snr_oversample = low_snr_oversample
        self.low_snr_threshold = low_snr_threshold

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 阵列配置
        self.d = 0.5  # 半波长间距
        self.array_pos = np.arange(M) * self.d

        # 生成阵列误差
        self.mc_mtx, self.ap_mtx, self.pos_err = self._generate_array_imperfection(rho)

        # 角度网格（用于空间谱）
        self.grid = np.arange(angle_range[0], angle_range[1] + 0.001, grid_step)
        self.num_classes = len(self.grid)

        # 预生成数据
        self.data = self._generate_all_data()

    def _generate_array_imperfection(self, rho: float):
        """生成阵列误差矩阵"""
        if rho == 0:
            mc_mtx = np.eye(self.M, dtype=np.complex64)
            ap_mtx = np.eye(self.M, dtype=np.complex64)
            pos_err = np.zeros(self.M)
        else:
            # 互耦矩阵
            mc_mtx = np.eye(self.M, dtype=np.complex64)
            for i in range(self.M):
                for j in range(self.M):
                    if i != j:
                        mc_mtx[i, j] = rho * 0.1 * np.exp(1j * np.random.uniform(-np.pi, np.pi))

            # 幅相误差
            amp_err = 1 + rho * 0.1 * np.random.randn(self.M)
            phase_err = rho * 0.1 * np.random.randn(self.M)
            ap_mtx = np.diag(amp_err * np.exp(1j * phase_err)).astype(np.complex64)

            # 位置误差
            pos_err = rho * 0.05 * self.d * np.random.randn(self.M)

        return mc_mtx, ap_mtx, pos_err

    def _generate_doa_angles(self, batch_size: int) -> np.ndarray:
        """生成满足最小间隔约束的DOA角度"""
        angles = np.zeros((batch_size, self.k))

        for i in range(batch_size):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                # 随机生成k个角度
                a = np.random.uniform(self.angle_range[0], self.angle_range[1], self.k)
                a = np.sort(a)

                # 检查最小间隔
                if self.k == 1:
                    valid = True
                else:
                    diffs = np.diff(a)
                    valid = np.all(diffs >= self.min_sep)
                attempts += 1

            if not valid:
                # 均匀分布作为备选
                a = np.linspace(self.angle_range[0] + 10, self.angle_range[1] - 10, self.k)

            angles[i] = a

        return angles

    def _generate_steering_vector(self, theta: np.ndarray) -> np.ndarray:
        """
        生成导向矢量

        Args:
            theta: 角度 (batch, k) 或 (k,)，单位：度

        Returns:
            A: 导向矢量 (batch, M, k) 或 (M, k)
        """
        theta_rad = np.deg2rad(theta)

        if theta.ndim == 1:
            # 单个样本
            phase = -1j * 2 * np.pi * np.outer(self.array_pos + self.pos_err, np.sin(theta_rad))
            A = np.exp(phase)
            A = self.mc_mtx @ self.ap_mtx @ A
        else:
            # 批量
            batch_size = theta.shape[0]
            sin_theta = np.sin(theta_rad)  # (batch, k)
            phase = -1j * 2 * np.pi * (self.array_pos + self.pos_err)[:, None, None] * sin_theta[None, :, :]
            phase = phase.transpose(1, 0, 2)  # (batch, M, k)
            A = np.exp(phase)
            # 应用阵列误差
            A = np.einsum('ij,bjk->bik', self.mc_mtx @ self.ap_mtx, A)

        return A.astype(np.complex64)

    def _generate_signal(self, batch_size: int, snr_db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成接收信号和协方差矩阵

        Returns:
            scm: 采样协方差矩阵 (batch, M, M)
            doa: 真实DOA角度 (batch, k)
        """
        # 生成DOA角度
        doa = self._generate_doa_angles(batch_size)

        # 导向矢量
        A = self._generate_steering_vector(doa)  # (batch, M, k)

        # 信号功率
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.sqrt(snr_linear / 2)[:, :, None]  # (batch, k, 1)

        # 生成信号 s(t): (batch, k, snap)
        s_real = np.random.randn(batch_size, self.k, self.snap)
        s_imag = np.random.randn(batch_size, self.k, self.snap)
        s = signal_power * (s_real + 1j * s_imag)

        # 生成噪声 n(t): (batch, M, snap)
        n_real = np.random.randn(batch_size, self.M, self.snap)
        n_imag = np.random.randn(batch_size, self.M, self.snap)
        n = (n_real + 1j * n_imag) / np.sqrt(2)

        # 接收信号 x(t) = A @ s(t) + n(t)
        x = np.einsum('bmk,bkt->bmt', A, s) + n  # (batch, M, snap)

        # 采样协方差矩阵
        scm = np.einsum('bmt,bnt->bmn', x, np.conj(x)) / self.snap

        return scm.astype(np.complex64), doa.astype(np.float32)

    def _scm_to_input(self, scm: np.ndarray) -> np.ndarray:
        """
        将协方差矩阵转换为网络输入

        根据 input_channels 参数选择通道数:
        - 2通道: 实部 + 虚部
        - 4通道: 实部 + 虚部 + sin(相位) + cos(相位)

        注意：先对SCM进行归一化 R/trace(R)，消除不同SNR的幅度差异
        """
        # 归一化协方差矩阵（关键！消除SNR导致的幅度差异）
        trace = np.trace(scm, axis1=-2, axis2=-1).real
        scm_norm = scm / (trace[..., None, None] + 1e-10)

        real_part = scm_norm.real
        imag_part = scm_norm.imag

        if self.input_channels == 2:
            # 2通道输入 (基线CNN)
            input_data = np.stack([real_part, imag_part], axis=1)
        else:
            # 4通道输入 (CA-DOA-Net)
            phase = np.angle(scm_norm)
            sin_phase = np.sin(phase)
            cos_phase = np.cos(phase)
            input_data = np.stack([real_part, imag_part, sin_phase, cos_phase], axis=1)

        return input_data.astype(np.float32)

    def _doa_to_spectrum(self, doa: np.ndarray) -> np.ndarray:
        """
        将DOA角度转换为目标空间谱（高斯峰）

        Args:
            doa: DOA角度 (batch, k)

        Returns:
            spectrum: 目标空间谱 (batch, num_classes)
        """
        batch_size = doa.shape[0]
        spectrum = np.zeros((batch_size, self.num_classes), dtype=np.float32)

        sigma = self.sigma  # 使用配置的高斯宽度

        for i in range(batch_size):
            for angle in doa[i]:
                # 在每个DOA位置放置高斯峰
                gaussian = np.exp(-0.5 * ((self.grid - angle) / sigma) ** 2)
                spectrum[i] += gaussian

        # 归一化
        spectrum = spectrum / (spectrum.max(axis=1, keepdims=True) + 1e-8)

        return spectrum

    def _doa_to_offset(self, doa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个DOA相对最近网格点的偏移量

        Args:
            doa: DOA角度 (batch, k)

        Returns:
            offset: 偏移量 (batch, k)，范围 [-0.5*grid_step, +0.5*grid_step]
            nearest_idx: 最近网格点索引 (batch, k)
        """
        # 计算相对于网格起点的位置
        relative_pos = (doa - self.angle_range[0]) / self.grid_step

        # 四舍五入到最近网格点
        nearest_idx = np.round(relative_pos).astype(np.int32)

        # 限制索引范围
        nearest_idx = np.clip(nearest_idx, 0, self.num_classes - 1)

        # 最近网格点的角度
        nearest_angle = self.angle_range[0] + nearest_idx * self.grid_step

        # 偏移量 = 真实角度 - 最近网格点角度
        offset = doa - nearest_angle

        return offset.astype(np.float32), nearest_idx

    def _generate_all_data(self):
        """预生成所有数据"""
        # 根据低SNR过采样比例生成SNR
        if self.low_snr_oversample > 0:
            # 低SNR样本数量
            num_low_snr = int(self.num_samples * self.low_snr_oversample)
            num_high_snr = self.num_samples - num_low_snr

            # 低SNR区间: [snr_min, low_snr_threshold]
            low_snr = np.random.uniform(
                self.snr_range[0], self.low_snr_threshold,
                (num_low_snr, 1)
            )

            # 高SNR区间: [low_snr_threshold, snr_max]
            high_snr = np.random.uniform(
                self.low_snr_threshold, self.snr_range[1],
                (num_high_snr, 1)
            )

            # 合并并打乱
            snr_db = np.vstack([low_snr, high_snr])
            np.random.shuffle(snr_db)
            snr_db = np.repeat(snr_db, self.k, axis=1)
        else:
            # 均匀采样
            snr_db = np.random.uniform(
                self.snr_range[0], self.snr_range[1],
                (self.num_samples, 1)
            )
            snr_db = np.repeat(snr_db, self.k, axis=1)

        # 生成信号
        scm, doa = self._generate_signal(self.num_samples, snr_db)

        # 转换为网络输入 (2通道或4通道)
        input_data = self._scm_to_input(scm)

        # 生成目标空间谱
        spectrum = self._doa_to_spectrum(doa)

        # 生成偏移量标签（用于回归头训练）
        offset, nearest_idx = self._doa_to_offset(doa)

        return {
            'input': torch.from_numpy(input_data),
            'spectrum': torch.from_numpy(spectrum),
            'doa': torch.from_numpy(doa),
            'snr': torch.from_numpy(snr_db.astype(np.float32)),
            'offset': torch.from_numpy(offset),
            'nearest_idx': torch.from_numpy(nearest_idx)
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input': self.data['input'][idx],
            'spectrum': self.data['spectrum'][idx],
            'doa': self.data['doa'][idx],
            'snr': self.data['snr'][idx],
            'offset': self.data['offset'][idx],
            'nearest_idx': self.data['nearest_idx'][idx]
        }


def create_dataloader(
    M: int = 8,
    k: int = 3,
    snr_range: Tuple[float, float] = (-10, 10),
    snap: int = 200,
    num_samples: int = 10000,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """创建DataLoader的便捷函数"""
    dataset = DOADataset(
        M=M, k=k, snr_range=snr_range, snap=snap,
        num_samples=num_samples, **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':
    print("测试数据集...")

    # 创建数据集
    dataset = DOADataset(M=8, k=3, snr_range=(-10, 10), num_samples=100, seed=42)
    print(f"数据集大小: {len(dataset)}")
    print(f"网格大小: {dataset.num_classes}")

    # 获取一个样本
    sample = dataset[0]
    print(f"输入形状: {sample['input'].shape}")  # (4, 8, 8)
    print(f"空间谱形状: {sample['spectrum'].shape}")  # (121,)
    print(f"DOA形状: {sample['doa'].shape}")  # (3,)

    # 测试DataLoader
    loader = create_dataloader(M=8, k=3, num_samples=100, batch_size=16)
    for batch in loader:
        print(f"Batch输入: {batch['input'].shape}")
        print(f"Batch空间谱: {batch['spectrum'].shape}")
        break

    print("测试通过!")
