"""
传统子空间DOA估计方法

包含:
- MUSIC: 多重信号分类算法
- ESPRIT: 旋转不变子空间算法

参考: models/subspace_model/ 下的实现
"""

import numpy as np
from scipy.signal import find_peaks


class Music:
    """
    MUSIC (Multiple Signal Classification) 算法

    Args:
        M: 阵元数量
        d: 阵元间距（波长倍数），默认0.5
        start: 搜索起始角度
        end: 搜索结束角度
        step: 搜索步长
    """

    def __init__(self, M: int = 8, d: float = 0.5, start: float = -60, end: float = 60, step: float = 0.1):
        self.M = M
        self.d = d
        self._search_grid = np.arange(start, end + 0.001, step)

        # 预计算导向矢量矩阵
        array_pos = np.arange(M) * d
        theta_rad = np.deg2rad(self._search_grid)
        phase = -1j * 2 * np.pi * np.outer(array_pos, np.sin(theta_rad))
        self._A = np.exp(phase)  # (M, num_grid)

    def estimate(self, R: np.ndarray, k: int, return_sp: bool = False):
        """
        估计DOA角度

        Args:
            R: 协方差矩阵 (M, M)
            k: 信源数量
            return_sp: 是否返回空间谱

        Returns:
            success: 是否成功找到k个峰值
            doa: 估计的DOA角度
            spectrum: 空间谱（可选）
        """
        # 获取噪声子空间
        U_n = self._get_noise_subspace(R, k)

        # 计算MUSIC空间谱
        spectrum = self._compute_spectrum(U_n)

        # 寻峰
        peak_indices = find_peaks(spectrum)[0]

        n_peaks = len(peak_indices)
        if n_peaks < k:
            # 峰值不足，返回失败
            doa = self._search_grid[peak_indices] if n_peaks > 0 else np.array([])
            doa = np.concatenate([doa, np.full(k - len(doa), np.nan)])
            if return_sp:
                return False, doa, spectrum
            return False, doa

        # 选择最大的k个峰值
        peak_values = spectrum[peak_indices]
        top_indices = np.argsort(peak_values)[-k:]
        selected_peaks = sorted([peak_indices[i] for i in top_indices])
        doa = self._search_grid[selected_peaks]

        if return_sp:
            return True, doa, spectrum
        return True, doa

    def _get_noise_subspace(self, R: np.ndarray, k: int) -> np.ndarray:
        """获取噪声子空间"""
        eigenvalues, U = np.linalg.eigh(R)
        # 特征值按升序排列，前M-k个对应噪声子空间
        U_n = U[:, :-k]
        return U_n

    def _compute_spectrum(self, U_n: np.ndarray) -> np.ndarray:
        """计算MUSIC空间谱"""
        # P_MUSIC = 1 / (a^H * U_n * U_n^H * a)
        v = U_n.T.conj() @ self._A  # (M-k, num_grid)
        denominator = np.sum(v * v.conj(), axis=0).real
        spectrum = np.reciprocal(denominator + 1e-10)
        return spectrum.astype(np.float32)


class ESPRIT:
    """
    ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) 算法

    Args:
        M: 阵元数量
        d: 阵元间距（波长倍数），默认0.5
        displacement: 子阵列位移，默认1
    """

    def __init__(self, M: int = 8, d: float = 0.5, displacement: int = 1):
        self.M = M
        self.d = d
        self.displacement = displacement

        # 定义两个子阵列的索引
        idx = np.arange(M)
        self.sub_arr1 = idx[:-displacement]
        self.sub_arr2 = idx[displacement:]

    def estimate(self, R: np.ndarray, k: int) -> tuple:
        """
        TLS-ESPRIT估计DOA角度

        Args:
            R: 协方差矩阵 (M, M)
            k: 信源数量

        Returns:
            success: 是否成功
            doa: 估计的DOA角度
        """
        try:
            # 获取信号子空间
            U_s = self._get_signal_subspace(R, k)

            # 提取两个子阵列的信号子空间
            U_s1 = U_s[self.sub_arr1, :]
            U_s2 = U_s[self.sub_arr2, :]

            # TLS-ESPRIT
            U_s12 = np.concatenate([U_s1, U_s2], axis=1)
            G = U_s12.T.conj() @ U_s12

            _, E = np.linalg.eigh(G)
            E = np.fliplr(E)  # 降序排列

            E1 = E[:k, k:]
            E2 = E[k:, k:]

            # Phi = -E1 * inv(E2)
            Phi = -np.linalg.solve(E2.T, E1.T).T

            # 计算特征值
            z = np.linalg.eigvals(Phi)

            # 从z计算角度
            doa = self._z_to_theta(z)
            doa = np.sort(doa.real)

            # 检查角度是否在有效范围内
            if np.any(np.abs(doa) > 90) or np.any(np.isnan(doa)):
                return False, doa

            return True, doa

        except Exception:
            return False, np.full(k, np.nan)

    def _get_signal_subspace(self, R: np.ndarray, k: int) -> np.ndarray:
        """获取信号子空间"""
        eigenvalues, U = np.linalg.eigh(R)
        # 特征值按升序排列，后k个对应信号子空间
        U_s = U[:, -k:]
        return U_s

    def _z_to_theta(self, z: np.ndarray) -> np.ndarray:
        """从z计算角度"""
        # z = exp(-j * 2 * pi * d * sin(theta))
        # theta = arcsin(angle(z) / (2 * pi * d))
        return np.rad2deg(np.arcsin(np.angle(z) / (2 * np.pi * self.d)))
