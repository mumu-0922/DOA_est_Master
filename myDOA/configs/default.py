"""
默认配置
"""

class Config:
    # 阵列参数
    M = 8                    # 阵元数量
    k = 3                    # 信源数量
    d = 0.5                  # 阵元间距（半波长）

    # 信号参数
    snr_range = (-10, 10)    # SNR范围 (dB)
    snap = 200               # 快拍数
    rho = 0.0                # 阵列误差程度

    # 角度参数
    angle_range = (-60, 60)  # 角度范围
    min_sep = 5.0            # 最小角度间隔
    grid_step = 1.0          # 网格步长

    # 数据集
    train_samples = 20000    # 训练样本数
    val_samples = 4000       # 验证样本数
    test_samples = 2000      # 测试样本数

    # 模型参数
    in_channels = 4          # 输入通道（实部+虚部+sin相位+cos相位）
    base_channels = 64       # 基础通道数
    num_blocks = 4           # 残差块数量
    attention_type = 'coord' # 注意力类型: 'coord', 'dual', 'none'
    dropout = 0.2

    # 训练参数
    batch_size = 64
    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    early_stop_patience = 15

    # 损失函数权重
    lambda_spectrum = 1.0    # 空间谱损失权重
    lambda_sparse = 0.1      # 稀疏损失权重
    lambda_sep = 0.1         # 分离损失权重

    # 设备
    device = 'cuda'

    # 保存路径
    save_dir = './results'
    exp_name = 'ca_doa_net'

    @classmethod
    def get_num_classes(cls):
        return int((cls.angle_range[1] - cls.angle_range[0]) / cls.grid_step) + 1


# 实验配置预设
class LowSNRConfig(Config):
    """低SNR实验配置"""
    snr_range = (-20, 0)
    exp_name = 'low_snr'


class HighSNRConfig(Config):
    """高SNR实验配置"""
    snr_range = (0, 20)
    exp_name = 'high_snr'


class LowSnapConfig(Config):
    """低快拍数实验配置"""
    snap = 50
    exp_name = 'low_snap'


class ArrayErrorConfig(Config):
    """阵列误差实验配置"""
    rho = 0.1
    exp_name = 'array_error'
