# myDOA: 低信噪比下融合注意力机制的DOA估计框架

## 项目简介

本项目是硕士论文《低信噪比下融合注意力机制的DOA方法研究》的代码实现。

### 创新点

1. **四通道输入特征增强**：实部 + 虚部 + sin(相位) + cos(相位)
2. **坐标注意力机制**：捕获阵元间的空间相关性
3. **物理约束损失函数**：空间谱稀疏约束 + 角度分离约束

## 项目结构

```
myDOA/
├── models/                 # 模型定义
│   ├── coord_attention.py  # 坐标注意力模块
│   ├── ca_doa_net.py       # CA-DOA-Net 网络
│   └── base_network.py     # 基础网络类
├── data/                   # 数据模块
│   └── signal_datasets.py  # DOA数据集（4通道输入）
├── utils/                  # 工具模块
│   ├── loss_function.py    # 损失函数（含稀疏约束）
│   ├── trainer.py          # 训练器
│   └── metrics.py          # 评估指标
├── train/                  # 训练脚本
│   └── train_ca_doa.py     # 主训练脚本
├── test/                   # 测试脚本
│   └── test_snr.py         # SNR扫描测试
├── configs/                # 配置文件
│   └── default.py          # 默认配置
└── results/                # 实验结果
```

## 快速开始

### 1. 环境配置

```bash
conda activate DOA  # 使用已有环境
```

### 2. 训练模型

```powershell
cd myDOA/train

# 基础训练
python train_ca_doa.py --device cuda --epochs 100

# 低SNR训练
python train_ca_doa.py --device cuda --snr_min -20 --snr_max 0 --epochs 150

# 轻量版模型
python train_ca_doa.py --device cuda --model light --epochs 80
```

### 3. 测试模型

```powershell
cd myDOA/test

python test_snr.py --weights ../results/xxx/best_model.pth --snr_list -20 -15 -10 -5 0 5 10
```

## 核心组件说明

### 坐标注意力 (Coordinate Attention)

```python
from models.coord_attention import CoordAttention

ca = CoordAttention(in_channels=256)
out = ca(feature_map)  # (B, C, H, W) -> (B, C, H, W)
```

### 四通道输入

```python
# 协方差矩阵 -> 四通道
input = [
    scm.real,              # 通道1：实部
    scm.imag,              # 通道2：虚部
    sin(angle(scm)),       # 通道3：相位sin
    cos(angle(scm))        # 通道4：相位cos
]
```

### 联合损失函数

```python
from utils.loss_function import CombinedLoss

loss_fn = CombinedLoss(
    lambda_spectrum=1.0,   # 空间谱损失
    lambda_sparse=0.1,     # 稀疏约束
    lambda_sep=0.1         # 分离约束
)
```

## 消融实验设计

| 实验 | 输入 | 注意力 | 损失 | 目的 |
|------|------|--------|------|------|
| Exp1 | 2通道 | 无 | MSE | Baseline |
| Exp2 | 4通道 | 无 | MSE | 验证输入改进 |
| Exp3 | 4通道 | CA | MSE | 验证注意力 |
| Exp4 | 4通道 | CA | MSE+Sparse | 验证损失函数 |
| Exp5 | 4通道 | CA | 全部 | 完整方法 |

## 参考文献

1. Coordinate Attention (CVPR 2021)
2. SPCII Attention (2024)
3. TransMUSIC (ICASSP 2024)
