"""
myDOA: 低信噪比下融合注意力机制的DOA估计框架

硕士论文《低信噪比下融合注意力机制的DOA方法研究》代码实现

项目结构:
    myDOA/
    ├── models/          # 模型定义
    │   ├── coord_attention.py   # 坐标注意力模块
    │   ├── ca_doa_net.py        # CA-DOA-Net网络
    │   └── base_network.py      # 基础网络类
    ├── data/            # 数据模块
    │   └── signal_datasets.py   # DOA数据集
    ├── utils/           # 工具模块
    │   ├── loss_function.py     # 损失函数
    │   ├── trainer.py           # 训练器
    │   └── metrics.py           # 评估指标
    ├── train/           # 训练脚本
    │   └── train_ca_doa.py
    ├── test/            # 测试脚本
    ├── configs/         # 配置文件
    └── results/         # 实验结果

创新点:
    1. 四通道输入: 实部 + 虚部 + sin(相位) + cos(相位)
    2. 坐标注意力: 捕获阵元间的空间相关性
    3. 稀疏约束损失: 利用DOA稀疏先验

使用方法:
    # 训练
    cd myDOA/train
    python train_ca_doa.py --device cuda --epochs 100

    # 测试
    cd myDOA/test
    python test_snr.py --weights ../results/xxx/best_model.pth
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
