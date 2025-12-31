"""
CA-DOA-Net 训练脚本

使用方法:
    # 使用传统 MSE 损失
    python train_ca_doa.py --device cuda --epochs 100 --snr_min -10 --snr_max 10

    # 使用 SI-SDR 损失（推荐，基于 Chen & Rao 2025）
    python train_ca_doa.py --device cuda --loss_type si_sdr --lr 5e-4

参考论文:
    Chen & Rao, "A Comparative Study of Invariance-Aware Loss Functions
    for Deep Learning-based Gridless DoA Estimation", ICASSP 2025
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from datetime import datetime

from data.signal_datasets import DOADataset, create_dataloader
from models.ca_doa_net import CA_DOA_Net, CA_DOA_Net_Light
from utils.trainer import Trainer
from configs.default import Config


def parse_args():
    parser = argparse.ArgumentParser(description='CA-DOA-Net 训练脚本')

    # 阵列参数
    parser.add_argument('--M', type=int, default=8, help='阵元数量')
    parser.add_argument('--k', type=int, default=3, help='信源数量')

    # 信号参数
    parser.add_argument('--snr_min', type=float, default=-10, help='最小SNR (dB)')
    parser.add_argument('--snr_max', type=float, default=10, help='最大SNR (dB)')
    parser.add_argument('--snap', type=int, default=200, help='快拍数')
    parser.add_argument('--rho', type=float, default=0.0, help='阵列误差程度')

    # 数据集参数
    parser.add_argument('--train_samples', type=int, default=20000, help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=4000, help='验证样本数')
    parser.add_argument('--min_sep', type=float, default=5.0, help='最小角度间隔')
    parser.add_argument('--low_snr_oversample', type=float, default=0.0,
                        help='低SNR过采样比例，如0.6表示60%样本来自低SNR区间')
    parser.add_argument('--low_snr_threshold', type=float, default=-10.0,
                        help='低SNR阈值，低于此值的SNR被视为低SNR')

    # 模型参数
    parser.add_argument('--model', type=str, default='full', choices=['full', 'light'],
                        help='模型类型: full=完整版, light=轻量版')
    parser.add_argument('--attention', type=str, default='coord', choices=['coord', 'dual', 'none'],
                        help='注意力类型')
    parser.add_argument('--base_channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--num_blocks', type=int, default=4, help='残差块数量')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--early_stop', type=int, default=15, help='早停耐心值')

    # 损失函数权重
    parser.add_argument('--lambda_spectrum', type=float, default=1.0, help='空间谱损失权重')
    parser.add_argument('--lambda_sparse', type=float, default=0.1, help='稀疏损失权重')
    parser.add_argument('--lambda_sep', type=float, default=0.1, help='分离损失权重')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'bce', 'kl', 'si_sdr'],
                        help='空间谱损失类型: mse(默认), bce, kl, si_sdr(推荐)')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 实验名称
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_suffix = f"_{args.loss_type}" if args.loss_type != 'mse' else ""
        attn_suffix = f"_{args.attention}" if args.attention != 'coord' else ""
        args.exp_name = f"ca_doa_M{args.M}_k{args.k}_snr{args.snr_min}to{args.snr_max}{loss_suffix}{attn_suffix}_{timestamp}"

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {save_dir}")

    # 计算网格大小
    angle_range = (-60, 60)
    grid_step = 1.0
    num_classes = int((angle_range[1] - angle_range[0]) / grid_step) + 1

    # 创建数据集
    print("\n创建数据集...")
    train_loader = create_dataloader(
        M=args.M, k=args.k,
        snr_range=(args.snr_min, args.snr_max),
        snap=args.snap,
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        min_sep=args.min_sep,
        rho=args.rho,
        seed=args.seed,
        shuffle=True,
        low_snr_oversample=args.low_snr_oversample,
        low_snr_threshold=args.low_snr_threshold
    )

    val_loader = create_dataloader(
        M=args.M, k=args.k,
        snr_range=(args.snr_min, args.snr_max),
        snap=args.snap,
        num_samples=args.val_samples,
        batch_size=args.batch_size,
        min_sep=args.min_sep,
        rho=args.rho,
        seed=args.seed + 1,
        shuffle=False,
        low_snr_oversample=args.low_snr_oversample,
        low_snr_threshold=args.low_snr_threshold
    )

    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    # 创建模型
    print("\n创建模型...")
    if args.model == 'light':
        model = CA_DOA_Net_Light(
            in_channels=4,
            M=args.M,
            num_classes=num_classes
        )
    else:
        model = CA_DOA_Net(
            in_channels=4,
            M=args.M,
            num_classes=num_classes,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            attention_type=args.attention
        )

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params / 1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(model, device=device, save_dir=str(save_dir))
    trainer.compile(
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_spectrum=args.lambda_spectrum,
        lambda_sparse=args.lambda_sparse,
        lambda_sep=args.lambda_sep,
        spectrum_loss_type=args.loss_type,
        epochs=args.epochs
    )

    # 打印损失函数配置
    print(f"\n损失函数配置:")
    print(f"  空间谱损失类型: {args.loss_type}")
    print(f"  λ_spectrum={args.lambda_spectrum}, λ_sparse={args.lambda_sparse}, λ_sep={args.lambda_sep}")

    # 保存配置
    config_path = save_dir / 'config.txt'
    with open(config_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # 开始训练
    print("\n开始训练...")
    print("=" * 50)

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        k=args.k,
        early_stopping_patience=args.early_stop
    )

    print("\n训练完成!")
    print(f"最佳验证RMSE: {min(history['val_rmse']):.4f}")
    print(f"模型已保存到: {save_dir}")


if __name__ == '__main__':
    main()
