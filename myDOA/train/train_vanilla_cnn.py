"""
Vanilla CNN 训练脚本 (基线模型)

特点:
- 2通道输入 (实部 + 虚部)
- 无注意力机制
- 无残差连接

用于与 CA-DOA-Net 进行公平对比

使用方法:
    python train_vanilla_cnn.py --device cuda --epochs 100 --snr_min -20 --snr_max 5
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from datetime import datetime

from data.signal_datasets import DOADataset, DataLoader
from models.vanilla_cnn import VanillaCNN
from utils.reproducibility import build_run_metadata, write_run_metadata
from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Vanilla CNN 训练脚本 (基线模型)')

    # 阵列参数
    parser.add_argument('--M', type=int, default=8, help='阵元数量')
    parser.add_argument('--k', type=int, default=3, help='信源数量')

    # 信号参数
    parser.add_argument('--snr_min', type=float, default=-20, help='最小SNR (dB)')
    parser.add_argument('--snr_max', type=float, default=5, help='最大SNR (dB)')
    parser.add_argument('--snap', type=int, default=200, help='快拍数')
    parser.add_argument('--rho', type=float, default=0.0, help='阵列误差程度')

    # 数据集参数
    parser.add_argument('--train_samples', type=int, default=20000, help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=4000, help='验证样本数')
    parser.add_argument('--min_sep', type=float, default=5.0, help='最小角度间隔')
    parser.add_argument('--low_snr_oversample', type=float, default=0.0,
                        help='低SNR过采样比例')
    parser.add_argument('--low_snr_threshold', type=float, default=-10.0,
                        help='低SNR阈值')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比例')
    parser.add_argument('--early_stop', type=int, default=15, help='早停耐心值')

    # 损失函数
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'bce', 'kl', 'si_sdr'],
                        help='空间谱损失类型')
    parser.add_argument('--lambda_spectrum', type=float, default=1.0, help='空间谱损失权重')
    parser.add_argument('--lambda_sparse', type=float, default=0.1, help='稀疏损失权重')
    parser.add_argument('--lambda_sep', type=float, default=0.1, help='分离损失权重')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')

    return parser.parse_args()


def create_dataloader(
    M: int,
    k: int,
    snr_range: tuple,
    snap: int,
    num_samples: int,
    batch_size: int,
    input_channels: int = 2,
    shuffle: bool = True,
    **kwargs
):
    """创建2通道数据集的DataLoader"""
    dataset = DOADataset(
        M=M, k=k,
        snr_range=snr_range,
        snap=snap,
        num_samples=num_samples,
        input_channels=input_channels,
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 实验名称 (参考CA-DOA-Net命名风格)
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_suffix = f"_{args.loss_type}" if args.loss_type != 'mse' else ""
        args.exp_name = f"vanilla_cnn_M{args.M}_k{args.k}_snr{args.snr_min}to{args.snr_max}{loss_suffix}_{timestamp}"

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {save_dir}")
    run_metadata = build_run_metadata(
        args=args,
        run_type='train_vanilla_cnn',
        extra={
            'model': {
                'name': 'VanillaCNN',
                'input_channels': 2,
                'attention': 'none',
                'residual': False,
            },
            'data': {
                'train_seed': args.seed,
                'val_seed': args.seed + 1,
                'snr_range': [args.snr_min, args.snr_max],
                'grid_step': 1.0,
            },
            'outputs': {
                'config_text': 'config.txt',
                'metadata': 'run_metadata.json',
                'command': 'command.txt',
                'best_checkpoint': 'best_model.pth',
                'final_checkpoint': 'final_model.pth',
                'history': 'history.json',
            },
            'checkpoint_selection': {
                'monitor': 'val_loss',
                'mode': 'min',
            },
        },
    )
    write_run_metadata(save_dir, run_metadata)

    # 计算网格大小
    angle_range = (-60, 60)
    grid_step = 1.0
    num_classes = int((angle_range[1] - angle_range[0]) / grid_step) + 1

    # 创建数据集 (2通道输入)
    print("\n创建数据集 (2通道输入: 实部+虚部)...")
    train_loader = create_dataloader(
        M=args.M, k=args.k,
        snr_range=(args.snr_min, args.snr_max),
        snap=args.snap,
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        input_channels=2,  # 2通道输入
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
        input_channels=2,  # 2通道输入
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
    print("\n创建模型: VanillaCNN (2通道输入，无注意力，无残差)")
    model = VanillaCNN(
        in_channels=2,
        M=args.M,
        num_classes=num_classes,
        dropout=args.dropout
    )

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params / 1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(model, device=device, save_dir=str(save_dir), run_metadata=run_metadata)
    trainer.compile(
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_spectrum=args.lambda_spectrum,
        lambda_sparse=args.lambda_sparse,
        lambda_sep=args.lambda_sep,
        spectrum_loss_type=args.loss_type,
        epochs=args.epochs
    )

    # 打印配置
    print(f"\n损失函数配置:")
    print(f"  空间谱损失类型: {args.loss_type}")
    print(f"  lambda_spectrum={args.lambda_spectrum}, lambda_sparse={args.lambda_sparse}, lambda_sep={args.lambda_sep}")

    # 保存配置
    config_path = save_dir / 'config.txt'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Vanilla CNN 训练配置\n")
        f.write("=" * 50 + "\n")
        f.write("模型特点: 2通道输入, 无注意力, 无残差\n")
        f.write("-" * 50 + "\n")
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
