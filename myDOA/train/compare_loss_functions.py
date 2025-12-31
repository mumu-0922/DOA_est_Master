"""
损失函数对比实验脚本

对比 MSE 损失与 SI-SDR 损失的训练效果

基于论文: Chen & Rao, "A Comparative Study of Invariance-Aware Loss
Functions for Deep Learning-based Gridless DoA Estimation", ICASSP 2025

使用方法:
    python compare_loss_functions.py --device cuda --epochs 50
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime

from data.signal_datasets import create_dataloader
from models.ca_doa_net import CA_DOA_Net
from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='损失函数对比实验')

    # 阵列参数
    parser.add_argument('--M', type=int, default=8, help='阵元数量')
    parser.add_argument('--k', type=int, default=3, help='信源数量')

    # 信号参数
    parser.add_argument('--snr_min', type=float, default=-10, help='最小SNR')
    parser.add_argument('--snr_max', type=float, default=10, help='最大SNR')
    parser.add_argument('--snap', type=int, default=200, help='快拍数')

    # 数据集参数
    parser.add_argument('--train_samples', type=int, default=20000, help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=4000, help='验证样本数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--early_stop', type=int, default=15, help='早停耐心值')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results/compare', help='保存目录')

    return parser.parse_args()


def train_with_loss_type(args, loss_type: str, lr: float, save_dir: Path):
    """使用指定损失函数类型训练模型"""

    print(f"\n{'='*60}")
    print(f"训练配置: loss_type={loss_type}, lr={lr}")
    print(f"{'='*60}")

    # 设置随机种子（确保公平对比）
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # 计算网格大小
    num_classes = 121  # -60 到 60，步长 1

    # 创建数据集
    train_loader = create_dataloader(
        M=args.M, k=args.k,
        snr_range=(args.snr_min, args.snr_max),
        snap=args.snap,
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        min_sep=5.0,
        seed=args.seed,
        shuffle=True
    )

    val_loader = create_dataloader(
        M=args.M, k=args.k,
        snr_range=(args.snr_min, args.snr_max),
        snap=args.snap,
        num_samples=args.val_samples,
        batch_size=args.batch_size,
        min_sep=5.0,
        seed=args.seed + 1,
        shuffle=False
    )

    # 创建模型
    model = CA_DOA_Net(
        in_channels=4,
        M=args.M,
        num_classes=num_classes,
        base_channels=64,
        num_blocks=4,
        attention_type='coord'
    )

    # 创建训练器
    exp_save_dir = save_dir / loss_type
    exp_save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, device=device, save_dir=str(exp_save_dir))
    trainer.compile(
        lr=lr,
        weight_decay=1e-4,
        lambda_spectrum=1.0,
        lambda_sparse=0.1,
        lambda_sep=0.1,
        spectrum_loss_type=loss_type,
        epochs=args.epochs
    )

    # 训练
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        k=args.k,
        early_stopping_patience=args.early_stop
    )

    # 返回结果
    best_rmse = min(history['val_rmse'])
    final_rmse = history['val_rmse'][-1]

    return {
        'loss_type': loss_type,
        'lr': lr,
        'best_rmse': best_rmse,
        'final_rmse': final_rmse,
        'epochs_trained': len(history['val_rmse']),
        'history': history
    }


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"compare_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("损失函数对比实验")
    print("基于 Chen & Rao, ICASSP 2025")
    print("=" * 60)
    print(f"\n实验配置:")
    print(f"  M={args.M}, k={args.k}")
    print(f"  SNR: {args.snr_min} ~ {args.snr_max} dB")
    print(f"  训练样本: {args.train_samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  保存目录: {save_dir}")

    # 实验配置
    # 根据论文，SI-SDR 需要更高的学习率（约 5 倍）
    experiments = [
        ('mse', 1e-4),      # 基线：MSE + 标准学习率
        ('si_sdr', 5e-4),   # SI-SDR + 5倍学习率（论文推荐）
        ('si_sdr', 1e-3),   # SI-SDR + 10倍学习率（可选尝试）
    ]

    results = []

    for loss_type, lr in experiments:
        try:
            result = train_with_loss_type(args, loss_type, lr, save_dir)
            results.append(result)

            print(f"\n[{loss_type}, lr={lr}] 完成!")
            print(f"  最佳 RMSE: {result['best_rmse']:.4f}°")
            print(f"  最终 RMSE: {result['final_rmse']:.4f}°")

        except Exception as e:
            print(f"\n[{loss_type}, lr={lr}] 训练失败: {e}")
            results.append({
                'loss_type': loss_type,
                'lr': lr,
                'error': str(e)
            })

    # 保存对比结果
    print("\n" + "=" * 60)
    print("对比结果汇总")
    print("=" * 60)

    summary = []
    for r in results:
        if 'error' not in r:
            print(f"  {r['loss_type']:8s} (lr={r['lr']:.0e}): "
                  f"Best RMSE = {r['best_rmse']:6.2f}°, "
                  f"Final RMSE = {r['final_rmse']:6.2f}°")
            summary.append({
                'loss_type': r['loss_type'],
                'lr': r['lr'],
                'best_rmse': r['best_rmse'],
                'final_rmse': r['final_rmse'],
                'epochs_trained': r['epochs_trained']
            })

    # 保存汇总
    summary_path = save_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': summary
        }, f, indent=2)

    print(f"\n结果已保存到: {save_dir}")

    # 找出最佳配置
    if summary:
        best = min(summary, key=lambda x: x['best_rmse'])
        print(f"\n推荐配置: loss_type={best['loss_type']}, lr={best['lr']:.0e}")
        print(f"最佳 RMSE: {best['best_rmse']:.4f}°")


if __name__ == '__main__':
    main()
