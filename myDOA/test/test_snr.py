"""
SNR扫描测试脚本

测试模型在不同SNR下的性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from data.signal_datasets import DOADataset
from models.ca_doa_net import CA_DOA_Net, CA_DOA_Net_Light
from utils.metrics import compute_rmse, compute_success_rate


def parse_args():
    parser = argparse.ArgumentParser(description='SNR扫描测试')

    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--M', type=int, default=8, help='阵元数量')
    parser.add_argument('--k', type=int, default=3, help='信源数量')
    parser.add_argument('--snap', type=int, default=200, help='快拍数')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-20, -15, -10, -5, 0, 5, 10],
                        help='测试SNR列表')
    parser.add_argument('--num_samples', type=int, default=1000, help='每个SNR的测试样本数')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'light'])
    parser.add_argument('--base_channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--num_blocks', type=int, default=4, help='残差块数量')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存根目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称（默认从权重路径提取）')

    return parser.parse_args()


def test_single_snr(model, snr, args, device):
    """测试单个SNR下的性能"""

    # 创建测试数据集
    dataset = DOADataset(
        M=args.M,
        k=args.k,
        snr_range=(snr, snr),  # 固定SNR
        snap=args.snap,
        num_samples=args.num_samples,
        seed=42
    )

    rmse_list = []
    success_list = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            inputs = sample['input'].unsqueeze(0).to(device)
            target_doa = sample['doa'].to(device)

            # 预测
            spectrum = model(inputs)
            success, pred_doa = model.spectrum_to_doa(spectrum, args.k, min_sep=5.0)

            success_list.append(success.item())

            if success.item():
                rmse = torch.sqrt(torch.mean((pred_doa.squeeze() - target_doa) ** 2)).item()
                rmse_list.append(rmse)

    success_rate = np.mean(success_list)
    mean_rmse = np.mean(rmse_list) if rmse_list else float('nan')

    return {
        'snr': snr,
        'rmse': mean_rmse,
        'success_rate': success_rate,
        'num_success': len(rmse_list),
        'num_total': len(dataset)
    }


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 生成实验名称（从权重路径提取或自动生成）
    if args.exp_name is None:
        # 尝试从权重路径提取训练实验名称
        weights_path = Path(args.weights)
        train_exp_name = weights_path.parent.name  # 如 ca_doa_M8_k3_snr-15.0to5.0_si_sdr_20251230_200328

        # 生成测试实验名称：test_{训练实验名}_{时间戳}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snr_range = f"snr{min(args.snr_list):.0f}to{max(args.snr_list):.0f}"
        args.exp_name = f"test_{train_exp_name}_{snr_range}_{timestamp}"

    # 创建保存目录
    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("加载模型...")
    num_classes = 121  # -60到60，步长1

    if args.model == 'light':
        model = CA_DOA_Net_Light(in_channels=4, M=args.M, num_classes=num_classes)
    else:
        model = CA_DOA_Net(
            in_channels=4,
            M=args.M,
            num_classes=num_classes,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks
        )

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型加载完成: {args.weights}")
    print(f"结果保存目录: {save_dir}")

    # 测试每个SNR
    results = []
    print("\n开始SNR扫描测试...")

    for snr in tqdm(args.snr_list, desc='SNR扫描'):
        result = test_single_snr(model, snr, args, device)
        results.append(result)
        print(f"SNR={snr:4.0f}dB: RMSE={result['rmse']:.4f}, Success={result['success_rate']:.2%}")

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = save_dir / 'snr_test_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE曲线
    axes[0].plot(df['snr'], df['rmse'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('RMSE (°)', fontsize=12)
    axes[0].set_title('RMSE vs SNR', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 成功率曲线
    axes[1].plot(df['snr'], df['success_rate'] * 100, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[1].set_title('Success Rate vs SNR', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    fig_path = save_dir / 'snr_test_results.png'
    plt.savefig(fig_path, dpi=150)
    print(f"图表已保存: {fig_path}")

    plt.show()


if __name__ == '__main__':
    main()
