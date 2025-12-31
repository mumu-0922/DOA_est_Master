"""
DOA估计方法对比测试脚本

对比方法：
1. MUSIC - 传统子空间方法
2. ESPRIT - 传统子空间方法
3. Vanilla CNN - 无注意力机制的CNN
4. CA-DOA-Net - 带坐标注意力的CNN（本文方法）

输出：
- CSV结果文件
- RMSE vs SNR 对比曲线图
"""

import sys
import os

# 添加 myDOA 目录到路径
myDOA_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, myDOA_root)

import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 深度学习模型
from models.ca_doa_net import CA_DOA_Net

# 传统方法（myDOA内部实现）
from models.subspace_methods import Music, ESPRIT


def parse_args():
    parser = argparse.ArgumentParser(description='DOA方法对比测试')

    # 阵列参数
    parser.add_argument('--M', type=int, default=8, help='阵元数量')
    parser.add_argument('--k', type=int, default=3, help='信源数量')
    parser.add_argument('--snap', type=int, default=200, help='快拍数')

    # 测试参数
    parser.add_argument('--snr_list', type=float, nargs='+',
                        default=[-15, -10, -5, 0, 5],
                        help='测试SNR列表')
    parser.add_argument('--num_samples', type=int, default=500, help='每个SNR的测试样本数')
    parser.add_argument('--min_sep', type=float, default=5.0, help='最小角度间隔')

    # 模型路径
    parser.add_argument('--ca_doa_weights', type=str,
                        default='../train/results/ca_doa_M8_k3_snr-20.0to5.0_si_sdr_20251230_220617/best_model.pth',
                        help='CA-DOA-Net权重路径')
    parser.add_argument('--vanilla_cnn_weights', type=str,
                        default='../train/results/ca_doa_M8_k3_snr-20.0to5.0_si_sdr_none20251230_222743/best_model.pth',
                        help='Vanilla CNN权重路径')

    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./results/compare')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def generate_test_data(M, k, snr_db, snap, num_samples, min_sep, seed):
    """生成测试数据"""
    np.random.seed(seed)

    d = 0.5
    array_pos = np.arange(M) * d
    angle_range = (-60, 60)

    scm_list = []
    doa_list = []
    four_channel_list = []

    for _ in range(num_samples):
        # 生成DOA角度
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            doa = np.random.uniform(angle_range[0], angle_range[1], k)
            doa = np.sort(doa)
            if k == 1:
                valid = True
            else:
                diffs = np.diff(doa)
                valid = np.all(diffs >= min_sep)
            attempts += 1

        if not valid:
            doa = np.linspace(angle_range[0] + 10, angle_range[1] - 10, k)

        # 导向矢量
        theta_rad = np.deg2rad(doa)
        phase = -1j * 2 * np.pi * np.outer(array_pos, np.sin(theta_rad))
        A = np.exp(phase)

        # 信号
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.sqrt(snr_linear / 2)
        s = signal_power * (np.random.randn(k, snap) + 1j * np.random.randn(k, snap))

        # 噪声
        n = (np.random.randn(M, snap) + 1j * np.random.randn(M, snap)) / np.sqrt(2)

        # 接收信号
        x = A @ s + n

        # 协方差矩阵
        scm = (x @ x.conj().T) / snap

        # 四通道输入
        real_part = scm.real
        imag_part = scm.imag
        phase_angle = np.angle(scm)
        sin_phase = np.sin(phase_angle)
        cos_phase = np.cos(phase_angle)
        four_channel = np.stack([real_part, imag_part, sin_phase, cos_phase], axis=0)

        scm_list.append(scm)
        doa_list.append(doa)
        four_channel_list.append(four_channel)

    return {
        'scm': np.array(scm_list),
        'doa': np.array(doa_list),
        'four_channel': np.array(four_channel_list).astype(np.float32)
    }


def test_music(music_estimator, scm_list, doa_true, k):
    """测试MUSIC算法"""
    rmse_list = []
    success_count = 0

    for i, scm in enumerate(scm_list):
        success, doa_est = music_estimator.estimate(scm, k)
        if success:
            doa_est = np.sort(doa_est)
            # 直接计算RMSE
            mse = np.mean((doa_est - doa_true[i]) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
            success_count += 1
        else:
            rmse_list.append(np.nan)

    valid_rmse = [r for r in rmse_list if not np.isnan(r)]
    mean_rmse = np.mean(valid_rmse) if valid_rmse else np.nan
    success_rate = success_count / len(scm_list)

    return mean_rmse, success_rate


def test_esprit(esprit_estimator, scm_list, doa_true, k):
    """测试ESPRIT算法"""
    rmse_list = []
    success_count = 0

    for i, scm in enumerate(scm_list):
        try:
            success, doa_est = esprit_estimator.estimate(scm, k)
            if success and not np.any(np.isnan(doa_est)):
                doa_est = np.sort(doa_est.real)
                # 过滤超出范围的估计
                if np.all(np.abs(doa_est) <= 90):
                    # 直接计算RMSE
                    mse = np.mean((doa_est - doa_true[i]) ** 2)
                    rmse = np.sqrt(mse)
                    rmse_list.append(rmse)
                    success_count += 1
                else:
                    rmse_list.append(np.nan)
            else:
                rmse_list.append(np.nan)
        except:
            rmse_list.append(np.nan)

    valid_rmse = [r for r in rmse_list if not np.isnan(r)]
    mean_rmse = np.mean(valid_rmse) if valid_rmse else np.nan
    success_rate = success_count / len(scm_list)

    return mean_rmse, success_rate


def test_dl_model(model, four_channel, doa_true, k, device):
    """测试深度学习模型"""
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(four_channel).to(device)
        outputs = model(inputs)

        # 模型直接输出空间谱 (B, num_classes)
        spectrum = outputs.cpu().numpy()
        grid = np.arange(-60, 61, 1.0)

        rmse_list = []
        for i in range(len(spectrum)):
            # 找峰值
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(spectrum[i])

            if len(peaks) >= k:
                peak_values = spectrum[i][peaks]
                top_indices = np.argsort(peak_values)[-k:]
                top_peaks = sorted([peaks[j] for j in top_indices])
                doa_est = grid[top_peaks]
            else:
                # 取最大的几个点
                top_indices = np.argsort(spectrum[i])[-k:]
                doa_est = np.sort(grid[top_indices])

            # 直接计算RMSE
            mse = np.mean((doa_est - doa_true[i]) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)

    mean_rmse = np.mean(rmse_list)
    success_rate = 1.0  # 深度学习模型总是输出结果

    return mean_rmse, success_rate


def plot_comparison(results_df, save_path):
    """绘制对比曲线图"""
    plt.figure(figsize=(10, 6))

    methods = ['MUSIC', 'ESPRIT', 'Vanilla CNN', 'CA-DOA-Net']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for method, color, marker in zip(methods, colors, markers):
        col_name = f'{method}_rmse'
        if col_name in results_df.columns:
            plt.plot(results_df['snr'], results_df[col_name],
                    color=color, marker=marker, markersize=8,
                    linewidth=2, label=method)

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE (°)', fontsize=12)
    plt.title('DOA Estimation Performance Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(results_df['snr'])

    # 设置y轴范围
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


def main():
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"compare_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化传统方法
    print("\n初始化算法...")
    music_estimator = Music(M=args.M, d=0.5, start=-60, end=60, step=0.1)
    esprit_estimator = ESPRIT(M=args.M, d=0.5, displacement=1)

    # 加载深度学习模型
    num_classes = 121  # -60到60，步长1

    # CA-DOA-Net（有注意力）
    print(f"加载CA-DOA-Net: {args.ca_doa_weights}")
    ca_doa_model = CA_DOA_Net(in_channels=4, M=args.M, num_classes=num_classes, attention_type='coord')
    ca_doa_ckpt = torch.load(args.ca_doa_weights, map_location=device)
    ca_doa_model.load_state_dict(ca_doa_ckpt['model_state_dict'])
    ca_doa_model = ca_doa_model.to(device)
    ca_doa_model.eval()

    # Vanilla CNN（无注意力）
    print(f"加载Vanilla CNN: {args.vanilla_cnn_weights}")
    vanilla_cnn_model = CA_DOA_Net(in_channels=4, M=args.M, num_classes=num_classes, attention_type='none')
    vanilla_cnn_ckpt = torch.load(args.vanilla_cnn_weights, map_location=device)
    vanilla_cnn_model.load_state_dict(vanilla_cnn_ckpt['model_state_dict'])
    vanilla_cnn_model = vanilla_cnn_model.to(device)
    vanilla_cnn_model.eval()

    # 测试每个SNR
    results = []
    print("\n开始对比测试...")

    for snr in tqdm(args.snr_list, desc='SNR扫描'):
        # 生成测试数据（所有方法使用相同数据）
        # seed 需要是正数
        data_seed = args.seed + int((snr + 100) * 100)
        test_data = generate_test_data(
            args.M, args.k, snr, args.snap,
            args.num_samples, args.min_sep, data_seed
        )

        # 测试各方法
        music_rmse, music_sr = test_music(music_estimator, test_data['scm'], test_data['doa'], args.k)
        esprit_rmse, esprit_sr = test_esprit(esprit_estimator, test_data['scm'], test_data['doa'], args.k)
        vanilla_rmse, vanilla_sr = test_dl_model(vanilla_cnn_model, test_data['four_channel'], test_data['doa'], args.k, device)
        ca_doa_rmse, ca_doa_sr = test_dl_model(ca_doa_model, test_data['four_channel'], test_data['doa'], args.k, device)

        results.append({
            'snr': snr,
            'MUSIC_rmse': music_rmse,
            'MUSIC_sr': music_sr,
            'ESPRIT_rmse': esprit_rmse,
            'ESPRIT_sr': esprit_sr,
            'Vanilla CNN_rmse': vanilla_rmse,
            'Vanilla CNN_sr': vanilla_sr,
            'CA-DOA-Net_rmse': ca_doa_rmse,
            'CA-DOA-Net_sr': ca_doa_sr
        })

        print(f"\nSNR={snr:4.0f}dB:")
        print(f"  MUSIC:       RMSE={music_rmse:6.2f}°, SR={music_sr:.1%}")
        print(f"  ESPRIT:      RMSE={esprit_rmse:6.2f}°, SR={esprit_sr:.1%}")
        print(f"  Vanilla CNN: RMSE={vanilla_rmse:6.2f}°, SR={vanilla_sr:.1%}")
        print(f"  CA-DOA-Net:  RMSE={ca_doa_rmse:6.2f}°, SR={ca_doa_sr:.1%}")

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = save_dir / 'comparison_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")

    # 绘制对比图
    plot_path = save_dir / 'comparison_plot.png'
    plot_comparison(df, plot_path)

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("汇总结果 (RMSE °)")
    print("=" * 60)
    print(f"{'SNR':>6} | {'MUSIC':>8} | {'ESPRIT':>8} | {'V-CNN':>8} | {'CA-DOA':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['snr']:>6.0f} | {row['MUSIC_rmse']:>8.2f} | {row['ESPRIT_rmse']:>8.2f} | {row['Vanilla CNN_rmse']:>8.2f} | {row['CA-DOA-Net_rmse']:>8.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
