"""
DOA估计方法对比测试脚本

对比方法：
1. MUSIC - 传统子空间方法
2. TLS-ESPRIT - 总体最小二乘旋转不变子空间方法
3. Vanilla CNN - 真正的基础CNN (2通道输入，无注意力，无残差)
4. CA-DOA-Net - 带坐标注意力的CNN（本文方法，4通道输入）

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
from models.vanilla_cnn import VanillaCNN

# 传统方法（myDOA内部实现）
from models.subspace_methods import Music, TLS_ESPRIT
from utils.doa_evaluation import evaluate_doa_sample
from utils.peak_search import (
    min_sep_to_bins,
    refine_peak_centroid,
    refine_peak_parabolic,
    select_peaks_with_suppression,
)
from utils.reproducibility import save_run_metadata


def compute_rmse_with_hungarian(doa_est: np.ndarray, doa_true: np.ndarray, tol: float = 2.0):
    """
    使用匈牙利算法计算最优匹配的RMSE和成功率

    Args:
        doa_est: 估计的DOA角度 (k,)
        doa_true: 真实的DOA角度 (k,)
        tol: 成功判定容差（度），所有源误差都小于tol才算成功

    Returns:
        rmse: 最优匹配后的RMSE
        success: 是否成功（所有源误差都小于tol）
    """
    result = evaluate_doa_sample(doa_est, doa_true, tol=tol)
    return result['rmse'], result['success']


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
    parser.add_argument('--tol', type=float, default=2.0, help='成功率判定容差（度），所有源误差<tol才算成功')
    parser.add_argument('--grid_step', type=float, default=1.0, help='CA-DOA-Net的网格步长')
    parser.add_argument('--refinement', type=str, default='none',
                        choices=['none', 'parabolic', 'centroid'],
                        help='深度模型谱峰细化方式；正式评测默认none，与test_snr.py口径一致')

    # 模型路径
    parser.add_argument('--ca_doa_weights', type=str,
                        default='../train/results/ca_doa_M8_k3_snr-20.0to5.0_si_sdr_20260102_153000/best_model.pth',
                        help='CA-DOA-Net权重路径 (4通道输入)')
    parser.add_argument('--vanilla_cnn_weights', type=str,
                        default='../train/results/vanilla_cnn_M8_k3_snr-20.0to5.0_bce_20260102_003831/best_model.pth',
                        help='Vanilla CNN权重路径 (2通道输入)')

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
    two_channel_list = []  # 新增：2通道数据 (实部+虚部)

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

        # 归一化协方差矩阵（关键修复！）
        scm_norm = scm / (np.trace(scm).real + 1e-10)

        # 四通道输入 (CA-DOA-Net用) - 使用归一化后的SCM
        real_part = scm_norm.real
        imag_part = scm_norm.imag
        phase_angle = np.angle(scm_norm)
        sin_phase = np.sin(phase_angle)
        cos_phase = np.cos(phase_angle)
        four_channel = np.stack([real_part, imag_part, sin_phase, cos_phase], axis=0)

        # 二通道输入 (Vanilla CNN用) - 使用归一化后的SCM
        two_channel = np.stack([real_part, imag_part], axis=0)

        scm_list.append(scm)  # 传统方法用原始SCM
        doa_list.append(doa)
        four_channel_list.append(four_channel)
        two_channel_list.append(two_channel)

    return {
        'scm': np.array(scm_list),
        'doa': np.array(doa_list),
        'four_channel': np.array(four_channel_list).astype(np.float32),
        'two_channel': np.array(two_channel_list).astype(np.float32)
    }


def test_music(music_estimator, scm_list, doa_true, k, tol=2.0):
    """测试MUSIC算法"""
    rmse_list = []
    success_count = 0

    for i, scm in enumerate(scm_list):
        success_est, doa_est = music_estimator.estimate(scm, k)
        if success_est:
            doa_est = np.sort(doa_est)
            # 使用匈牙利算法计算RMSE和严格成功率
            rmse, success = compute_rmse_with_hungarian(doa_est, doa_true[i], tol)
            rmse_list.append(rmse)
            if success:
                success_count += 1
        else:
            rmse_list.append(np.nan)

    valid_rmse = [r for r in rmse_list if not np.isnan(r)]
    mean_rmse = np.mean(valid_rmse) if valid_rmse else np.nan
    success_rate = success_count / len(scm_list)

    return mean_rmse, success_rate


def test_tls_esprit(esprit_estimator, scm_list, doa_true, k, tol=2.0):
    """测试TLS-ESPRIT算法"""
    rmse_list = []
    success_count = 0

    for i, scm in enumerate(scm_list):
        try:
            success_est, doa_est = esprit_estimator.estimate(scm, k)
            if success_est and not np.any(np.isnan(doa_est)):
                doa_est = np.sort(doa_est.real)
                # 过滤超出范围的估计
                if np.all(np.abs(doa_est) <= 90):
                    # 使用匈牙利算法计算RMSE和严格成功率
                    rmse, success = compute_rmse_with_hungarian(doa_est, doa_true[i], tol)
                    rmse_list.append(rmse)
                    if success:
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


def test_dl_model(model, four_channel, doa_true, k, device, grid_step=1.0, tol=2.0, refinement='parabolic', min_sep=3.0):
    """
    测试深度学习模型

    Args:
        refinement: 峰值细化方法
            - 'parabolic': 三点抛物线插值（推荐，更精确）
            - 'centroid': 加权质心法
            - 'none': 不细化，直接取网格点
        min_sep: 最小角度间隔（度），用于计算抑制窗口
    """
    model.eval()

    # 计算抑制窗口大小
    min_sep_bins = min_sep_to_bins(min_sep, grid_step)

    with torch.no_grad():
        inputs = torch.tensor(four_channel).to(device)
        outputs = model(inputs)

        # 模型直接输出空间谱 (B, num_classes)
        spectrum = outputs.cpu().numpy()
        grid = np.arange(-60, 60 + 0.001, grid_step)

        rmse_list = []
        success_count = 0
        for i in range(len(spectrum)):
            # 使用 Top-K + 抑制窗口策略提取峰值
            top_peaks = select_peaks_with_suppression(spectrum[i], k, min_sep_bins)

            # 根据细化方法提取DOA
            if refinement == 'parabolic':
                doa_est = np.array([refine_peak_parabolic(spectrum[i], grid, p) for p in top_peaks])
            elif refinement == 'centroid':
                doa_est = np.array([refine_peak_centroid(spectrum[i], grid, p) for p in top_peaks])
            else:  # 'none'
                doa_est = grid[top_peaks]

            # 使用匈牙利算法计算RMSE和严格成功率
            rmse, success = compute_rmse_with_hungarian(doa_est, doa_true[i], tol)
            rmse_list.append(rmse)
            if success:
                success_count += 1

    mean_rmse = np.mean(rmse_list)
    success_rate = success_count / len(spectrum)

    return mean_rmse, success_rate


def plot_comparison(results_df, save_path):
    """绘制对比曲线图"""
    plt.figure(figsize=(10, 6))

    methods = ['MUSIC', 'TLS-ESPRIT', 'Vanilla CNN', 'CA-DOA-Net']
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


def build_result_record(snr, args, music, esprit, vanilla, ca_doa):
    """Build one CSV result row with the evaluation protocol attached."""
    music_rmse, music_sr = music
    esprit_rmse, esprit_sr = esprit
    vanilla_rmse, vanilla_sr = vanilla
    ca_doa_rmse, ca_doa_sr = ca_doa

    return {
        'snr': snr,
        'refinement': args.refinement,
        'min_sep': args.min_sep,
        'grid_step': args.grid_step,
        'tol': args.tol,
        'MUSIC_rmse': music_rmse,
        'MUSIC_sr': music_sr,
        'TLS-ESPRIT_rmse': esprit_rmse,
        'TLS-ESPRIT_sr': esprit_sr,
        'Vanilla CNN_rmse': vanilla_rmse,
        'Vanilla CNN_sr': vanilla_sr,
        'CA-DOA-Net_rmse': ca_doa_rmse,
        'CA-DOA-Net_sr': ca_doa_sr
    }


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
    save_run_metadata(
        save_dir=save_dir,
        args=args,
        run_type='compare_methods',
        extra={
            'weights': {
                'ca_doa': args.ca_doa_weights,
                'vanilla_cnn': args.vanilla_cnn_weights,
            },
            'outputs': {
                'csv': 'comparison_results.csv',
                'figure': 'comparison_plot.png',
            },
            'evaluation': {
                'refinement': args.refinement,
                'min_sep': args.min_sep,
                'grid_step': args.grid_step,
                'tol': args.tol,
            },
        },
    )

    # 初始化传统方法
    print("\n初始化算法...")
    music_estimator = Music(M=args.M, d=0.5, start=-60, end=60, step=0.1)
    esprit_estimator = TLS_ESPRIT(M=args.M, d=0.5, displacement=1)

    # 加载深度学习模型
    # Vanilla CNN 固定使用 1° 网格
    vanilla_num_classes = 121
    # CA-DOA-Net 使用指定的网格步长
    ca_doa_num_classes = int(120 / args.grid_step) + 1
    print(f"CA-DOA-Net 网格步长: {args.grid_step}°, 输出维度: {ca_doa_num_classes}")
    print(f"深度模型谱峰细化方式: {args.refinement}")

    # CA-DOA-Net（有注意力）
    print(f"加载CA-DOA-Net: {args.ca_doa_weights}")
    ca_doa_model = CA_DOA_Net(in_channels=4, M=args.M, num_classes=ca_doa_num_classes,
                               attention_type='coord', step=args.grid_step)
    ca_doa_ckpt = torch.load(args.ca_doa_weights, map_location=device)
    ca_doa_model.load_state_dict(ca_doa_ckpt['model_state_dict'])
    ca_doa_model = ca_doa_model.to(device)
    ca_doa_model.eval()

    # Vanilla CNN（真正的基线CNN，2通道输入，无注意力，无残差）
    print(f"加载Vanilla CNN: {args.vanilla_cnn_weights}")
    vanilla_cnn_model = VanillaCNN(in_channels=2, M=args.M, num_classes=vanilla_num_classes)
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
        music_rmse, music_sr = test_music(music_estimator, test_data['scm'], test_data['doa'], args.k, args.tol)
        esprit_rmse, esprit_sr = test_tls_esprit(esprit_estimator, test_data['scm'], test_data['doa'], args.k, args.tol)
        vanilla_rmse, vanilla_sr = test_dl_model(vanilla_cnn_model, test_data['two_channel'], test_data['doa'], args.k, device, grid_step=1.0, tol=args.tol, refinement=args.refinement, min_sep=args.min_sep)
        ca_doa_rmse, ca_doa_sr = test_dl_model(ca_doa_model, test_data['four_channel'], test_data['doa'], args.k, device, grid_step=args.grid_step, tol=args.tol, refinement=args.refinement, min_sep=args.min_sep)

        result = build_result_record(
            snr=snr,
            args=args,
            music=(music_rmse, music_sr),
            esprit=(esprit_rmse, esprit_sr),
            vanilla=(vanilla_rmse, vanilla_sr),
            ca_doa=(ca_doa_rmse, ca_doa_sr),
        )

        results.append(result)

        print(f"\nSNR={snr:4.0f}dB:")
        print(f"  MUSIC:       RMSE={music_rmse:6.2f}°, SR={music_sr:.1%}")
        print(f"  TLS-ESPRIT:  RMSE={esprit_rmse:6.2f}°, SR={esprit_sr:.1%}")
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
    print("\n" + "=" * 70)
    print("汇总结果 (RMSE °)")
    print("=" * 70)
    print(f"{'SNR':>6} | {'MUSIC':>8} | {'TLS-ESPRIT':>10} | {'V-CNN':>8} | {'CA-DOA':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['snr']:>6.0f} | {row['MUSIC_rmse']:>8.2f} | {row['TLS-ESPRIT_rmse']:>10.2f} | {row['Vanilla CNN_rmse']:>8.2f} | {row['CA-DOA-Net_rmse']:>8.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
