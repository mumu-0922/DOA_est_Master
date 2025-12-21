import argparse
import os
import sys
import time

import numpy as np

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from data_creater.signal_datasets import ULA_dataset
from data_creater.Create_k_source_dataset import Create_random_k_input_theta, Create_datasets
from models.subspace_model.music import Music


def _rmse_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 计算 DOA 估计误差的 RMSE（单位：度），按每个样本内部排序后再对齐计算
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    y_true = np.sort(y_true, axis=-1)
    y_pred = np.sort(y_pred, axis=-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _try_import_matplotlib():
    # matplotlib 不是硬依赖：如果环境缺少 matplotlib，就只输出 CSV，不画图
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        return None
    return plt


def main():
    # ========= 开题“最小可行性”脚本入口 =========
    # 目标：
    # 1) 直接用本仓库的 ULA 数据生成器产生低 SNR 数据（SCM/DOA）
    # 2) 用本仓库的 MUSIC 实现跑基线，输出 RMSE vs SNR 曲线 + 一张谱图示例
    parser = argparse.ArgumentParser(description="Opening demo: MUSIC baseline under low SNR (ULA).")
    # 场景参数：
    parser.add_argument("--M", type=int, default=8)  # 阵元数（天线/传感器个数），ULA 常用 M=8/16...
    parser.add_argument("--k", type=int, default=3)  # 信号源个数（入射信号数），例如 k=3 表示 3 个 DOA
    parser.add_argument("--snrs", type=int, nargs="+", default=[-10, -5, 0])  # 要扫描的 SNR 列表（单位 dB）
    parser.add_argument("--snap", type=int, default=10)  # 快拍数 L（每个样本的采样点数/观测次数）
    parser.add_argument("--num_samples", type=int, default=1500)  # 每个 SNR 下生成多少个随机样本（样本数）
    parser.add_argument("--min_delta_theta", type=float, default=3.0)  # 每个样本内 DOA 的最小角间隔（度），避免太近难分
    parser.add_argument("--rho", type=float, default=0.0)  # 阵列误差强度参数：rho=0 表示不加阵列误差（理想阵列）
    parser.add_argument("--signal_range", type=int, nargs=2, default=[-60, 60])  # DOA 角度范围 [start, end]（度）
    parser.add_argument("--grid_step", type=float, default=1.0)  # MUSIC 搜索栅格步长（度），越小越精但计算更慢
    parser.add_argument("--seed", type=int, default=0)  # 随机种子：控制随机 DOA/信号/噪声，保证可复现
    # 可选：把生成的数据落盘为 .npz，后续可直接用 test/ 里的评测脚本复用
    parser.add_argument("--save_npz", action="store_true", help="Save generated datasets to data/ as .npz.")  # 是否保存数据集到 data/
    args = parser.parse_args()

    # 固定随机种子（保证开题时可复现）
    np.random.seed(args.seed)

    root_path = ROOT_PATH
    # 输出目录：results/opening/...（自动加时间戳防止覆盖）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_root = os.path.join(root_path, "results", "opening", f"music_only_M{args.M}_k{args.k}_rho{args.rho}_{timestamp}")
    os.makedirs(save_root, exist_ok=True)

    if args.save_npz:
        # 数据保存目录：data/ULA_data/test/...（与仓库其他脚本的习惯一致）
        data_root = os.path.join(root_path, "data", "ULA_data", "test", f"M_{args.M}_k_{args.k}_opening_rho{args.rho}")
        os.makedirs(data_root, exist_ok=True)
    else:
        data_root = None

    start_angle, end_angle = args.signal_range
    # 生成随机 DOA 样本（每个样本 k 个角度，并控制最小角间隔 min_delta_theta）
    theta_set = Create_random_k_input_theta(
        args.k,
        start_angle=start_angle,
        end_angle=end_angle,
        theta_num=args.num_samples,
        min_delta_theta=args.min_delta_theta,
    )

    rmse_by_snr = []
    succ_by_snr = []

    # 保存一个示例谱图（用于 PPT：展示低 SNR 下谱峰“乱/塌陷”）
    example_spectrum = {}

    for snr_db in args.snrs:
        # 1) 初始化数据集对象：内部包含阵列模型、协方差矩阵计算、增强特征等
        dataset = ULA_dataset(args.M, start_angle, end_angle, args.grid_step, args.rho)

        # 2) 生成该 SNR 下的数据：y(t)、样本协方差 ori_scm、训练特征 scm/enhance_scm、真值 doa 等
        batch_size = min(256, args.num_samples)
        Create_datasets(
            dataset,
            args.k,
            theta_set,
            batch_size=batch_size,
            snap=args.snap,
            snr=snr_db,
            snr_set=0,
        )

        if data_root is not None:
            # 需要落盘时保存为 npz（注意：save_all_data 内部会自动加 .npz 压缩保存）
            dataset.save_all_data(os.path.join(data_root, f"test_random_input_snr_{snr_db}"))

        # 3) 初始化 MUSIC：用 ULA_dataset.get_A 生成导向矢量；search grid = [start_angle, end_angle]，步长 grid_step
        music = Music(dataset.get_A, start=start_angle, end=end_angle, step=args.grid_step)

        doa_hat = np.zeros((args.num_samples, args.k), dtype=np.float32)
        succ_idx = np.zeros(args.num_samples, dtype=bool)

        # 4) 对每个样本做 MUSIC：输入为样本协方差矩阵 R（dataset.ori_scm）
        sp_save = None
        sp_true = None
        sp_pred = None
        for i, R in enumerate(dataset.ori_scm):
            succ, doa, sp = music.estimate(R, args.k, return_sp=True)
            doa_hat[i] = doa
            succ_idx[i] = succ
            if sp_save is None and succ:
                sp_save = sp
                sp_true = np.array(dataset.doa[i], dtype=np.float32)
                sp_pred = np.array(doa, dtype=np.float32)

        # 5) 统计指标：
        # success_ratio：算法是否“成功返回 k 个峰”（不代表估计正确）
        # rmse_deg：在 succ=True 的样本子集上统计 DOA 估计误差（单位：度）
        succ_ratio = float(np.mean(succ_idx))
        rmse = _rmse_deg(np.array(dataset.doa)[succ_idx], doa_hat[succ_idx])

        rmse_by_snr.append(rmse)
        succ_by_snr.append(succ_ratio)

        if sp_save is not None:
            example_spectrum[int(snr_db)] = {
                "spectrum": sp_save,
                "grid": np.arange(start_angle, end_angle + 0.001, args.grid_step),
                "true": sp_true,
                "pred": sp_pred,
            }

        print(f"SNR={snr_db:>4} dB | RMSE={rmse:.3f} deg | succ={succ_ratio:.3f} | saved={save_root}")

    # 6) 保存 CSV：用于开题报告表格/画图（snr_db, rmse_deg, success_ratio）
    csv_path = os.path.join(save_root, "music_rmse.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("snr_db,rmse_deg,success_ratio\n")
        for snr_db, rmse, succ in zip(args.snrs, rmse_by_snr, succ_by_snr):
            f.write(f"{snr_db},{rmse},{succ}\n")

    plt = _try_import_matplotlib()
    if plt is None:
        print(f"Saved CSV only: {csv_path}")
        return

    # 7) 画 RMSE vs SNR 曲线（开题最核心的“传统算法低 SNR 崩塌”证据图）
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.plot(args.snrs, rmse_by_snr, marker="o", linewidth=2, label="MUSIC")
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (deg)")
    plt.title(f"MUSIC RMSE vs SNR (M={args.M}, k={args.k}, snap={args.snap})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_root, "music_rmse.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # 8) 画一张示例谱图：默认取 snr=-10 dB（如果你扫的 SNR 里包含 -10）
    if -10 in example_spectrum:
        ex = example_spectrum[-10]
        fig = plt.figure(figsize=(7.0, 4.0))
        plt.plot(ex["grid"], ex["spectrum"], linewidth=1.5)
        for t in ex["true"]:
            plt.axvline(float(t), color="g", linestyle="--", linewidth=1, label="true" if t == ex["true"][0] else None)
        for p in ex["pred"]:
            plt.axvline(float(p), color="r", linestyle=":", linewidth=1, label="pred" if p == ex["pred"][0] else None)
        plt.xlabel("Angle (deg)")
        plt.ylabel("MUSIC Spectrum")
        plt.title("Example MUSIC Spectrum (SNR=-10 dB)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        sp_path = os.path.join(save_root, "music_spectrum_snr_-10.png")
        plt.savefig(sp_path, dpi=200)
        plt.close(fig)

    print(f"Done. Outputs in: {save_root}")


if __name__ == "__main__":
    main()
