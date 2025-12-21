import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from data_creater.Create_k_source_dataset import Create_datasets, Create_random_k_input_theta
from data_creater.signal_datasets import ULA_dataset
from models.subspace_model.esprit import ESPRIT
from models.subspace_model.music import Music


def _try_import_matplotlib():
    # matplotlib 不是硬依赖：若环境缺少 matplotlib，只输出 CSV/JSON
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        return None
    return plt


def _is_close_int(x: float) -> bool:
    return float(x).is_integer()


def _fmt_float_tag(x: float) -> str:
    # 用于路径命名：0.1 -> 0p1, 8.0 -> 8
    if _is_close_int(x):
        return str(int(x))
    return str(float(x)).replace(".", "p")


def _snr_tag(snrs: List[int]) -> str:
    # 优先生成 snr-15to10_step5 这种可读标签；否则退化成 snr_-15_-10_...
    if len(snrs) <= 1:
        return f"snr{snrs[0]}"
    diffs = np.diff(snrs)
    if np.all(diffs == diffs[0]):
        return f"snr{snrs[0]}to{snrs[-1]}_step{int(diffs[0])}"
    joined = "_".join(str(s) for s in snrs)
    return f"snr_{joined}"


def _sort_align_error_deg(theta_true: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """
    多信源 DOA 的排列不确定性：将真值与估计值各自排序后再对齐计算误差。
    返回误差矩阵 |theta_hat - theta_true|，shape=(N,k)。
    """
    theta_true = np.sort(theta_true, axis=-1)
    theta_hat = np.sort(theta_hat, axis=-1)
    return np.abs(theta_hat - theta_true)


def _por_and_cond_rmse(theta_true: np.ndarray, theta_hat: np.ndarray, tau_deg: float) -> Tuple[float, float, int]:
    """
    PoR/ACC_τ 与 Conditional RMSE（只在成功样本上统计）。
    success 判据：
      1) 必须输出 k 个有效角度（无 NaN/inf）
      2) 排序对齐后满足 max(|e_i|) < τ
    """
    theta_true = np.asarray(theta_true, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64)

    if theta_true.size == 0:
        return float("nan"), float("nan"), 0

    valid = np.isfinite(theta_hat).all(axis=1)
    err = _sort_align_error_deg(theta_true, theta_hat)
    success = valid & (np.max(err, axis=1) < float(tau_deg))
    n_success = int(np.sum(success))
    por = float(np.mean(success))

    if n_success == 0:
        return por, float("nan"), 0

    rmse = float(np.sqrt(np.mean((err[success]) ** 2)))
    return por, rmse, n_success


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_csv(path: str, rows: List[Dict], header: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")


def _plot_two_curves(
    plt,
    x: List[int],
    y1: List[float],
    y2: List[float],
    label1: str,
    label2: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
):
    fig = plt.figure(figsize=(6.6, 4.2))
    plt.plot(x, y1, marker="o", linewidth=2, label=label1)
    plt.plot(x, y2, marker="s", linewidth=2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Opening baselines: generate shared testset (SCM) and evaluate MUSIC/ESPRIT with PoR + Conditional RMSE."
    )

    # ===== 冻结主配置（默认值就是开题协议）=====
    parser.add_argument("--M", type=int, default=8, help="阵元数 M（ULA）")
    parser.add_argument("--k", type=int, default=3, help="信源数 k（多信源 DOA）")
    parser.add_argument("--snap", type=int, default=20, help="快拍数 T（少快拍主配置：20）")
    parser.add_argument("--snrs", type=int, nargs="+", default=[-15, -10, -5, 0, 5, 10], help="SNR 扫描列表（dB）")
    parser.add_argument("--num_samples", type=int, default=1000, help="每个 SNR 的测试样本数 N")
    parser.add_argument("--min_delta_theta", type=float, default=8.0, help="最小角间隔（度），避免过近导致不可分")
    parser.add_argument("--signal_range", type=int, nargs=2, default=[-60, 60], help="角度范围 [start, end]（度）")
    parser.add_argument("--rho", type=float, default=0.0, help="阵列误差强度 rho（开题默认 0）")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子（保证三种方法同一批数据）")

    # ===== 指标定义 =====
    parser.add_argument("--tau", type=float, default=2.0, help="PoR/ACC 阈值 τ（度），默认 2°")

    # ===== 算法设置 =====
    parser.add_argument("--music_grid_step", type=float, default=0.1, help="MUSIC 搜索网格步长（度），开题建议 0.1°")
    parser.add_argument(
        "--dataset_grid_step",
        type=float,
        default=1.0,
        help="数据集内部网格步长（用于生成 spatial_sp 等标签；与 MUSIC 搜索网格无关，默认 1° 可明显省时省盘）",
    )
    parser.add_argument("--snr_set", type=int, default=1, choices=[0, 1], help="SNR 定义方式：0 信号功率=1；1 噪声功率=1")

    # ===== 落盘路径（协议即路径）=====
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="测试集保存根目录（空则使用 data/ULA_data/opening_snap20_k3/...）",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="",
        help="结果保存根目录（空则使用 results/opening_snap20_k3/...）",
    )
    parser.add_argument(
        "--no_save_npz",
        action="store_true",
        help="不保存测试集 npz（不建议：CNN 后续评测需要复用同一批数据）",
    )
    args = parser.parse_args()

    start_angle, end_angle = args.signal_range
    snrs = list(args.snrs)

    # ===== 目录命名：协议即路径 =====
    protocol = (
        f"M{args.M}_k{args.k}_T{args.snap}"
        f"_grid{_fmt_float_tag(args.music_grid_step)}"
        f"_rho{_fmt_float_tag(args.rho)}"
        f"_seed{args.seed}"
        f"_sep{_fmt_float_tag(args.min_delta_theta)}"
        f"_{_snr_tag(snrs)}"
        f"_N{args.num_samples}"
    )

    base_data_root = args.data_root or os.path.join(ROOT_PATH, "data", "ULA_data", "opening_snap20_k3")
    base_results_root = args.results_root or os.path.join(ROOT_PATH, "results", "opening_snap20_k3")
    data_dir = os.path.join(base_data_root, protocol)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(base_results_root, f"{protocol}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    if not args.no_save_npz:
        os.makedirs(data_dir, exist_ok=True)

    # ===== 固定随机种子：核心是保证 theta_test_set 一致 =====
    np.random.seed(args.seed)

    # 生成固定 test DOA 集合（所有 SNR 共用；所有算法共用）
    theta_test_set = Create_random_k_input_theta(
        args.k,
        start_angle=start_angle,
        end_angle=end_angle,
        theta_num=args.num_samples,
        min_delta_theta=args.min_delta_theta,
    )

    config = {
        "protocol": protocol,
        "M": args.M,
        "k": args.k,
        "snap": args.snap,
        "snrs": snrs,
        "num_samples_per_snr": args.num_samples,
        "min_delta_theta": args.min_delta_theta,
        "signal_range": [start_angle, end_angle],
        "rho": args.rho,
        "seed": args.seed,
        "tau_deg": args.tau,
        "music_grid_step": args.music_grid_step,
        "dataset_grid_step": args.dataset_grid_step,
        "snr_set": args.snr_set,
        "data_dir": data_dir if not args.no_save_npz else None,
        "results_dir": results_dir,
    }
    _save_json(os.path.join(results_dir, "protocol.json"), config)

    # ===== 评测缓存 =====
    por_music_list: List[float] = []
    rmse_music_list: List[float] = []
    por_esprit_list: List[float] = []
    rmse_esprit_list: List[float] = []
    n_success_music_list: List[int] = []
    n_success_esprit_list: List[int] = []

    example_music_sp = None  # 保存一张低 SNR 谱图示例（用于 PPT）

    for snr_db in snrs:
        # 1) 生成该 SNR 的共享测试数据（SCM/DOA 等）
        dataset = ULA_dataset(args.M, start_angle, end_angle, args.dataset_grid_step, args.rho)
        batch_size = min(256, args.num_samples)
        Create_datasets(
            dataset,
            args.k,
            theta_test_set,
            batch_size=batch_size,
            snap=args.snap,
            snr=snr_db,
            snr_set=args.snr_set,
            keep_lists={"ori_scm", "doa", "enhance_scm", "scm"},
        )

        # 保存该 SNR 的测试集（供后续 CNN 复用）
        if not args.no_save_npz:
            dataset.save_all_data(os.path.join(data_dir, f"test_snr_{snr_db}"))

        doa_true = np.array(dataset.doa, dtype=np.float32)[:, : args.k]

        # 2) MUSIC
        music = Music(dataset.get_A, start=start_angle, end=end_angle, step=args.music_grid_step)
        doa_hat_music = np.zeros((args.num_samples, args.k), dtype=np.float32)
        for i, R in enumerate(dataset.ori_scm):
            _, doa_hat, sp = music.estimate(R, args.k, return_sp=True)
            doa_hat_music[i] = doa_hat.astype(np.float32)
            if example_music_sp is None and snr_db == -10:
                example_music_sp = {
                    "snr_db": snr_db,
                    "grid": np.arange(start_angle, end_angle + 0.001, args.music_grid_step),
                    "spectrum": sp,
                    "true": doa_true[i].copy(),
                    "pred": doa_hat.astype(np.float32).copy(),
                }

        por_music, rmse_music, n_succ_music = _por_and_cond_rmse(doa_true, doa_hat_music, args.tau)

        # 3) ESPRIT（TLS-ESPRIT）
        esprit = ESPRIT(dataset.get_theta_fromz, M=args.M, displacement=1)
        doa_hat_esprit = np.zeros((args.num_samples, args.k), dtype=np.float32)
        doa_hat_esprit[:] = np.nan
        for i, R in enumerate(dataset.ori_scm):
            try:
                _, doa_hat = esprit.tls_estimate(R, args.k)
                doa_hat_esprit[i] = doa_hat.astype(np.float32)
            except Exception:
                # 数值不稳定/分解失败：保持 NaN，后续按失败样本处理
                pass

        por_esprit, rmse_esprit, n_succ_esprit = _por_and_cond_rmse(doa_true, doa_hat_esprit, args.tau)

        por_music_list.append(por_music)
        rmse_music_list.append(rmse_music)
        por_esprit_list.append(por_esprit)
        rmse_esprit_list.append(rmse_esprit)
        n_success_music_list.append(n_succ_music)
        n_success_esprit_list.append(n_succ_esprit)

        print(
            f"SNR={snr_db:>4} dB | "
            f"MUSIC: PoR={por_music:.3f}, RMSE_cond={rmse_music:.3f}, n_succ={n_succ_music:>4} | "
            f"ESPRIT: PoR={por_esprit:.3f}, RMSE_cond={rmse_esprit:.3f}, n_succ={n_succ_esprit:>4}"
        )

    # ===== 输出汇总 CSV（开题直接用）=====
    rows = []
    for idx, snr_db in enumerate(snrs):
        rows.append(
            {
                "snr_db": snr_db,
                "por_music": por_music_list[idx],
                "rmse_cond_music_deg": rmse_music_list[idx],
                "n_success_music": n_success_music_list[idx],
                "por_esprit": por_esprit_list[idx],
                "rmse_cond_esprit_deg": rmse_esprit_list[idx],
                "n_success_esprit": n_success_esprit_list[idx],
            }
        )

    metrics_csv = os.path.join(results_dir, "metrics_music_esprit_snap20.csv")
    _save_csv(
        metrics_csv,
        rows,
        header=[
            "snr_db",
            "por_music",
            "rmse_cond_music_deg",
            "n_success_music",
            "por_esprit",
            "rmse_cond_esprit_deg",
            "n_success_esprit",
        ],
    )

    plt = _try_import_matplotlib()
    if plt is None:
        print(f"Saved metrics: {metrics_csv}")
        print(f"Done. Outputs in: {results_dir}")
        return

    # 图1：PoR vs SNR
    _plot_two_curves(
        plt,
        snrs,
        por_music_list,
        por_esprit_list,
        label1="MUSIC",
        label2="ESPRIT",
        xlabel="SNR (dB)",
        ylabel=f"PoR / ACC (τ={args.tau}°)",
        title=f"PoR vs SNR (M={args.M}, k={args.k}, snap={args.snap})",
        save_path=os.path.join(results_dir, "por_vs_snr_music_esprit.png"),
    )

    # 图2：Conditional RMSE vs SNR（注意：某些 SNR 可能无成功样本 -> NaN）
    _plot_two_curves(
        plt,
        snrs,
        rmse_music_list,
        rmse_esprit_list,
        label1="MUSIC",
        label2="ESPRIT",
        xlabel="SNR (dB)",
        ylabel="Conditional RMSE (deg)",
        title=f"Conditional RMSE vs SNR (success-only, τ={args.tau}°)",
        save_path=os.path.join(results_dir, "rmse_cond_vs_snr_music_esprit.png"),
    )

    # 额外：保存一张 MUSIC 谱图示例（默认 snr=-10）
    if example_music_sp is not None:
        fig = plt.figure(figsize=(7.2, 4.2))
        plt.plot(example_music_sp["grid"], example_music_sp["spectrum"], linewidth=1.4)
        for t in example_music_sp["true"]:
            plt.axvline(float(t), color="g", linestyle="--", linewidth=1, label="true" if t == example_music_sp["true"][0] else None)
        for p in example_music_sp["pred"]:
            if np.isfinite(p):
                plt.axvline(float(p), color="r", linestyle=":", linewidth=1, label="pred" if p == example_music_sp["pred"][0] else None)
        plt.xlabel("Angle (deg)")
        plt.ylabel("MUSIC Spectrum")
        plt.title(f"Example MUSIC Spectrum (SNR={example_music_sp['snr_db']} dB, grid={args.music_grid_step}°)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"music_spectrum_snr_{example_music_sp['snr_db']}.png"), dpi=220)
        plt.close(fig)

    print(f"Saved metrics: {metrics_csv}")
    print(f"Done. Outputs in: {results_dir}")
    if not args.no_save_npz:
        print(f"Saved test npz in: {data_dir}")


if __name__ == "__main__":
    main()
