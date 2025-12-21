import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from data_creater.file_dataloader import file_array_Dataloader
from models.dl_model.CNN.literature_CNN import std_CNN
from models.dl_model.CNN.std_cnn_cbam import std_CNN_CBAM
from models.dl_model.CNN.std_cnn_se import std_CNN_SE
from models.dl_model.CNN.std_cnn_specse import std_CNN_SpecSE


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        return None
    return plt


def _sort_align_error_deg(theta_true: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    theta_true = np.sort(theta_true, axis=-1)
    theta_hat = np.sort(theta_hat, axis=-1)
    return np.abs(theta_hat - theta_true)


def _por_and_cond_rmse(theta_true: np.ndarray, theta_hat: np.ndarray, tau_deg: float) -> Tuple[float, float, int]:
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


def _load_json_if_exists(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_csv(path: str, rows: List[Dict], header: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")


def _plot_curves(plt, snrs: List[int], series: Dict[str, List[float]], ylabel: str, title: str, save_path: str):
    fig = plt.figure(figsize=(6.8, 4.3))
    for name, y in series.items():
        plt.plot(snrs, y, marker="o", linewidth=2, label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def _build_model_from_train_args(train_args: Dict) -> Tuple[torch.nn.Module, str]:
    """
    复用 train_snr_sp.py 的配置来构建同结构模型，确保能正确 load state_dict。
    """
    model_type = train_args.get("model", "std")
    input_type = train_args.get("input_type", "enhance_scm")
    M = int(train_args.get("M", 8))
    start_angle, end_angle = train_args.get("signal_range", [-60, 60])
    step = float(train_args.get("grid_step", 1.0))

    if input_type == "enhance_scm":
        in_c = 3
    elif input_type == "scm":
        in_c = 2
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    out_dims = int(round((end_angle - start_angle) / step)) + 1
    common_kwargs = dict(start_angle=start_angle, end_angle=end_angle, step=step)

    if model_type == "std":
        return std_CNN(in_c, M, out_dims, sp_mode=True, **common_kwargs), "std_CNN"
    if model_type == "se":
        return std_CNN_SE(in_c, M, out_dims, sp_mode=True, **common_kwargs), "std_CNN_SE"
    if model_type == "cbam":
        cbam_reduction = int(train_args.get("cbam_reduction", 16))
        cbam_spatial_kernel = int(train_args.get("cbam_spatial_kernel", 3))
        cbam_each_stage = bool(train_args.get("cbam_each_stage", True))
        return (
            std_CNN_CBAM(
                in_c,
                M,
                out_dims,
                sp_mode=True,
                cbam_reduction=cbam_reduction,
                spatial_kernel_size=cbam_spatial_kernel,
                cbam_each_stage=cbam_each_stage,
                **common_kwargs,
            ),
            "std_CNN_CBAM",
        )
    if model_type == "specse":
        specse_reduction = int(train_args.get("specse_reduction", 16))
        return (
            std_CNN_SpecSE(in_c, M, out_dims, sp_mode=True, specse_reduction=specse_reduction, **common_kwargs),
            "std_CNN_SpecSE",
        )
    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def _predict_doa(model: torch.nn.Module, dataloader, device: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 test npz 中读取 (input, doa_true)，输出 doa_pred（形状 N×k）与 doa_true（N×k）。
    """
    model.eval()
    doa_true_all = []
    doa_pred_all = []

    for x, doa_true in dataloader:
        logits = model(x.to(device))
        sp = torch.sigmoid(logits)
        _, doa_pred = model.sp_to_doa(sp, k)
        doa_true_all.append(doa_true.detach().cpu().numpy())
        doa_pred_all.append(doa_pred.detach().cpu().numpy())

    doa_true_all = np.concatenate(doa_true_all, axis=0)
    doa_pred_all = np.concatenate(doa_pred_all, axis=0)
    return doa_pred_all, doa_true_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN on opening shared testset (npz) and merge with MUSIC/ESPRIT metrics.")
    parser.add_argument("--results_dir", type=str, required=True, help="opening_baselines_protocol.py 输出的 results 目录（包含 protocol.json 与 metrics_music_esprit_snap20.csv）")
    parser.add_argument("--weights_dir", type=str, required=True, help="train_snr_sp.py 的保存目录（包含 laboratory_set.json 与 weight_snr_*.pth）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--tau", type=float, default=2.0, help="PoR/ACC 阈值 τ（度），默认 2°")
    parser.add_argument(
        "--peak_min_sep_deg",
        type=float,
        default=0.0,
        help="从模型输出的空间谱提取 k 个 DOA 时的最小角间隔（度）；>0 时启用 NMS-topk（建议与 min_delta_theta 一致或略小）",
    )
    parser.add_argument("--snrs_eval", type=int, nargs="+", default=None, help="只评测/绘制指定 SNR 子集（例如 -15 -10）")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出目录（空则写入 results_dir；建议单点预实验写到 results_dir 下的子目录以免覆盖主结果）",
    )
    args = parser.parse_args()

    protocol = _load_json_if_exists(os.path.join(args.results_dir, "protocol.json"))
    snrs = protocol.get("snrs")
    data_dir = protocol.get("data_dir")
    if not snrs or not data_dir:
        raise RuntimeError("results_dir 中缺少 protocol.json 或其中不包含 snrs/data_dir，请先运行 opening_baselines_protocol.py。")

    # 只评测指定 SNR 子集（用于注意力单点预实验）
    if args.snrs_eval is not None:
        snrs = [int(s) for s in args.snrs_eval]

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    train_args = _load_json_if_exists(os.path.join(args.weights_dir, "laboratory_set.json"))
    if not train_args:
        raise RuntimeError("weights_dir 中找不到 laboratory_set.json（train_snr_sp.py 会自动生成）。")

    k = int(train_args.get("k", 3))
    input_type = train_args.get("input_type", "enhance_scm")
    batch_size = int(train_args.get("batch_size", 256))

    model, model_name = _build_model_from_train_args(train_args)
    # 评测阶段可选：启用 NMS-topk 的最小间隔先验（不会影响已训练权重）
    if hasattr(model, "peak_min_sep_deg"):
        model.peak_min_sep_deg = float(args.peak_min_sep_deg)
    model.to(args.device)

    # 对每个 SNR：加载对应权重 + 在共享 test npz 上评测
    rows = []
    por_list = []
    rmse_list = []

    for snr_db in snrs:
        weight_path = os.path.join(args.weights_dir, f"weight_snr_{snr_db}.pth")
        test_npz = os.path.join(data_dir, f"test_snr_{snr_db}.npz")

        if not os.path.exists(weight_path):
            por, rmse, n_succ = 0.0, float("nan"), 0
            rows.append(
                {"snr_db": snr_db, "por_cnn": por, "rmse_cond_cnn_deg": rmse, "n_success_cnn": n_succ, "missing": "weight"}
            )
            por_list.append(por)
            rmse_list.append(rmse)
            continue

        if not os.path.exists(test_npz):
            por, rmse, n_succ = 0.0, float("nan"), 0
            rows.append(
                {"snr_db": snr_db, "por_cnn": por, "rmse_cond_cnn_deg": rmse, "n_success_cnn": n_succ, "missing": "npz"}
            )
            por_list.append(por)
            rmse_list.append(rmse)
            continue

        state_dict = torch.load(weight_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        model.to(args.device)

        dataloader = file_array_Dataloader(
            test_npz,
            batch_size=batch_size,
            shuffle=False,
            load_style="torch",
            input_type=input_type,
            output_type="doa",
        )

        doa_pred, doa_true = _predict_doa(model, dataloader, args.device, k)
        doa_true = doa_true[:, :k]
        doa_pred = doa_pred[:, :k]

        por, rmse, n_succ = _por_and_cond_rmse(doa_true, doa_pred, args.tau)
        rows.append({"snr_db": snr_db, "por_cnn": por, "rmse_cond_cnn_deg": rmse, "n_success_cnn": n_succ, "missing": ""})
        por_list.append(por)
        rmse_list.append(rmse)

        print(f"SNR={snr_db:>4} dB | {model_name}: PoR={por:.3f}, RMSE_cond={rmse:.3f}, n_succ={n_succ:>4}")

    # 保存 CNN 自身指标
    cnn_metrics_csv = os.path.join(out_dir, f"metrics_{model_name}_on_opening_testset.csv")
    _save_csv(
        cnn_metrics_csv,
        rows,
        header=["snr_db", "por_cnn", "rmse_cond_cnn_deg", "n_success_cnn", "missing"],
    )

    # 合并 MUSIC/ESPRIT + CNN（开题两张主图）
    baseline_csv = os.path.join(args.results_dir, "metrics_music_esprit_snap20.csv")
    if not os.path.exists(baseline_csv):
        print(f"Saved CNN metrics only: {cnn_metrics_csv}")
        return

    baseline = np.genfromtxt(baseline_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    baseline_by_snr = {int(r["snr_db"]): r for r in baseline}

    merged_rows = []
    por_music = []
    por_esprit = []
    por_cnn = []
    rmse_music = []
    rmse_esprit = []
    rmse_cnn = []

    cnn_by_snr = {int(r["snr_db"]): r for r in rows}

    for snr_db in snrs:
        b = baseline_by_snr.get(int(snr_db))
        c = cnn_by_snr.get(int(snr_db), {})

        merged_rows.append(
            {
                "snr_db": snr_db,
                "por_music": float(b["por_music"]) if b is not None else "",
                "por_esprit": float(b["por_esprit"]) if b is not None else "",
                "por_cnn": float(c.get("por_cnn", "")),
                "rmse_cond_music_deg": float(b["rmse_cond_music_deg"]) if b is not None else "",
                "rmse_cond_esprit_deg": float(b["rmse_cond_esprit_deg"]) if b is not None else "",
                "rmse_cond_cnn_deg": float(c.get("rmse_cond_cnn_deg", "")) if c.get("rmse_cond_cnn_deg", "") != "" else "",
            }
        )

        por_music.append(float(b["por_music"]) if b is not None else float("nan"))
        por_esprit.append(float(b["por_esprit"]) if b is not None else float("nan"))
        por_cnn.append(float(c.get("por_cnn", "nan")))
        rmse_music.append(float(b["rmse_cond_music_deg"]) if b is not None else float("nan"))
        rmse_esprit.append(float(b["rmse_cond_esprit_deg"]) if b is not None else float("nan"))
        rmse_cnn.append(float(c.get("rmse_cond_cnn_deg", "nan")))

    merged_csv = os.path.join(out_dir, "metrics_music_esprit_cnn_opening.csv")
    _save_csv(
        merged_csv,
        merged_rows,
        header=[
            "snr_db",
            "por_music",
            "por_esprit",
            "por_cnn",
            "rmse_cond_music_deg",
            "rmse_cond_esprit_deg",
            "rmse_cond_cnn_deg",
        ],
    )

    plt = _try_import_matplotlib()
    if plt is None:
        print(f"Saved: {cnn_metrics_csv}")
        print(f"Saved: {merged_csv}")
        return

    _plot_curves(
        plt,
        snrs,
        {"MUSIC": por_music, "ESPRIT": por_esprit, model_name: por_cnn},
        ylabel=f"PoR / ACC (τ={args.tau}°)",
        title=f"PoR vs SNR (snap=20, M=8, k={k})",
        save_path=os.path.join(out_dir, "opening_por_vs_snr_music_esprit_cnn.png"),
    )
    _plot_curves(
        plt,
        snrs,
        {"MUSIC": rmse_music, "ESPRIT": rmse_esprit, model_name: rmse_cnn},
        ylabel="Conditional RMSE (deg)",
        title=f"Conditional RMSE vs SNR (success-only, τ={args.tau}°)",
        save_path=os.path.join(out_dir, "opening_rmse_cond_vs_snr_music_esprit_cnn.png"),
    )

    # 保存一个小的记录，方便回溯“用哪个权重目录评测的”
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(out_dir, f"cnn_eval_record_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"weights_dir": args.weights_dir, "model_name": model_name, "tau_deg": args.tau, "device": args.device},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {cnn_metrics_csv}")
    print(f"Saved: {merged_csv}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
