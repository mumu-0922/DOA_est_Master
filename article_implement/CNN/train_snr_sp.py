import numpy as np
import argparse
import os
import json

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import gc

from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import inspect

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from data_creater.signal_datasets import ULA_dataset, array_Dataloader
from data_creater.Create_k_source_dataset import Create_random_k_input_theta, \
    Create_datasets
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot
from utils.loss_function import multi_classification_loss

from models.dl_model.CNN.literature_CNN import std_CNN
from models.dl_model.CNN.std_cnn_se import std_CNN_SE
from models.dl_model.CNN.std_cnn_cbam import std_CNN_CBAM

from utils.early_stop import EarlyStopping


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch, grid_to_theta=True, k=3):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:
            pred = F.sigmoid(pred)
            _, pred = model.sp_to_doa(pred, k)

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1))
                                                                       )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, epoch, grid_to_theta=True, k=3):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:
            pred = F.sigmoid(pred)
            _, pred = model.sp_to_doa(pred, k)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1))
                                                                       )

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_doa_rmse(model, data_loader, device, k=3):
    """
    将模型输出的空间谱（grid-based）通过 sp_to_doa 转成 DOA，再计算 RMSE（单位：度）。
    同时返回 success_ratio：是否能成功找到 k 个峰（不等同于估计正确）。
    """
    model.eval()
    succ_all = []
    doa_true_all = []
    doa_pred_all = []

    for x, doa_true in data_loader:
        pred_logits = model(x.to(device))
        pred_sp = torch.sigmoid(pred_logits)
        succ, doa_pred = model.sp_to_doa(pred_sp, k)

        succ_all.append(succ.detach().cpu().numpy().astype(bool))
        doa_true_all.append(doa_true.detach().cpu().numpy())
        doa_pred_all.append(doa_pred.detach().cpu().numpy())

    succ_all = np.concatenate(succ_all, axis=0)
    doa_true_all = np.concatenate(doa_true_all, axis=0)
    doa_pred_all = np.concatenate(doa_pred_all, axis=0)

    succ_ratio = float(np.mean(succ_all))
    if np.sum(succ_all) == 0:
        return float("nan"), succ_ratio

    doa_true = np.sort(doa_true_all[succ_all], axis=-1)
    doa_pred = np.sort(doa_pred_all[succ_all], axis=-1)
    rmse = float(np.sqrt(np.mean((doa_true - doa_pred) ** 2)))
    return rmse, succ_ratio


def build_model(model_type: str, M: int):
    """
    一键切换模型：
    - std：原始 std_CNN
    - se：std_CNN + SE 通道注意力（models/dl_model/CNN/std_cnn_se.py）
    """
    if model_type == "std":
        return std_CNN(3, M, 121, sp_mode=True), "std_CNN"
    if model_type == "se":
        return std_CNN_SE(3, M, 121, sp_mode=True), "std_CNN_SE"
    if model_type == "cbam":
        return std_CNN_CBAM(3, M, 121, sp_mode=True), "std_CNN_CBAM"
    raise ValueError(f"Unknown model_type: {model_type}")


def build_model_v2(
        model_type: str,
        input_type: str,
        M: int,
        start_angle: int,
        end_angle: int,
        step: float,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 3,
        cbam_each_stage: bool = True,
):
    """
    Build model with a configurable input type / angle grid.

    input_type:
      - enhance_scm: 3-channel input (Real/Imag/Angle), uses in_c=3
      - scm:         2-channel input (Real/Imag), uses in_c=2
    """
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
        return std_CNN_CBAM(
            in_c,
            M,
            out_dims,
            sp_mode=True,
            cbam_reduction=cbam_reduction,
            spatial_kernel_size=cbam_spatial_kernel,
            cbam_each_stage=cbam_each_stage,
            **common_kwargs,
        ), "std_CNN_CBAM"
    raise ValueError(f"Unknown model_type: {model_type}")


def main(args):
    # ===== Opening/quick experiment runner (std_CNN vs std_CNN_SE) =====
    # This block enables one-click switching and comparison plotting.
    # It returns early so the legacy code below is not executed.
    start_angle, end_angle = args.signal_range
    step = args.grid_step

    if args.quick:
        args.epochs = min(args.epochs, 50)
        args.train_theta_num = min(args.train_theta_num, 6000)
        args.val_theta_num = min(args.val_theta_num, 2000)
        args.early_stop_patience = min(args.early_stop_patience, 10)

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Same DOA sets for all models (fair comparison)
    train_theta_set = Create_random_k_input_theta(
        args.k, start_angle, end_angle, args.train_theta_num, min_delta_theta=args.min_delta_theta
    )
    val_theta_set = Create_random_k_input_theta(
        args.k, start_angle, end_angle, args.val_theta_num, min_delta_theta=args.min_delta_theta
    )

    if args.compare:
        if args.model == "std":
            model_types = ["std", "cbam"]
        else:
            model_types = ["std", args.model]
    else:
        model_types = [args.model]
    rmse_curves = {}

    for model_type in model_types:
        # Each model saves into a separate directory (avoid overwriting)
        _, model_name = build_model_v2(
            model_type,
            args.input_type,
            args.M,
            start_angle,
            end_angle,
            step,
            cbam_reduction=args.cbam_reduction,
            cbam_spatial_kernel=args.cbam_spatial_kernel,
            cbam_each_stage=args.cbam_each_stage,
        )
        save_path = os.path.join(
            args.save_root,
            f"{model_name}_M_{args.M}_k_{args.k}_snap_{args.snap}_rho_{args.rho}_in_{args.input_type}",
        )
        os.makedirs(save_path, exist_ok=True)

        save_args(args, os.path.join(save_path, "laboratory_set.json"))
        tb_writer = SummaryWriter(logdir=os.path.join(save_path, "run"))
        tags = ["train_loss", "val_loss", "learning_rate"]
        loss_function = multi_classification_loss()

        vloss_total = np.zeros((len(args.snrs)))
        doa_rmse_total = np.zeros((len(args.snrs)))
        succ_total = np.zeros((len(args.snrs)))

        for step_1, snr in enumerate(args.snrs):
            id1 = f"_snr_{snr}"

            # Ensure the same random signal/noise for std vs se under the same SNR
            data_seed = args.seed + 1000 + 97 * step_1
            np.random.seed(data_seed)
            random.seed(data_seed)

            model, _ = build_model_v2(
                model_type,
                args.input_type,
                args.M,
                start_angle,
                end_angle,
                step,
                cbam_reduction=args.cbam_reduction,
                cbam_spatial_kernel=args.cbam_spatial_kernel,
                cbam_each_stage=args.cbam_each_stage,
            )
            model.to(args.device)

            dataset = ULA_dataset(args.M, start_angle, end_angle, step, args.rho)
            val_dataset = ULA_dataset(args.M, start_angle, end_angle, step, args.rho)
            Create_datasets(
                dataset,
                args.k,
                train_theta_set,
                batch_size=args.train_batch_theta,
                snap=args.snap,
                snr=snr,
                snr_set=args.snr_set,
            )
            Create_datasets(
                val_dataset,
                args.k,
                val_theta_set,
                batch_size=args.val_batch_theta,
                snap=args.snap,
                snr=snr,
                snr_set=args.snr_set,
            )

            # Prune unused large lists to reduce RAM peak.
            # We must keep:
            # - ori_scm: __len__ uses it
            # - input_type: model input
            # - spatial_sp: training label
            # - doa: evaluation label
            keep_lists = {"ori_scm", args.input_type, "spatial_sp", "doa"}
            for _ds in (dataset, val_dataset):
                for name, value in list(_ds.__dict__.items()):
                    if isinstance(value, list) and name not in keep_lists:
                        setattr(_ds, name, [])

            train_dataloader = array_Dataloader(
                dataset, args.batch_size, load_style="torch", input_type=args.input_type, output_type="spatial_sp"
            )
            val_dataloader = array_Dataloader(
                val_dataset,
                args.batch_size,
                shuffle=False,
                load_style="torch",
                input_type=args.input_type,
                output_type="spatial_sp",
            )

            init_dir = os.path.join(save_path, "_init_weight")
            os.makedirs(init_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(init_dir, f"_init_weight" + id1 + ".pth"))

            model_class_file = os.path.join(save_path, "model.py")
            with open(model_class_file, "w", encoding="utf-8") as f:
                f.write(inspect.getsource(model.__class__))

            parm = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(parm, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
            lr_scheduler = None
            if args.use_lr_scheduler:
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=args.lr_factor,
                    patience=args.lr_patience,
                    min_lr=args.min_lr,
                )

            early_stopping = EarlyStopping(args.early_stop_patience, 0)

            min_val_loss = 1e9
            best_weight_path = os.path.join(save_path, f"weight" + id1 + ".pth")

            for epoch in range(args.epochs):
                train_loss = train_one_epoch(
                    model, train_dataloader, loss_function, optimizer, args.device, epoch + 1, False, args.k
                )
                val_loss = evaluate(model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)

                tb_writer.add_scalar(tags[0] + id1, train_loss, epoch)
                tb_writer.add_scalar(tags[1] + id1, val_loss, epoch)
                tb_writer.add_scalar(tags[2] + id1, optimizer.param_groups[0]["lr"], epoch)

                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), best_weight_path)
                    print(f"[{model_name}]{id1} saved, min val_loss={min_val_loss}")

                if epoch == args.epochs - 1:
                    torch.save(model.state_dict(), os.path.join(save_path, f"weight_end" + id1 + ".pth"))

            vloss_total[step_1] = min_val_loss

            # DOA RMSE on validation set (convert SP -> DOA)
            val_dataloader_doa = array_Dataloader(
                val_dataset, args.batch_size, shuffle=False, load_style="torch", input_type=args.input_type, output_type="doa"
            )
            state_dict = torch.load(best_weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            model.to(args.device)
            doa_rmse, succ_ratio = evaluate_doa_rmse(model, val_dataloader_doa, args.device, args.k)
            doa_rmse_total[step_1] = doa_rmse
            succ_total[step_1] = succ_ratio
            print(f"[{model_name}]{id1} DOA_RMSE={doa_rmse:.3f} deg | succ={succ_ratio:.3f}")

            # Explicitly release large Python objects between SNR runs (important for compare mode).
            del train_dataloader, val_dataloader, val_dataloader_doa, dataset, val_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        header = [f"snr_{i}" for i in args.snrs]
        index = [f"snap_{args.snap}"]
        random_name = str(np.random.rand(1))

        save_array(vloss_total, os.path.join(save_path, "validation_loss_" + random_name + ".csv"), header=header, index=index)
        loss_1d_plot(vloss_total, model_name, args.snrs, "SNR(db)", False, os.path.join(save_path, f"validation_loss_{random_name}.png"))

        save_array(doa_rmse_total, os.path.join(save_path, "doa_rmse_" + random_name + ".csv"), header=header, index=index)
        loss_1d_plot(doa_rmse_total, model_name, args.snrs, "SNR(db)", False, os.path.join(save_path, f"doa_rmse_{random_name}.png"))

        save_array(succ_total, os.path.join(save_path, "success_ratio_" + random_name + ".csv"), header=header, index=index)

        rmse_curves[model_name] = doa_rmse_total
        tb_writer.close()

    # Comparison plot (std vs attention)
    if args.compare and len(rmse_curves) == 2:
        compare_dir = os.path.join(
            args.save_root,
            f"compare_{model_types[0]}_vs_{model_types[1]}_M_{args.M}_k_{args.k}_snap_{args.snap}_rho_{args.rho}",
        )
        os.makedirs(compare_dir, exist_ok=True)
        tags = list(rmse_curves.keys())
        curves = [rmse_curves[tags[0]], rmse_curves[tags[1]]]
        loss_1d_plot(curves, tags, args.snrs, "SNR(db)", False, os.path.join(compare_dir, "compare_doa_rmse.png"))
        save_array(np.vstack(curves), os.path.join(compare_dir, "compare_doa_rmse.csv"), header=[f"snr_{i}" for i in args.snrs], index=tags)

    return

    # save_path
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                  args.signal_range[1], 50000, min_delta_theta=2)
    val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                args.signal_range[1], 20000, min_delta_theta=2)
    # theta_set = Create_determined_sep_doas(args.k, args.signal_range[0], args.signal_range[1], None, 10, True, 0.1)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))
    # save the run_file
    tb_writer = SummaryWriter(logdir=os.path.join(save_path, 'run'))
    tags = ["train_loss", "val_loss", "learning_rate"]

    # loss function
    loss_function = multi_classification_loss()

    vloss_total = np.zeros((len(args.snrs)))
    for step_1, snr in enumerate(args.snrs):
        id1 = f'_snr_{snr}'
        model = std_CNN(3, args.M, 121, sp_mode=True)
        # model = modified_CNN(2, args.M, 121, sp_mode=True)
        model_name = 'std_CNN'
        model.to(args.device)

        dataset, val_dataset = ULA_dataset(args.M, -60, 60, 1, args.rho), ULA_dataset(args.M, -60, 60, 1, args.rho)
        Create_datasets(dataset, args.k, train_theta_set, batch_size=100, snap=args.snap, snr=snr)
        Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=args.snap, snr=snr)

        train_dataloader = array_Dataloader(dataset, 256, load_style='torch', input_type='enhance_scm',
                                            output_type='spatial_sp')
        val_dataloader = array_Dataloader(val_dataset, 256, shuffle=False, load_style='torch',
                                          input_type='enhance_scm', output_type='spatial_sp')
        # train_dataloader = array_Dataloader(dataset, 256, load_style='torch', input_type='scm',
        #                                     output_type='spatial_sp')
        # val_dataloader = array_Dataloader(val_dataset, 256, shuffle=False, load_style='torch',
        #                                   input_type='scm', output_type='doa')

        # save initial model weight,
        if not os.path.exists(os.path.join(save_path, '_init_weight')):
            os.makedirs(os.path.join(save_path, '_init_weight'))
        torch.save(model.state_dict(), os.path.join(save_path, '_init_weight', f'_init_weight' + id1 + '.pth'))

        # save the model structure，知道自己跑了啥代码
        model_class_file = os.path.join(save_path, 'model.py')
        with open(model_class_file, 'w') as f:
            f.write(inspect.getsource(model.__class__))

        # 优化器初始化
        parm = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(parm,lr=0.0001,momentum=0.9,weight_decay=0)
        optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        # optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        # lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, 0.0001, 'rel', 3, 0, 1e-8,
        #                                                    False)
        # lr_schedule = optim.lr_scheduler.StepLR(optimizer, 10, 0.5, -1)

        # stop training earlier if validation loss doesn't decrease
        early_stopping = EarlyStopping(30, 0)

        # 对所有的信噪比情况进行训练
        min_val_loss = 100
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer, args.device, epoch + 1,
                                         False, args.k)
            val_loss = evaluate(model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)

            # 根据val_loss,学习率调度器
            # lr_schedule.step(val_loss)

            # 保存数据
            tb_writer.add_scalar(tags[0] + id1, train_loss, epoch)
            tb_writer.add_scalar(tags[1] + id1, val_loss, epoch)
            tb_writer.add_scalar(tags[2] + id1, optimizer.param_groups[0]["lr"], epoch)

            # 提早停止训练
            early_stopping(val_loss)
            # 若满足 early stopping 要求,结束模型训练
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, f'weight' + id1 + '.pth'))
                print(f'model saved, minimun val_loss:{min_val_loss}')

            # 保存最后一个epoch的模型参数
            if epoch == args.epochs - 1:
                torch.save(model.state_dict(), os.path.join(save_path, f'weight_end' + id1 + '.pth'))

        vloss_total[step_1] = min_val_loss

    # save loss
    random_name = str(np.random.rand(1))
    save_array(vloss_total, os.path.join(save_path, 'validation_loss_' + random_name + '.csv'),
               index=['snap_' + str(args.snap)],
               header=['snr_' + str(i) for i in args.snrs])
    # plot loss
    loss_1d_plot(vloss_total, model_name, args.snrs, 'SNR(db)', False,
                 os.path.join(save_path, f'validation_loss_{random_name}.png'))


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # scenario parameters
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snrs', type=int, nargs='+', default=[-20, -15, -10, -5, 0, 5])
    parser.add_argument('--snap', type=int, default=10)
    parser.add_argument('--signal_range', type=int, nargs=2, default=[-60, 60])
    parser.add_argument('--grid_step', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # one-click model switch / comparison
    parser.add_argument('--model', type=str, default='std', choices=['std', 'se', 'cbam'])
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--input_type', type=str, default='enhance_scm', choices=['enhance_scm', 'scm'])
    parser.add_argument('--snr_set', type=int, default=1, choices=[0, 1])

    root_path = ROOT_PATH
    save_root = os.path.join(root_path, 'results', 'CNN_load_path')
    parser.add_argument('--save_root', type=str, default=save_root)

    # training parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--quick', action='store_true')

    # learning rate scheduler (optional)
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    # dataset size / generation
    parser.add_argument('--train_theta_num', type=int, default=50000)
    parser.add_argument('--val_theta_num', type=int, default=20000)
    parser.add_argument('--min_delta_theta', type=float, default=2.0)
    parser.add_argument('--train_batch_theta', type=int, default=100)
    parser.add_argument('--val_batch_theta', type=int, default=512)

    # CBAM hyper-parameters (only used when --model cbam or --compare includes cbam)
    parser.add_argument('--cbam_reduction', type=int, default=16)
    parser.add_argument('--cbam_spatial_kernel', type=int, default=3, choices=[3, 7])
    parser.add_argument('--cbam_each_stage', action='store_true', default=True)
    parser.add_argument('--cbam_last_only', dest='cbam_each_stage', action='store_false')

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=True)

    args = parser.parse_args()

    main(args)
