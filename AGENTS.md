# DOA_est_Master 项目结构与脚本说明（开题/复现实验“结束文档”）

本文件用于：
1) 让第一次接触仓库的人能快速知道“每个目录/脚本是干什么的”；  
2) 给开题阶段提供“最小可行性验证（MUSIC → CNN → 注意力模块对比）”的一键运行入口与结果解释；  
3) 记录本仓库常见坑点（路径、PowerShell 参数换行、GPU/PyTorch 兼容、内存/数据规模）。

---

## 0. 快速入口（你最可能用到的 3 条命令）

### 0.1 开题证据图：MUSIC 在低 SNR 下的 RMSE 曲线
从仓库根目录运行：

```powershell
python .\test\test_file\opening_music_only.py --M 8 --k 3 --snap 200 --snrs -20 -15 -10 -5 0 --num_samples 2000 --min_delta_theta 8 --grid_step 0.5 --rho 0
```

输出：`results/opening/.../music_rmse.csv`、`music_rmse.png`、`music_spectrum_snr_-10.png`

### 0.2 一键对比：`std_CNN` vs `std_CNN_SE/CBAM`（并自动画 RMSE 对比图）
在 `article_implement/CNN/` 目录运行：

```powershell
cd .\article_implement\CNN
python .\train_snr_sp.py --device cuda --M 8 --k 3 --snap 200 --snrs -10 --model cbam --compare --cbam_last_only --cbam_spatial_kernel 3 --epochs 150 --early_stop_patience 30 --lr 1e-4 --weight_decay 1e-4 --use_lr_scheduler --train_theta_num 30000 --val_theta_num 8000 --train_batch_theta 200 --val_batch_theta 400 --min_delta_theta 8 --input_type enhance_scm --snr_set 1 --seed 0
```

输出：`results/CNN_load_path/compare_std_vs_cbam_.../compare_doa_rmse.csv`、`compare_doa_rmse.png` 以及每个模型各自的训练曲线/权重文件。

### 0.3 PowerShell 参数换行（重要）
PowerShell 续行符是反引号 `` ` ``，不是 `^`（`^` 是 cmd.exe 的）。

---

## 1. 目录结构（按“做什么”划分）

- `data_creater/`：仿真数据集生成（ULA/UCA、快拍数、SNR 扫描、空间谱 SP 等）。
- `models/`：模型与算法库（传统：MUSIC/ESPRIT/Capon；深度：CNN/ViT；压缩感知：l1-SVD 等）。
- `train/`：训练入口脚本（按 SNR、快拍数、样本数等维度组织）。
- `test/`：评测/对比脚本（偏“可复现实验脚本”，非单元测试套件）。
- `utils/`：训练与评测公用工具（early stop、loss、矩阵算子、训练/测试流程封装）。
- `data_save/`：结果落盘与绘图工具（CSV 保存、loss 曲线、CDF、混淆矩阵等）。
- `results/`：运行时生成的权重、CSV、图片（通常不建议提交产物）。
- `data/`：运行时生成/缓存的数据集（`.npz` 等；通常不建议提交大文件）。
- `article_implement/`：论文/方法复现集合（SPE-CNN、ASL、SubspaceNet、Learning-SPICE 等）。
- `vit_tranfer_learning/`：迁移学习相关实验（含 2D DOA）。
- `matlab_post_process/`：部分算法的 MATLAB 后处理/对照脚本（需要 MATLAB Engine 时会用到）。

---

## 2. 常见输出文件怎么读（尤其是开题要用的）

### 2.1 `music_rmse.csv`
示例字段：
- `snr_db`：信噪比（dB）。
- `rmse_deg`：在 `succ=True` 的样本子集上，DOA（角度）估计的 RMSE（单位：度）。
- `success_ratio`：算法是否“成功返回 k 个峰”的比例（不等于估计正确率；只是峰搜索是否找到了 k 个峰）。

解读要点：
- 低 SNR/快拍数少时，`rmse_deg` 往往偏大，这是“传统算法在低 SNR 失效/退化”的证据图。
- `success_ratio` 很高但 RMSE 也很高是正常的：表示“能找出 k 个峰”，但峰位置偏差大。

### 2.2 CNN 对比输出（`compare_doa_rmse.csv/png`）
对比脚本会输出：
- 每个模型在指定 SNR 的 DOA_RMSE（度）与 success_ratio；
- 汇总对比图：`std_CNN` vs `std_CNN_SE` 或 `std_CNN_CBAM`。

注意：
- 注意力模块不一定在所有设置下都更好；更可能在“更低 SNR / 更少快拍 / 更强阵列误差 rho”时体现优势。

---

## 3. 环境与兼容性（Windows/Conda/GPU）

- 推荐环境：`conda env create -f environment.yml`，然后 `conda activate DOA`
- 若出现 `sm_120 is not compatible`（例如 RTX 5060 Ti）：说明当前 PyTorch 不包含该算力架构的 CUDA kernel，需要安装支持更高算力的 PyTorch（通常是 nightly 或更新版本）。
- 运行脚本建议从脚本所在目录执行（仓库里存在相对路径拼接 `results/`、`data/` 的写法）。

---

## 4. “每个文件”的作用索引（按路径分组）

> 说明：下表只描述“脚本/模块用途”，具体参数含义以脚本内 argparse 为准；`results/` 与 `data/` 目录内容为运行时生成，不在此枚举。

### 4.1 根目录
- `README.markdown`：项目总体介绍、依赖说明、算法/脚本概览（英文为主）。
- `environment.yml`：Conda 环境依赖（注意：GPU 新型号可能需要更新 PyTorch）。
- `.gitignore`：忽略规则（一般会忽略 `results/`、大数据等）。
- `AGENTS.md`：本说明文件（项目结构/脚本用途/开题一键跑通指南）。

### 4.2 `article_implement/ASL/`
- `article_implement/ASL/asl_train.py`：ASL 方法训练脚本入口。
- `article_implement/ASL/asl_used.py`：ASL 方法的使用/评测脚本（实验脚本）。
- `article_implement/ASL/asl_used_min_sep.py`：ASL 在最小角间隔（min separation）场景下的实验脚本。
- `article_implement/ASL/model.py`：ASL 相关网络结构/模块定义。

### 4.3 `article_implement/CNN/`
- `article_implement/CNN/literature_CNN.py`：论文/基线 CNN 结构（或与文献一致的版本）。
- `article_implement/CNN/train_snr_sp.py`：按 SNR 训练/验证并可 `--compare` 输出 RMSE 对比图（开题一键对比常用）。
- `article_implement/CNN/train_snr_sp_ideal.py`：理想协方差/理想设置版本的训练脚本（用于上限对照）。
- `article_implement/CNN/train_snap_sp.py`：按快拍数（snap）训练/验证的实验脚本。
- `article_implement/CNN/tests_snr.py`：按 SNR 测试/评测脚本（偏实验复现）。
- `article_implement/CNN/tests_snap.py`：按快拍数测试/评测脚本。
- `article_implement/CNN/tests_min_sep.py`：按最小角间隔测试/评测脚本。
- `article_implement/CNN/tests_snr_article_set.py`：按论文设置（固定参数组合）测试/评测脚本。

### 4.4 `article_implement/Learning_SPICE/`
- `article_implement/Learning_SPICE/model.py`：Learning-SPICE 模型结构定义。
- `article_implement/Learning_SPICE/extend_R_dataset.py`：扩展/构造 R（协方差）数据的辅助脚本。
- `article_implement/Learning_SPICE/fit_snr.py`：Learning-SPICE 按 SNR 拟合/训练脚本。
- `article_implement/Learning_SPICE/fit_snap.py`：Learning-SPICE 按快拍数拟合/训练脚本。
- `article_implement/Learning_SPICE/post_tests_snr2.py`：Learning-SPICE 的后处理测试脚本（SNR 扫描版本）。
- `article_implement/Learning_SPICE/post_tests_snap2.py`：Learning-SPICE 的后处理测试脚本（快拍扫描版本）。
- `article_implement/Learning_SPICE/post_tests_min_sep.py`：Learning-SPICE 的后处理测试脚本（最小间隔版本）。
- `article_implement/Learning_SPICE/post_tests_snr2_M_16_k_3.py`：特定场景（M=16,k=3）的 SNR 后处理测试脚本。
- `article_implement/Learning_SPICE/post_tests_snr2_M_8_k_7.py`：特定场景（M=8,k=7）的 SNR 后处理测试脚本。
- `article_implement/Learning_SPICE/post_tests_snr2_rho.py`：阵列误差参数 rho 扫描的后处理测试脚本。

### 4.5 `article_implement/SubspaceNet/`
- `article_implement/SubspaceNet/model.py`：SubspaceNet 网络结构定义。
- `article_implement/SubspaceNet/signal_datasets.py`：SubspaceNet 版本的数据集/数据处理（与主框架可能有差异）。
- `article_implement/SubspaceNet/train_snr.py`：SubspaceNet 按 SNR 训练脚本。
- `article_implement/SubspaceNet/tests_snr.py`：SubspaceNet 按 SNR 测试脚本。
- `article_implement/SubspaceNet/tests_snr_rho.py`：SubspaceNet 在 rho 扫描下的测试脚本。
- `article_implement/SubspaceNet/tests_min_sep.py`：SubspaceNet 在最小角间隔下的测试脚本。
- `article_implement/SubspaceNet/src/utils.py`：SubspaceNet 子工程的工具函数。

### 4.6 `data_creater/`（数据生成核心）
- `data_creater/signal_datasets.py`：数据集核心类（ULA/UCA），生成/缓存 `y_t`、SCM、SP、DOA 等。
- `data_creater/Create_k_source_dataset.py`：k 源场景数据生成入口（随机 DOA 生成、批量生成数据等）。
- `data_creater/Create_classic_test_dataset.py`：生成传统算法/经典设置的测试数据。
- `data_creater/extend_R_dataset.py`：扩展/构造协方差矩阵 R 的数据脚本（辅助实验）。
- `data_creater/file_dataloader.py`：从文件加载数据集的 DataLoader 辅助。
- `data_creater/norm.py`：数据归一化/标准化相关工具。
- `data_creater/data_used/datasets_save_snr.py`：按 SNR 生成并落盘数据的脚本。
- `data_creater/data_used/datasets_save_snap.py`：按快拍数生成并落盘数据的脚本。
- `data_creater/data_used/datasets_save_min_sep.py`：按最小角间隔生成并落盘数据的脚本。
- `data_creater/UCA_datasets/UCA_datasets.py`：UCA 数据集核心类/工具。
- `data_creater/UCA_datasets/Create_2d_k_source_dataset.py`：UCA/2D k 源数据生成入口。
- `data_creater/UCA_datasets/Create_2d_classic_test_dataset.py`：UCA/2D 经典测试数据生成脚本。

### 4.7 `data_save/`（结果保存与绘图）
- `data_save/save_csv/loss_save.py`：将 loss/数组等保存为 CSV 的工具函数。
- `data_save/plot/plot_loss.py`：绘制训练/验证 loss 曲线（支持单点/多点场景）。
- `data_save/plot/plot_CDF.py`：CDF 曲线绘图工具。
- `data_save/plot/plot_confusion_matrix.py`：混淆矩阵绘图工具。
- `data_save/plot/plot_pre_result.py`：预测结果可视化脚本。
- `data_save/plot/plot_radar.py`：雷达图绘制脚本（多指标对比）。
- `data_save/plot/file_plot/load_predict_results_and_plot.py`：读取结果文件并绘图（通用入口）。
- `data_save/plot/file_plot/load_predict_results_and_plot2.py`：读取结果并绘图（另一版/扩展）。
- `data_save/plot/file_plot/load_predict_results_and_plot_M_16_k_3.py`：特定场景（M=16,k=3）的结果绘图。
- `data_save/plot/file_plot/load_predict_results_and_plot_M_8_k_7.py`：特定场景（M=8,k=7）的结果绘图。
- `data_save/plot/file_plot/load_predict_results_and_plot_rho.py`：rho 扫描结果绘图。
- `data_save/plot/file_plot/load_predict_results_and_plot_sep.py`：最小角间隔结果绘图（版本 1）。
- `data_save/plot/file_plot/load_predict_results_and_plot_sep2.py`：最小角间隔结果绘图（版本 2）。
- `data_save/plot/file_plot/load_predict_results_and_plot_snap.py`：快拍扫描结果绘图。
- `data_save/plot/file_plot/plot_bar.py`：柱状图绘制脚本。
- `data_save/plot/file_plot/plot_two_scenarios_transfer_learning.py`：迁移学习两场景对比绘图。
- `data_save/plot/file_plot/plot_two_scenarios_v1.py`：两场景对比绘图（版本 1）。
- `data_save/plot/file_plot/plot_two_scenarios_v2.py`：两场景对比绘图（版本 2）。
- `data_save/plot/file_plot/test1.py`：绘图/加载的小测试脚本。

### 4.8 `matlab_post_process/`（MATLAB 后处理/对照）
- `matlab_post_process/ASL_2/ASL_R_construct_k_2.m`：ASL-2 的 R 构造（k=2）脚本。
- `matlab_post_process/ASL_2/ASL_R_construct_k_3.m`：ASL-2 的 R 构造（k=3）脚本。
- `matlab_post_process/ASL_2/ASL_R_construct_k_n.m`：ASL-2 的通用 R 构造（k=n）脚本。
- `matlab_post_process/ASL_2/ASL_test_omp.m`：ASL-2 的 OMP 测试脚本。
- `matlab_post_process/ASL_2/ASL_test_omp_min_sep.m`：ASL-2 在 min-sep 场景下的 OMP 测试脚本。
- `matlab_post_process/ASL_2/generate_random_angles.m`：生成随机角度集合的 MATLAB 辅助脚本。
- `matlab_post_process/ASL_2/optimize_ASL_R_omp.m`：ASL-2 的 R/OMP 优化脚本。
- `matlab_post_process/ASL_2/rootmusicdoa.m`：Root-MUSIC DOA 的 MATLAB 实现/对照。
- `matlab_post_process/Learning_SPICE/optimize_threshold_R_construct.m`：Learning-SPICE 的阈值/构造优化脚本。
- `matlab_post_process/Learning_SPICE/R_construct.m`：Learning-SPICE 的 R 构造脚本。
- `matlab_post_process/Learning_SPICE/R_construct_test_omp.m`：Learning-SPICE 的 OMP 测试脚本。
- `matlab_post_process/Learning_SPICE/R_construct_test_omp_min_sep.m`：Learning-SPICE 的 min-sep OMP 测试脚本。
- `matlab_post_process/Learning_SPICE/R_construct_test_snap_omp.m`：Learning-SPICE 的快拍扫描 OMP 测试脚本。
- `matlab_post_process/Learning_SPICE/rootmusicdoa.m`：Root-MUSIC DOA 的 MATLAB 实现/对照（Learning-SPICE 目录版）。

### 4.9 `models/`（算法与网络）
#### 传统子空间/谱估计
- `models/subspace_model/music.py`：MUSIC 算法实现（支持输出空间谱）。
- `models/subspace_model/esprit.py`：ESPRIT 算法实现。
- `models/subspace_model/unity_esprit.py`：Unity-ESPRIT 算法实现。
- `models/subspace_model/capon.py`：Capon（MVDR）谱估计实现。

#### 压缩感知（MATLAB 联动）
- `models/compress_sensing/invoke_matlab/l1_svd.py`：`l1-SVD` 的 Python 侧封装（会调用 MATLAB 脚本）。
- `models/compress_sensing/invoke_matlab/python_call_l1_SVD_omp_plus.m`：l1-SVD（OMP+）MATLAB 调用脚本（常用于 SNR 扫描）。
- `models/compress_sensing/invoke_matlab/python_call_l1_SVD_snap.m`：l1-SVD（快拍扫描）MATLAB 调用脚本。

#### 深度学习（CNN/MLP/ViT）
- `models/dl_model/CNN/literature_CNN.py`：std_CNN 基线模型（本仓库 CNN 主干）。
- `models/dl_model/CNN/std_cnn_se.py`：std_CNN + SEBlock（通道注意力）变体。
- `models/dl_model/CNN/std_cnn_cbam.py`：std_CNN + CBAM（通道+空间注意力）变体。
- `models/dl_model/mlp/MLP.py`：MLP 基线模型。
- `models/dl_model/grid_based_network.py`：grid-based DOA 网络相关封装（输出空间谱/网格分类）。
- `models/dl_model/weight_init.py`：网络权重初始化工具。
- `models/dl_model/vision_transformer/embeding_layer.py`：ViT 嵌入层定义。
- `models/dl_model/vision_transformer/vit_model.py`：ViT 模型定义（DOA 任务版本）。
- `models/dl_model/vision_transformer/flops.py`：FLOPs/复杂度统计工具。

### 4.10 `test/`
- `test/test_file/opening_music_only.py`：开题“最小可行”MUSIC 基线脚本（输出 RMSE 曲线与示例谱图）。
- `test/test_file/tests_snr.py`：按 SNR 评测脚本（生成结果 CSV/图）。
- `test/test_file/tests_snap.py`：按快拍数评测脚本。
- `test/test_file/tests_min_sep.py`：按最小角间隔评测脚本。
- `test/test_file/tests_snr_M_16_k_3.py`：特定场景（M=16,k=3）SNR 评测脚本。
- `test/test_file/tests_snr_M_8_k_7.py`：特定场景（M=8,k=7）SNR 评测脚本。
- `test/test_file/tests_snr_rho.py`：rho 扫描评测脚本。
- `test/contrast_test/tests_snr.py`：对比测试脚本（可能用于不同模型/算法间对比）。
- `test/contrast_test/tests_snap.py`：快拍扫描对比测试脚本。

### 4.11 `train/`
- `train/train_snr.py`：按 SNR 训练入口脚本（框架主入口之一）。
- `train/train_snr_sp.py`：按 SNR 训练（空间谱输出/相关设置）的入口脚本。
- `train/train_snap.py`：按快拍数训练入口脚本。
- `train/train_snap_sp.py`：按快拍数训练（空间谱输出/相关设置）的入口脚本。
- `train/train_num.py`：按样本量/数据规模训练入口脚本。
- `train/train_snr_M_16.py`：特定场景（M=16）按 SNR 的训练入口。
- `train/train_snr_k_7.py`：特定场景（k=7）按 SNR 的训练入口。

### 4.12 `utils/`
- `utils/doa_train_and_test.py`：训练/测试流程封装（训练循环、评测指标等复用）。
- `utils/early_stop.py`：EarlyStopping 工具（验证集 loss 不下降则提前停止）。
- `utils/loss_function.py`：loss 定义（如多分类损失等）。
- `utils/util.py`：通用工具函数（路径、随机数、打印等）。
- `utils/matrix_operator.py`：矩阵运算工具（协方差、特征分解相关等）。
- `utils/batch_matrix_operator.py`：批量矩阵运算工具（加速仿真/训练）。

### 4.13 `vit_tranfer_learning/`
- `vit_tranfer_learning/train_num.py`：ViT/迁移学习按样本量训练入口。
- `vit_tranfer_learning/fine_tune.py`：微调（fine-tune）实验脚本。
- `vit_tranfer_learning/fine_tune_num.py`：按样本量微调实验脚本。
- `vit_tranfer_learning/tests_sample_num.py`：按样本量评测脚本。
- `vit_tranfer_learning/transfer_learning.py`：迁移学习主实验脚本。
- `vit_tranfer_learning/transfer_learning_num.py`：按样本量的迁移学习实验脚本。
- `vit_tranfer_learning/transfer_learning_one_set.py`：单一设置下的迁移学习实验脚本。
- `vit_tranfer_learning/theta_creater/theta_creater.py`：角度集合生成辅助脚本。
- `vit_tranfer_learning/2d_DOA/train_snr_2d.py`：2D DOA 按 SNR 训练脚本。
- `vit_tranfer_learning/2d_DOA/tests_snr_2d.py`：2D DOA 按 SNR 测试脚本。
- `vit_tranfer_learning/2d_DOA/tests_snr_2d_rho.py`：2D DOA rho 扫描测试脚本。
- `vit_tranfer_learning/2d_DOA/transfer_learning_2d.py`：2D DOA 迁移学习脚本。

---

## 5. 开题阶段建议的“最小可行路线图”（从做 → 写 → PPT）

1) 先跑 `opening_music_only.py` 得到 “RMSE vs SNR + 谱图” 两张证据图；  
2) 再跑 `train_snr_sp.py --compare` 得到 “std vs 注意力” 的一张对比图；  
3) 用这 3 张图支撑开题：  
   - 为什么做：传统方法低 SNR 退化（MUSIC 曲线/谱图）  
   - 怎么做：深度模型 + 注意力增强（对比图）  
   - 能不能做出来：脚本一键复现 + 自动出图（可行性证明）
