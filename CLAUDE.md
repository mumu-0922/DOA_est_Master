# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DOA（到达角）估计深度学习框架，支持 ULA/UCA 阵列配置，实现传统算法（MUSIC、ESPRIT、Capon）与深度学习模型（CNN、ViT）的对比实验。核心贡献是基于 Vision Transformer 的 DOA 估计模型及监督迁移学习方法。

## 环境配置

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate DOA

# 依赖：Python 3.11, PyTorch 2.5.1, CUDA 11.8
# 部分算法（l1-SVD, Learning-SPICE, ASL-2）需要 MATLAB Engine for Python
```

### VSCode 路径配置

项目依赖正确的 PYTHONPATH，在 `.vscode/settings.json` 中添加：
```json
{
  "terminal.integrated.env.windows": { "PYTHONPATH": "${workspaceFolder}" },
  "python.terminal.executeInFileDir": true
}
```

## 常用命令

### 训练模型
```powershell
# 按 SNR 训练（从脚本所在目录运行）
cd train
python train_snr_sp.py --device cuda --M 8 --k 3 --snap 200 --snrs -10

# CNN 对比实验（std vs CBAM）
cd article_implement/CNN
python train_snr_sp.py --device cuda --model cbam --compare --epochs 150
```

### 测试与评估
```powershell
# MUSIC 基线测试
python test/test_file/opening_music_only.py --M 8 --k 3 --snap 200 --snrs -20 -15 -10 -5 0

# 按 SNR 评测
python test/test_file/tests_snr.py
```

### 数据生成
```powershell
cd data_creater
python Create_k_source_dataset.py
```

## 代码架构

### 核心数据流
```
ULA_dataset.Create_DOA_data()
  → 生成 y_t, SCM, 空间谱 SP, DOA 标签
  → DataLoader
  → 模型训练/测试
```

### 模型层次
- `Grid_Based_network`：基类，提供空间谱到 DOA 的转换（`sp_to_doa`/`grid_to_theta`）
- `VisionTransformer`：继承 Grid_Based_network，用于 DOA 估计
- CNN 变体：`std_CNN`（基线）、`std_CNN_SE`（通道注意力）、`std_CNN_CBAM`（通道+空间注意力）

### 输入类型
模型支持多种输入格式（通过 `--input_type` 参数）：
- `ori_scm`：原始采样协方差矩阵
- `enhance_scm`：增强 SCM（实部+虚部拼接）
- `spatial_sp`：空间谱

### 关键工具
- `utils/doa_train_and_test.py`：训练/评估循环封装
- `utils/early_stop.py`：EarlyStopping
- `utils/batch_matrix_operator.py`：批量矩阵运算（张量化加速）
- `models/dl_model/grid_based_network.py`：空间谱→DOA 峰值搜索

## 目录结构（按功能）

| 目录 | 功能 |
|------|------|
| `data_creater/` | 仿真数据生成（ULA/UCA、SNR、快拍数扫描） |
| `models/` | 算法实现：`subspace_model/`（MUSIC/ESPRIT）、`dl_model/`（CNN/ViT） |
| `train/` | 训练脚本入口 |
| `test/` | 评测脚本（非单元测试） |
| `article_implement/` | 论文方法复现（SPE-CNN, ASL, SubspaceNet, Learning-SPICE） |
| `vit_tranfer_learning/` | 迁移学习实验 |
| `matlab_post_process/` | MATLAB 后处理脚本 |

## 注意事项

1. **路径问题**：脚本使用相对路径（`results/`、`data/`），建议从脚本所在目录运行
2. **MATLAB 联动**：`l1-SVD` 测试 SNR 时用 `python_call_l1_SVD_omp_plus.m`，测试快拍数时用 `python_call_l1_SVD_snap.m`
3. **GPU 兼容性**：新显卡（如 RTX 50 系）可能需要 nightly PyTorch
4. **PowerShell 续行**：使用反引号 `` ` ``，不是 `^`

## 预训练权重

HuggingFace：https://huggingface.co/zbb2025/DOA_data_and_results/tree/master
