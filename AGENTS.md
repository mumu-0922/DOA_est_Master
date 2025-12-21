# Repository Guidelines

回答问题都用中文，关键代码也要加中文注释。

## Project Structure & Module Organization

- `data_creater/`：仿真数据集生成（ULA/UCA、快拍数、SNR 扫描等）。
- `models/`：传统算法基线（MUSIC/ESPRIT/压缩感知）与深度模型（CNN/ViT）。
- `train/`：训练入口脚本（如 `train_snr.py`、`train_snap.py`）。
- `test/`：评测/对比脚本（偏实验脚本，不是单元测试套件）。
- `utils/`：训练与评测公用工具（early stop、loss、矩阵算子等）。
- `data_save/`：CSV 保存与绘图工具。
- `matlab_post_process/`：部分算法需要的 MATLAB 后处理脚本。
- `article_implement/`、`vit_tranfer_learning/`：论文复现与迁移学习相关实验。

## Build, Test, and Development Commands

- 创建环境：`conda env create -f environment.yml`（或 `conda env update -f environment.yml --prune`）。
- 激活环境：`conda activate DOA`。
- 建议从脚本所在目录运行：不少脚本用相对路径拼接 `results/`、`data/` 等目录。
  - 训练示例：`cd train; python train_snr.py`（脚本底部默认参数包含 `M/k/snrs/snap`）。
  - 测试示例：`cd test/test_file; python tests_snr.py`（通常会在 `results/` 下生成 CSV/图片）。
- 若缺依赖（如 `tensorboardX`），请安装并考虑同步更新 `environment.yml`。

## Coding Style & Naming Conventions

- Python：4 空格缩进；尽量保持与现有脚本风格一致、改动聚焦。
- 命名：函数/变量用 `snake_case`，类用 `PascalCase`；新文件建议 `snake_case.py`。
- 谨慎重命名/移动文件：训练/测试脚本里存在手写路径与相对路径依赖。

## Testing Guidelines

- 目前没有统一测试框架（未见 `pytest` 配置）；`test/` 主要是可复现实验脚本。
- 如需新增验证脚本，尽量保证快速、可重复，并说明输出文件（CSV/图）的位置与命名。

## Commit & Pull Request Guidelines

- 历史提交信息较短（如 `modify`、`readme_change`）；建议使用“范围 + 动作”的祈使句：`train: fix path handling`。
- PR 需说明：改动影响的场景（如 `M/k/snr/snap`）、如何运行复现，以及关键指标/图片（如适用）。
- 不要提交大型产物到 `data/` 或 `results/`；请使用外部存储链接或按需生成。

## MATLAB & External Dependencies

- 部分基线算法需要 MATLAB + Python MATLAB Engine，相关代码见 `models/compress_sensing/invoke_matlab/` 与 `matlab_post_process/`。
- 如修改了 MATLAB 调用链路，请在 PR 中补充 MATLAB 版本与本地配置要点。
