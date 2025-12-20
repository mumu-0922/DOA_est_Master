# A Generic DOA deep learning framework v1

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu118-red.svg)](https://pytorch.org/)    [![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


Most importantly, our code incorporates a universal deep learning framework for DOA estimation, capable of efficiently generating simulation data based on Uniform Linear Array (ULA), Uniform Circular Array (UCA), and other array configurations. It enables the effective extraction of raw data $y_t$, as well as the corresponding sampled covariance matrix (SCM), ideal covariance matrix, signal subspace $U_s$, target DOA values, and the spatial spectrum (SP) associated with the DOA. These data components are utilized to construct comprehensive datasets, which can be leveraged to train deep neural networks for both direct and indirect DOA estimation.

Due to the lower computational efficiency of Python in numerical operations compared to MATLAB, it is essential to adopt batch processing at the code level to enhance computational efficiency and minimize the time overhead associated with iterative loops. The array steering vector $A$, incident signals $s(t)$, and noise $n(t)$ are generated using tensorized methods to maximize efficiency. Furthermore, the processes for generating data such as the sampled covariance matrix (SCM) and spatial spectrum (SP) are fully implemented with batch processing to further improve performance.


<div align='center'>
<img src='https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png' width='100%' align=center/>
</div>

<!--[](https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png)-->

Once the dataset has been successfully generated, various deep neural networks can be efficiently designed and their performance tested. In the domain of DOA estimation, there is an expectation to design networks capable of achieving more accurate DOA estimation across diverse configurations.

2.Additionally, to compare the performance of different algorithms, we have implemented subspace-based algorithms such as MUSIC, Root MUSIC, ESPRIT, and Unity ESPRIT, as well as compressed sensing algorithms like $\ell_1$-SVD and SPICE, and deep learning-based algorithms such as SPE-CNN, ASL and Learning-SPICE within our code framework.

3.Furthermore, this repository includes the implementation of the methods presented in the paper [A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections](文章的URL). The significant contributions of our work can be outlined as follows.

* We propose a novel DOA estimation model based on Vision Transformer, which demonstrates exceptional performance under challenging conditions, such as low SNR and small snapshot scenarios.
* Due to the presence of array imperfections, the data distributions of the source and target domains differ significantly, leading to substantial performance degradation when the model is deployed in practical scenarios. To address this issue, we introduce a transfer learning algorithm to align the features between the source and target domains, enhancing the model's performance in practical scenarios.
* Via extensive simulation, we compare the proposed method with existing approaches across multiple evaluation metrics and demonstrate the superiority of our method in terms of DOA estimation accuracy and robustness.

All codes for simulation and plotting for results presented in the paper are available in this repository. We encourage the use of this code and look forward to further improvements and contributions.
## Citation Information

If the code or methods provided in this project have been helpful for your research or work, please cite the following reference:

> **A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections**  
> Authors: Bo Zhou, Kaijie Xu, Yinghui Quan, Mengdao Xing  
> Journal/Conference: To be determined  
> DOI: To be determined

### BibTeX Citation Format
```bibtex
@article{zhou2024doa,
  title     = {A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections},
  author    = {Bo Zhou, Kaijie Xu, Yinghui Quan and Mengdao Xing},
  journal   = {To be determined},
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {To be determined}
}
```

## Code Usage Notes

**1. Prerequisites** 
Make sure you have installed the required dependencies which is listed in the `environment.yml` file.

Make sure you have installed MATLAB and Python, along with the tools required to call MATLAB functions from Python. For more details, refer to: [https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
  
**2. Root Directory Set**
You need to confirm that the root directory is correct. The project root directory should be added to Python's search path to correctly import function packages. Additionally, Python's working directory should be set to the current file directory to ensure correct paths for reading and saving models and data.

These configurations are well set by PyCharm. However, if you're using VSCode or other IDEs, there may be a need for additional settings. For vscode, this may can be achieved by adding the following content to `./.vscode/settings.json`:
  ```
  {
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "${workspaceFolder}"
  }
  "python.terminal.executeInFileDir": true, 
  "code-runner.fileDirectoryAsCwd": true, 
  "terminal.integrated.cwd": "${fileDirname}"
}
  ```

**3. implementation of algorithms**
This repository contains the implementation of various algorithms, all algorithms are implemented through **Python** and **MATLAB**. Because some algorithms require joint execution of MATLAB and Python, you need to carefully adjust the directory and certain code before running it.
all test files are in the test/ directory.
- $\ell_1$-SVD algorithm is implemented in the file **l1_svd.py**, which invokes two files: python_call_l1_SVD_omp_plus.m and python_call_l1_SVD_snap.m. When testing snap variations, you should use python_call_l1_SVD_snap in matlab_l1_svd.predict and comment out python_call_l1_SVD_omp_plus. When testing SNR variations, use python_call_l1_SVD_omp_plus. Otherwise, an error will be triggered.
- Some of our models, such as *Learning-SPICE* and *ASL-2*, require MATLAB for compressed sensing post-processing. After the deep learning-based pre-processing is completed, the corresponding MATLAB code should be executed to obtain the final results.
- When running the training or testing scripts, please ensure that the paths for loading models, loading data, and saving results are correct to ensure smooth code execution.

**4. Available Weights**
All files generated during the execution of our code are uploaded to huggingface: https://huggingface.co/zbb2025/DOA_data_and_results/tree/master.


If you encounter any other problems, you can submit an issue, and I will try to resolve it.
## Description of Files

- **data_creater/**  
The file **data_creater** contains modules for dataset generation and management, primarily consisting of three parts:
  - `signal_datasets/`: Core modules for data generation, storage, and loading operations
  - `create_*_data/`: Scripts for generating angle sets under various configurations
  - `Other Files/`:  Implements additional functionalities

- **article_implement/**  
  Our implementation of state-of-the-art methods, including *SPE-CNN*, *ASL-2*, *SubspaceNet*, and *Learning-SPICE*.

- **models/**  
  Contains the definitions and related code for deep learning models.

- **utils/**  
  Includes various script files for data preprocessing, model training, and evaluation.

- **train/**  
  Contains training scripts for training the model under various SNRs and snapshots conditions.

- **test/**  
  Contains testing scripts, which evaluate our proposed model and compare it with other algorithms, generating loss curves and various evaluation metrics for visualization.

- **matlab_post_process/**  
  Contains MATLAB post-processing scripts that require execution in MATLAB.

- **results/**  
  Stores the model weights and test results.

- **data/**  
  Stores various testing datasets.

- **data_save/**  
  Contains plotting scripts and the final aggregated data.

- **vit_transfer_learning/**  
  Contains the implementation of our proposed transfer learning algorithms.

- **article implement/**  
  Our code implements the methods that presented in the paper.

- **environment.yml**  
  Lists the dependencies and their required versions for the project.

- **README.md**  
  Provides an overview of the project, including usage instructions and guidelines.


---

## 中文版说明（保留英文原文）

本仓库提供一个通用的 DOA（到达角）深度学习框架，重点在于**高效生成仿真数据**并支持多类算法对比。框架可针对均匀线阵（ULA）、均匀圆阵（UCA）等阵列配置批量生成数据，并提取原始数据 $y_t$、采样协方差矩阵（SCM）、理想协方差矩阵、信号子空间 $U_s$、目标 DOA 以及与 DOA 对应的空间谱（SP）。这些数据可用于构建训练数据集，从而训练直接/间接 DOA 估计的深度神经网络。

由于 Python 在数值运算方面相较 MATLAB 可能存在效率劣势，本项目在代码层面强调**批处理（batch）/张量化**实现，以降低循环带来的时间开销。阵列导向矢量 $A$、入射信号 $s(t)$ 与噪声 $n(t)$ 等均采用张量化生成；SCM 与 SP 等数据也以批处理方式生成以提升性能。

数据集生成完成后，可进一步设计并测试多种深度网络。在 DOA 估计任务中，我们希望网络在不同配置下都能获得更高的精度。

此外，为了对比不同算法的性能，本仓库实现了多类方法：子空间类算法（MUSIC、Root MUSIC、ESPRIT、Unity ESPRIT）、压缩感知类算法（如 $\ell_1$-SVD、SPICE），以及深度学习相关算法（如 SPE-CNN、ASL、Learning-SPICE）。

同时，本仓库包含论文 [A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections](文章的URL) 中方法的实现。主要贡献如下：

* 提出基于 Vision Transformer 的 DOA 估计模型，在低 SNR、小快拍数等苛刻条件下表现优异。
* 针对阵列误差导致的源域/目标域分布差异及实际部署性能下降问题，引入监督迁移学习方法对齐特征以提升实际场景性能。
* 通过大量仿真，从多个评估指标对比现有方法，验证了所提方法在精度与鲁棒性方面的优势。

论文中的仿真与绘图代码均包含在本仓库中，欢迎使用并参与改进。

## 引用信息（Citation Information）

如果本项目的代码或方法对你的研究/工作有帮助，请引用如下论文：

> **A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections**  
> Authors: Bo Zhou, Kaijie Xu, Yinghui Quan, Mengdao Xing  
> Journal/Conference: To be determined  
> DOI: To be determined

### BibTeX 引用格式（BibTeX Citation Format）

（保持与英文部分一致）

## 代码使用说明（Code Usage Notes）

**1. 环境依赖（Prerequisites）**  
请先安装 `environment.yml` 中列出的依赖。

如需运行 MATLAB 相关算法，请确保已安装 MATLAB 与 Python，并完成 Python 调用 MATLAB 所需工具（MATLAB Engine for Python）的安装。参考：  
https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

**2. 根目录与工作目录设置（Root Directory Set）**  
请确认项目根目录设置正确：需要将项目根目录加入 Python 搜索路径（`PYTHONPATH`）以便正确导入包；同时建议将 Python 的工作目录设置为当前脚本文件所在目录，以保证模型/数据的读取与保存路径正确。

PyCharm 通常会自动处理这些设置；如使用 VSCode 等 IDE，可在 `./.vscode/settings.json` 中参考英文部分示例进行配置。

**3. 算法实现说明（implementation of algorithms）**  
本仓库同时包含 **Python** 与 **MATLAB** 的算法实现。由于部分算法需要 Python 与 MATLAB 联合执行，运行前可能需要手动调整目录与部分代码。

测试脚本集中在 `test/` 目录：
- $\ell_1$-SVD 位于 **l1_svd.py**，并会调用 `python_call_l1_SVD_omp_plus.m` 与 `python_call_l1_SVD_snap.m`。测试快拍数变化时，在 `matlab_l1_svd.predict` 中应使用 `python_call_l1_SVD_snap` 并注释 `python_call_l1_SVD_omp_plus`；测试 SNR 变化时则使用 `python_call_l1_SVD_omp_plus`，否则可能报错。
- 部分模型（如 *Learning-SPICE*、*ASL-2*）需要 MATLAB 做压缩感知后处理：深度学习预处理完成后，需要继续执行对应 MATLAB 代码以获得最终结果。
- 运行训练/测试脚本前，请确认模型加载路径、数据加载路径与结果保存路径设置正确。

**4. 可用权重（Available Weights）**  
代码运行生成的文件已上传至 HuggingFace：  
https://huggingface.co/zbb2025/DOA_data_and_results/tree/master

如遇到问题，可提交 issue，我会尽量协助解决。

## 文件说明（Description of Files）

- **data_creater/**  
包含数据集生成与管理模块，主要包括：
  - `signal_datasets/`：数据生成、存储与加载核心模块
  - `create_*_data/`：生成不同配置角度集合的脚本
  - `Other Files/`：其他辅助功能

- **article_implement/**  
论文/方法实现集合，包括 *SPE-CNN*、*ASL-2*、*SubspaceNet*、*Learning-SPICE* 等。

- **models/**  
深度学习模型与传统算法相关定义代码。

- **utils/**  
数据预处理、训练与评估相关工具脚本。

- **train/**  
训练脚本（覆盖不同 SNR、快拍数等条件）。

- **test/**  
测试与对比脚本：评估所提模型并与其他算法对比，生成 loss 曲线与评估指标图。

- **matlab_post_process/**  
需要在 MATLAB 中执行的后处理脚本。

- **results/**  
保存模型权重与测试结果。

- **data/**  
保存测试数据集。

- **data_save/**  
绘图脚本与最终汇总数据相关代码。

- **vit_transfer_learning/**  
迁移学习算法实现。

- **article implement/**  
论文方法实现相关代码（目录名以仓库实际为准）。

- **environment.yml**  
项目依赖与版本。

- **README.md**  
项目概览与使用说明（本仓库实际文件为 `README.markdown`）。

