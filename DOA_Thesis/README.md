# DOA Thesis Workspace

This directory keeps the thesis materials separate from runnable code.

Working title:

> 低信噪比条件下基于坐标注意力网络的 DOA 估计方法研究

Core line:

> 针对低信噪比条件下传统 DOA 估计方法性能下降的问题，本文基于样本协方差矩阵构造四通道输入特征，引入坐标注意力机制增强协方差矩阵行、列方向的阵元相关结构特征，最终通过空间谱输出和峰值搜索实现多信源 DOA 估计。

## Directory Map

| Directory | Purpose |
| --- | --- |
| `00_literature/` | Literature table and reading cards. |
| `01_thesis_notes/` | Problem definition, method frame, experiment plan, terms. |
| `02_code/` | Pointer to the active implementation workspace. |
| `03_results/` | Curated thesis results copied from code runs. |
| `04_figures/` | Thesis figures, diagrams, and result curves. |
| `05_thesis_draft/` | Chapter-level draft skeletons. |

## Current Acceptance Checklist

- [ ] Problem definition fixed.
- [ ] Method chapter draft covers four-channel input, coordinate attention, CA-DOA-Net, spectrum output, and loss.
- [ ] Main SNR experiment finished.
- [ ] CNN vs CNN+CA ablation finished.
- [ ] 2-channel vs 4-channel ablation finished.
- [ ] MUSIC/ESPRIT comparison figures exported.
- [ ] Literature table filled with verified bibliographic metadata.

