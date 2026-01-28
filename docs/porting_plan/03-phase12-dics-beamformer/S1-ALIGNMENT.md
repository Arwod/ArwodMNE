# Phase 12: Frequency-Domain Beamformer (DICS) - S1 Alignment

## 1. 项目背景
Phase 12 旨在引入 **DICS** (Dynamic Imaging of Coherent Sources) 波束形成算法。DICS 是一种频域波束形成技术，利用互谱密度 (CSD) 矩阵来计算源的空间滤波器，从而估计特定频率下的源功率和相干性。

## 2. 任务目标
### 核心需求
1.  **DICS 类**: 在 `libraries/inverse` 中实现 `DICS` 类。
2.  **空间滤波器计算**:
    *   输入: `TFRLIB::CSD` (数据互谱密度), `MNEForwardSolution` (前向解)。
    *   输出: 空间滤波器权重 (Weights)。
    *   算法: $W(f) = (L^T C(f)^{-1} L)^{-1} L^T C(f)^{-1}$。
3.  **源功率估计**:
    *   应用滤波器计算源功率: $P(f) = W(f) C(f) W(f)^{*T}$。
    *   如果是 DICS，通常我们只关心对角线元素（功率）。
4.  **支持实部/复部**: DICS 通常使用实部 CSD 进行滤波器计算（以获得稳定的方向），或者复部（用于相干性）。需确认 MNE-Python 的默认行为。
    *   MNE-Python 默认使用 `real_filter=False` (使用复数 CSD 计算滤波器) 还是 `True`？
    *   通常 DICS 使用实部 CSD 来确保滤波器权重是实数（如果需要），或者复数权重。
    *   *Check*: MNE-Python 文档说 "Calculate the DICS spatial filter ... If real_filter is True, take only the real part of the cross-spectral density matrices to compute the filters."
    *   我们将提供 `real_filter` 选项。

## 3. 需求理解与边界确认

### 3.1 依赖分析
*   `TFRLIB` (`libraries/tfr`): 提供 `CSD` 类。
*   `MNELIB` (`libraries/mne`): 提供 `MNEForwardSolution` (需要确认是否有现成的 C++ 类，或者只是 MatrixXd leadfield)。
    *   查看 `mne_fiff` 或 `fs` 库中是否有 Forward 结构。
    *   为简化，本次实现可以接受 `Leadfield Matrix` (n_channels x n_sources) 和 `Source Locations`，而不是完整的 `ForwardSolution` 对象，以降低耦合。但为了长远，如果有 `MNEForwardSolution` 类最好复用。
    *   查看 `src/libraries/inverse/beamformer/lcmv.h`，它接受 `Eigen::MatrixXd leadfield`。我们将遵循此模式，接受 Leadfield 矩阵。

### 3.2 边界限制
*   **输入**: Leadfield Matrix, CSD Object, Regularization parameter.
*   **输出**: Source Power (n_sources x n_freqs).
*   **不包含**:
    *   完整的源相干性图 (Source-Source Coherence)，因为计算量巨大。
    *   复杂的 Label/ROI 级别的相干性。
    *   只实现 **Source Power Mapping**。

## 4. 关键决策点

### Q1: 正则化
*   DICS 需要对 CSD 矩阵求逆。通常使用 Tikhonov 正则化: $C_{reg} = C + \lambda I$。
*   $\lambda = \text{reg} \times \text{trace}(C) / n$。
*   **决策**: 实现简单的 Tikhonov 正则化。

### Q2: 滤波器类型
*   支持 "vector" (每个源点 3 个方向) 和 "scalar" (最大功率方向)。
*   **决策**: 初始版本支持 "vector" (返回 3xN_chan 滤波器) 和 "normal" (如果 Leadfield 限制在法向)。暂不实现 "max-power" 优化（需要特征值分解）。

### Q3: 频率处理
*   CSD 包含多个频率。DICS 滤波器通常是针对每个频率（或频带）独立计算的。
*   **决策**: `compute_source_power` 将遍历 CSD 中的每个频率，分别计算滤波器和功率。

## 5. 下一步计划
1.  确认 Leadfield 输入格式。
2.  编写 Consensus 文档。
