# Phase 14: Connectivity (Spectral Connectivity) - S1 Alignment

## 1. 项目背景
Phase 14 旨在完善和标准化 `libraries/connectivity` 库中的谱连接性分析功能。目前该库已包含 Coherence, PLI 等实现，但存在严重的代码重复（每个指标都实现了自己的谱估计逻辑），且未利用 Phase 11 中开发的 `mne_tfr` 库的高效 DPSS 功能。

## 2. 任务目标
### 核心需求
1.  **消除代码重复**: 提取通用的谱估计（Spectral Estimation）逻辑到基类或辅助类中。
2.  **集成 mne_tfr**: 使用 `TFRLIB::TFRUtils` 替代旧的 `UTILSLIB::Spectral` 进行 DPSS 窗函数生成，确保与 Multitaper TFR 的一致性。
3.  **支持多指标**: 确保 refactor 后的架构支持：
    *   Coherence (Coh)
    *   Imaginary Coherence (ImCoh)
    *   Phase Lag Index (PLI)
    *   Phase Locking Value (PLV) (现有代码有 implementation,需确认)
    *   Weighted PLI (WPLI)
4.  **并行计算**: 保留或优化现有的 `QtConcurrent` 并行处理机制。

### 对应 Python 功能
*   `mne.connectivity.spectral_connectivity_epochs`

## 3. 需求理解与边界确认

### 3.1 现有代码分析
*   `src/libraries/connectivity/metrics/` 下有多个指标类 (`Coherency`, `PhaseLagIndex` 等)。
*   每个类都有一个几乎完全相同的 `compute` 方法，负责去均值、加窗 (Multitaper)、FFT、CSD 计算。
*   这种重复导致维护困难，且难以统一优化（如替换 FFT 库或窗函数算法）。

### 3.2 边界限制
*   **输入**: Epochs 数据 (在 `ConnectivitySettings` 中)。
*   **输出**: `Network` 对象（包含连接性矩阵）。
*   **范围**: 仅针对 "Spectral" Connectivity。

## 4. 关键决策点

### Q1: 架构设计
*   **方案 A**: 创建 `SpectralConnectivity` 基类，处理 FFT 和 CSD 累加。具体的 Metric 子类只需提供 "Accumulation Strategy" (累加什么) 和 "Finalization Strategy" (如何计算最终指标)。
*   **方案 B**: 在 `mne_tfr` 中提供一个 `EpochSpectralCompute` 类，`connectivity` 库调用它。
*   **决策**: 采用 **方案 A**。Connectivity 的累加逻辑（如 PLI 需要 sign(imag)）比较特殊，放在 Connectivity 库内部更合适。但底层工具（DPSS, FFT）应调用 `mne_tfr`。

### Q2: 依赖关系
*   `connectivity` 库需要链接 `mne_tfr`。
*   这将引入依赖：`connectivity -> mne_tfr`。这是合理的。

## 5. 下一步计划
1.  修改 `src/libraries/connectivity/CMakeLists.txt`。
2.  创建 `AbstractSpectralMetric` 类（继承自 `AbstractMetric`）。
3.  将通用逻辑移入 `AbstractSpectralMetric`。
4.  重构 `Coherence` and `PhaseLagIndex` 以继承该类。
