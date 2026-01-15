# Phase 11: Advanced Time-Frequency (Multitaper & CSD) - S1 Alignment

## 1. 项目背景
Phase 11 旨在增强 `libraries/tfr` 库，引入高级时频分析方法，特别是 **Multitaper** (多锥度) 分析和 **CSD** (互谱密度) 计算。这些是实现 DICS (Dynamic Imaging of Coherent Sources) 波束形成的必要前置条件。

## 2. 任务目标
### 核心需求
1.  **DPSS 窗函数**: 实现 Slepian 序列 (Discrete Prolate Spheroidal Sequences) 生成算法。
2.  **Multitaper PSD**: 实现基于多锥度的功率谱密度估计 (`psd_multitaper`)。
3.  **CSD 计算**: 实现互谱密度矩阵计算 (`compute_csd`)，支持 Multitaper 模式。

### 对应 Python 功能
*   `mne.time_frequency.dpss_windows`
*   `mne.time_frequency.psd_multitaper` (or `Spectrum.compute_psd(method='multitaper')`)
*   `mne.time_frequency.csd_multitaper` / `CrossSpectralDensity`

## 3. 需求理解与边界确认

### 3.1 现有架构分析
*   `TFRLIB` (`libraries/tfr`) 包含 `PSD` (Welch) 和 `TFRUtils` (Morlet)。
*   依赖 `Eigen` 进行矩阵运算。
*   依赖 `FFTW` (如果可用) 或 `Eigen` 自带的 FFT (如果有) 或其他 FFT 库。
    *   *Check*: `mne_utils` 或 `tfr` 是否已经集成了 FFTW？
    *   `src/libraries/tfr/CMakeLists.txt` 可能链接了 FFTW。需确认。

### 3.2 边界限制
*   **输入**: Epochs 数据 (Eigen::MatrixXd or vector of MatrixXd).
*   **输出**: PSD (Freqs x Channels), CSD (Freqs x Channels x Channels).
*   **不包含**:
    *   Stockwell 变换。
    *   复杂的自适应加权 (Adaptive weighting) - 初始版本可仅实现均匀加权或简单的特征值加权。

## 4. 关键决策点

### Q1: FFT 实现
`tfr` 库目前使用什么进行 FFT？
*   查看 `src/libraries/tfr/CMakeLists.txt` 和 `tfr_compute.cpp`。
*   如果已有 FFTW 封装，直接复用。如果没有，可能需要引入或使用 Eigen Unsupported FFT。
*   **决策**: 检查 `tfr` 的依赖。

### Q2: DPSS 算法
DPSS 计算比较复杂（求解三对角矩阵特征值）。
*   **方案**: 移植 `scipy.signal.windows.dpss` 或类似的 C++ 实现。
*   核心是求解特定的特征值问题。

### Q3: CSD 数据结构
MNE-Python 中 `CrossSpectralDensity` 是一个类，存储 `data` (n_freqs, n_series, n_series)。
*   **决策**: 在 C++ 中创建一个 `CSD` 类，内部持有 `std::vector<Eigen::MatrixXcd>` (每个频率一个复数矩阵) 或 `Eigen::Tensor` (如果引入 Tensor 模块，但通常避免引入新依赖)。
*   建议使用 `std::vector<Eigen::MatrixXcd>`，index 为频率索引。

## 5. 下一步计划
1.  检查 FFT 依赖。
2.  编写 S1 Consensus。
