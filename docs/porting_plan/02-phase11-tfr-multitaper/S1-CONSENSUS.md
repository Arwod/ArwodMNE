# Phase 11: Advanced Time-Frequency (Multitaper & CSD) - S1 Consensus

## 1. 需求共识
确认在 `libraries/tfr` 中实现 DPSS 窗函数、Multitaper PSD 和 CSD 计算。

### 验收标准
1.  **DPSS 实现**:
    *   `TFRUtils::dpss_windows(N, NW, Kmax)` 能生成正交的 Slepian 序列。
    *   验证正交性 (dot product ~ 0 or 1)。
2.  **Multitaper PSD**:
    *   `PSD::psd_multitaper` 能计算 PSD。
    *   支持自适应加权 (Adaptive Weighting) 或简单的均值。
3.  **CSD 计算**:
    *   创建 `CSD` 类。
    *   `CSD::compute_csd` 能计算互谱密度。
    *   结果维度正确 (Freqs x Channels x Channels)。

## 2. 技术方案概要
*   **FFT**: 使用 `unsupported/Eigen/FFT`，与现有 `MNEMath` 保持一致。
*   **DPSS**: 使用 Slepian (1978) 描述的三对角矩阵法求解特征向量。使用 `Eigen::SelfAdjointEigenSolver` 求解。
*   **CSD**:
    *   对每个 Taper 计算 FFT。
    *   计算 Cross-Product: $X_i(f) X_j^*(f)$。
    *   在 Tapers 间平均 (加权)。
    *   在 Epochs 间平均。

## 3. 风险与约束
*   **性能**: 大量通道的 CSD 计算量大 (Ch^2)。需注意内存和循环优化。
*   **DPSS 精度**: 三对角矩阵法通常精度足够，但在极端参数下可能需要插值法（暂不考虑）。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
