# Phase 14: Connectivity (Spectral Connectivity) - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **迁移剩余指标**:
    *   目前仅迁移了 `Coherency` (Coh, ImCoh) 和 `PhaseLagIndex` (PLI)。
    *   **Action**: 将 `PhaseLockingValue`, `WeightedPhaseLagIndex`, `DebiasedSquaredWeightedPhaseLagIndex` 等也迁移到 `AbstractSpectralMetric`。

2.  **时频连接性 (TFR Connectivity)**:
    *   目前的实现是对整个 Epoch 计算 FFT，得到的是静态频谱连接性。
    *   **Action**: 实现基于 Wavelet 或 TFR 结果的随时间变化的连接性分析。

3.  **内存优化**:
    *   目前 `computeTaperedSpectra` 返回完整的频谱矩阵。对于高密度脑电和长数据，内存占用可能较高。
    *   **Action**: 考虑分块计算或流式处理。

4.  **互谱密度 (CSD) 存储**:
    *   目前连接性计算是即时累加的，不保存中间的 CSD 矩阵（除非 StorageMode 开启且逻辑支持）。
    *   **Action**: 考虑将 `CSD` 对象（Phase 11）作为连接性分析的一等公民，允许用户先计算 CSD，再计算各种连接性指标。
