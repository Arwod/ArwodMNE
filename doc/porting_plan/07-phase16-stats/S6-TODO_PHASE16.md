# Phase 16: Non-Parametric Statistics - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **Spatio-Temporal Adjacency**:
    *   目前用户需手动构建 2D/3D 邻接矩阵。
    *   **Action**: 在 `mne` 库中添加工具函数，根据 Source Space 和时间点数自动生成 Kronecker product 形式的时空邻接矩阵。

2.  **TFCE (Threshold-Free Cluster Enhancement)**:
    *   目前仅支持基于阈值的聚类。
    *   **Action**: 实现 TFCE 算法，避免手动选择阈值。

3.  **F-test / ANOVA**:
    *   目前仅实现了 T-test (1-sample, indep)。
    *   **Action**: 添加 `f_oneway` 或 `repeated_measures_anova` 以支持多组比较。

4.  **Permutation Optimizations**:
    *   对于小样本，可以枚举所有排列而不是随机抽样。
    *   **Action**: 检测样本量，若 $2^N$ 较小，使用精确检验。
