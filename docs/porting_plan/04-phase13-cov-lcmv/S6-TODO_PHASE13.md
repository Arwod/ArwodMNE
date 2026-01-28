# Phase 13: Covariance & LCMV Refactoring - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **自动秩估计 (Auto Rank Estimation)**:
    *   目前协方差求逆直接使用 `SelfAdjointEigenSolver` 并取逆。如果矩阵秩亏缺（例如经过 SSS 处理的数据），这会不稳定。
    *   **Action**: 实现 `compute_rank`，并在求逆时使用伪逆 (Moore-Penrose) 或截断特征值。

2.  **高级正则化**:
    *   目前仅支持简单的 Tikhonov (`reg * trace/n`).
    *   **Action**: 支持 `shrunk` (Ledoit-Wolf) 或 `auto` 正则化。

3.  **I/O 支持**:
    *   `Covariance` 类目前无法保存到磁盘。
    *   **Action**: 实现 `.fif` 格式的读写支持，以便与 MNE-Python 互通。

4.  **Beamformer 类独立**:
    *   目前 `BeamformerWeights` 只是一个结构体。
    *   **Action**: 随着功能增加（如 DICS 和 LCMV 的统一），考虑创建一个 `IBeamformer` 接口。
