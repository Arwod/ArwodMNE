# Phase 15: Source Estimate Morphing - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **Volume Source Morphing**:
    *   目前仅支持 Surface Source Estimate (`MNESourceEstimate`)。
    *   **Action**: 扩展支持 `VolSourceEstimate` (Volume-based morphing)，这通常涉及 MRI 体素的插值，机制不同。

2.  **即时计算 (On-the-fly Calculation)**:
    *   目前必须先用 Python 生成 `.fif` 文件。
    *   **Action**: 移植 `compute_source_morph` 的几何算法（基于球面配准和平滑）。

3.  **多线程优化**:
    *   Morphing 涉及稀疏矩阵乘法，对于长数据可能较慢。
    *   **Action**: 使用 OpenMP 或 QtConcurrent 加速大数据的 Morphing。

4.  **目标顶点处理**:
    *   目前的 `morphSourceEstimate` 返回的 `vertices` 只是简单的 0..N 索引。
    *   **Action**: 应该加载目标 Subject 的 SourceSpace (`src.fif`) 并正确设置 vertices ID，以便后续绘图或处理能正确对应到脑区。
