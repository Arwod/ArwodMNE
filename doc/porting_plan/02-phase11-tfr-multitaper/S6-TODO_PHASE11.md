# Phase 11: Advanced Time-Frequency (Multitaper & CSD) - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **Adaptive Weighting**:
    *   目前 `psd_multitaper` 和 `compute_multitaper` 仅实现了均匀加权。
    *   **Action**: 实现基于特征值的自适应加权 (Adaptive Weighting)，以减少宽带泄露。

2.  **并行化**:
    *   CSD 计算涉及大量的 FFT 和外积运算。
    *   **Action**: 使用 `QtConcurrent::map` 或 OpenMP 并行化 Epoch 循环。

3.  **Cross-Spectral Matrix 存储优化**:
    *   目前存储所有频率的完整矩阵。对于高密度脑电 (e.g. 256通道) 和高频分辨率，内存占用较大。
    *   **Action**: 考虑支持仅存储上三角矩阵 (Hermitian)，或者仅计算感兴趣的频率。

4.  **CSD I/O**:
    *   目前没有实现 CSD 的文件读写 (.h5 或 .fif)。
    *   **Action**: 实现 `CSD::save` 和 `CSD::read`。
