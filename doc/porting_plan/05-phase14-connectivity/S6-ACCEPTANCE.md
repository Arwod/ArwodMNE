# Phase 14: Connectivity (Spectral Connectivity) - S6 Acceptance

## 验收结果

### 1. 功能完整性
*   [x] **Task 14.1 Update CMake**: `connectivity` 库已链接 `mne_tfr`。
*   [x] **Task 14.2 AbstractSpectralMetric**: 实现了 `metrics/abstractspectralmetric.h/cpp`，提供通用的 `generateTapers` (基于 `mne_tfr`) 和 `computeTaperedSpectra`。
*   [x] **Task 14.3 Refactor Coherency**: `Coherency` 类已重构，移除了重复的 FFT 代码，使用 `AbstractSpectralMetric`。
*   [x] **Task 14.4 Refactor PLI**: `PhaseLagIndex` 类已重构，同样使用 `AbstractSpectralMetric`。
*   [x] **Task 14.5 Verification**: `test_connectivity` 通过，验证了 Coherence (Amplitude) 和 PLI (Phase) 的计算准确性。

### 2. 代码质量
*   **消除重复**: 核心谱估计逻辑现在集中在 `AbstractSpectralMetric` 中。
*   **复用性**: 利用了 Phase 11 开发的 `mne_tfr` 库，确保了窗函数生成的一致性。
*   **可维护性**: 新增指标只需关注如何累加谱统计量，无需关心 FFT 细节。

## 结论
Phase 14 任务圆满完成。`connectivity` 库的架构得到了显著优化，为后续添加更多谱连接性指标打下了良好基础。
