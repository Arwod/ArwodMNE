# Phase 14: Connectivity (Spectral Connectivity) - Final Report

## 项目总结
本项目对 `libraries/connectivity` 进行了重构，旨在消除代码冗余并集成 Phase 11 开发的 `mne_tfr` 库。通过引入 `AbstractSpectralMetric` 基类，我们将分散在各个连接性指标（如 Coherence, PLI）中的谱估计逻辑（加窗、FFT）进行了统一。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/connectivity/metrics/abstractspectralmetric.h/cpp`: 新增基类。
    *   `src/libraries/connectivity/metrics/coherency.h/cpp`: 重构后的 Coherency。
    *   `src/libraries/connectivity/metrics/phaselagindex.h/cpp`: 重构后的 PLI。
2.  **构建脚本**:
    *   更新 `src/libraries/connectivity/CMakeLists.txt`。
3.  **测试代码**:
    *   `src/testframes/test_connectivity`: 验证测试。
4.  **文档**:
    *   `doc/porting_plan/05-phase14-connectivity/*`.

## 技术亮点
*   **统一的谱估计**: 所有基于谱的连接性指标现在共享同一套 FFT 和 Tapering 逻辑。
*   **DPSS 集成**: 默认支持多锥度 (Multitaper) 分析，利用 `mne_tfr` 的高效实现。
*   **并行计算保留**: 重构过程中保留了 `QtConcurrent` 的并行加速机制。

## 后续建议
*   **Phase 15**: 将其他谱指标（如 PLV, WPLI, wPLI2_debiased）也迁移到 `AbstractSpectralMetric`。
*   **Time-Frequency Connectivity**: 目前只支持静态谱连接性（Epochs level）。未来可扩展支持时频连接性（TFR Connectivity），这需要计算每个时间点的 CSD。
