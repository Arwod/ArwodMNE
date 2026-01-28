# Phase 12: Frequency-Domain Beamformer (DICS) - Final Report

## 项目总结
本项目实现了 **DICS (Dynamic Imaging of Coherent Sources)** 波束形成算法，填补了 MNE-CPP 在频域源定位方面的空白。该实现基于 Phase 11 开发的 `CSD` 模块，提供了从传感器层面的互谱密度到源空间功率分布的映射能力。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/inverse/beamformer/dics.h/cpp`: DICS 核心算法。
2.  **构建脚本**:
    *   更新 `src/libraries/inverse/CMakeLists.txt`。
3.  **测试代码**:
    *   `src/testframes/test_dics/test_dics.cpp`: 验证测试。
4.  **文档**:
    *   `doc/porting_plan/03-phase12-dics-beamformer/*`.

## 技术亮点
*   **灵活的滤波器设计**: 支持基于实部 CSD (用于更稳定的方向估计) 或全复数 CSD 的滤波器计算。
*   **高效计算**: 利用 Eigen 的高级线性代数求解器处理矩阵求逆。
*   **模块化**: 清晰地分离了时频分析 (`tfr`) 和逆问题求解 (`inverse`) 模块。

## 后续建议
*   **Phase 13**: 实现 LCMV 波束形成器的完整重构（目前 LCMV 代码较旧），使其与 DICS 共享通用的 Beamformer 基类或工具函数。
*   **性能优化**: 对于全脑源空间（~5000-10000个源点），建议引入 OpenMP 或 CUDA 加速。
