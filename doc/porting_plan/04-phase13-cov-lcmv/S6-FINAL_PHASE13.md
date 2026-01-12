# Phase 13: Covariance & LCMV Refactoring - Final Report

## 项目总结
本项目对 `libraries/inverse` 中的协方差处理 (`Covariance`) 和 LCMV 波束形成器进行了深度重构。现在的实现不再是简单的静态工具函数，而是具备了状态管理和高级配置能力的类结构。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/inverse/beamformer/covariance.h/cpp`: 全新的 Covariance 类。
    *   `src/libraries/inverse/beamformer/lcmv.h/cpp`: 增强的 LCMV 类，支持 Max-Power 方向选择和 Unit-Noise-Gain 归一化。
2.  **测试代码**:
    *   `src/testframes/test_lcmv_cov`: 验证测试。
3.  **文档**:
    *   `doc/porting_plan/04-phase13-cov-lcmv/*`.

## 技术亮点
*   **API 设计**: `make_lcmv` 和 `apply` 的分离使得计算开销大的滤波器只需计算一次即可多次应用。
*   **数值计算**: 在 `Covariance::compute_empirical` 中实现了 Epochs 数据的自动中心化和 DoF 处理。
*   **方向优化**: 实现了基于特征值分解的 `max-power` 方向选择，这是现代波束形成的标配功能。

## 后续建议
*   **Phase 14**: 考虑引入 `Whitening` 类，将 Whitening 逻辑从 LCMV 中剥离，使其更通用。
*   **Rank Estimation**: 目前 Covariance 正则化依赖简单的缩放。未来应引入自动秩估计 (Auto Rank) 以处理秩亏缺数据（如 MaxFilter 后的数据）。
