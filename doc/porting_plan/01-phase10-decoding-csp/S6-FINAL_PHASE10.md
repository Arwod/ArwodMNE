# Phase 10: Decoding Foundation (CSP) - Final Report

## 项目总结
本项目成功为 MNE-CPP 添加了 `decoding` 库，并实现了 BCI 领域最核心的算法之一：CSP (Common Spatial Patterns)。这标志着 MNE-CPP 从单纯的信号处理和源定位库，向脑机接口（BCI）和机器学习领域迈出了关键一步。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/decoding/csp.h/cpp`: CSP 算法实现。
    *   `src/libraries/decoding/decoding_global.h/cpp`: 库基础设施。
    *   `src/libraries/decoding/CMakeLists.txt`: 构建脚本。
2.  **测试代码**:
    *   `src/testframes/test_decoding/test_decoding.cpp`: 验证测试。
3.  **文档**:
    *   `docs/01-phase10-decoding-csp/*`: 全套 6A 工作流文档。

## 技术亮点
*   **Eigen 集成**: 充分利用 Eigen 的 `GeneralizedSelfAdjointEigenSolver` 进行数值稳定的特征值分解。
*   **接口对齐**: API 设计参考了 `mne-python` 和 `sklearn` 的风格 (`fit`, `transform`)，降低了 Python 用户的迁移成本。
*   **轻量化**: 依赖最小化，仅依赖 `mne_utils` 和 `Eigen`。

## 后续建议
*   在 `mne_scan` 或其他应用中集成 CSP 插件，实现实时 BCI。
*   扩展 CSP 支持多分类（One-vs-Rest）。
*   添加正则化 CSP 支持以处理协方差估计不准的情况。
