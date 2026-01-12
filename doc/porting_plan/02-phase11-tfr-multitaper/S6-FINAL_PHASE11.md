# Phase 11: Advanced Time-Frequency (Multitaper & CSD) - Final Report

## 项目总结
本项目成功在 `libraries/tfr` 中引入了高级时频分析功能，重点实现了多锥度 (Multitaper) 分析框架和互谱密度 (CSD) 计算。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/tfr/tfr_utils.cpp`: 新增 `dpss_windows`。
    *   `src/libraries/tfr/psd.cpp`: 新增 `psd_multitaper`。
    *   `src/libraries/tfr/csd.h/cpp`: 新增 `CSD` 类。
2.  **测试代码**:
    *   `src/testframes/test_tfr_multitaper`: 全面的单元测试。
3.  **文档**:
    *   `doc/porting_plan/02-phase11-tfr-multitaper/*`.

## 技术亮点
*   **DPSS 生成**: 实现了基于三对角矩阵特征值分解的高效算法 (Slepian 1978)，避免了复杂的数值积分。
*   **CSD 架构**: 设计了通用的 `CSD` 类，支持多频率、多通道数据的存储和访问，接口设计参考了 MNE-Python。

## 后续建议
*   **Phase 12**: 利用 `CSD` 类实现 DICS 波束形成。
*   **优化**: 目前 `compute_multitaper` 是单线程的，对于大通道数数据，可以利用 OpenMP 或 QtConcurrent 并行化。
