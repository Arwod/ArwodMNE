# Phase 15: Source Estimate Morphing - Final Report

## 项目总结
本项目为 MNE-CPP 添加了源空间 Morphing 能力。通过增强 `MneMorphMap` 类和 `FiffSparseMatrix` 类，实现了从 MNE-Python/FreeSurfer 生成的 morph map 文件中读取映射矩阵，并将其应用到 `MNESourceEstimate` 对象上。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/fiff/c/fiff_sparse_matrix.h/cpp`: 增加了 `toEigenSparse`。
    *   `src/libraries/mne/c/mne_morph_map.h/cpp`: 实现了 `readMorphMap` 和 `morphSourceEstimate`。
2.  **测试代码**:
    *   `src/testframes/test_morphing`: 验证测试。
3.  **文档**:
    *   `doc/porting_plan/06-phase15-morphing/*`.

## 技术亮点
*   **稀疏矩阵集成**: 高效地将 FIFF 格式的稀疏矩阵转换为 Eigen 格式，利用 Eigen 的优化进行计算。
*   **自动化**: `morphSourceEstimate` 封装了从文件读取到应用变换的全过程，简化了用户调用。

## 后续建议
*   **Phase 16**: 利用 Morphing 后的数据进行组统计分析（Cluster-based permutation test）。
*   **实时 Morphing**: 目前依赖预计算的 `.fif` 文件。未来可考虑实现基于网格几何的实时 Morph Map 计算（需要 `MneSurface` 的 geodesic distance 等算法）。
