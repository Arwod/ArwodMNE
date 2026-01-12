# Phase 15: Source Estimate Morphing - S6 Acceptance

## 验收结果

### 1. 功能完整性
*   [x] **Task 15.1 Enhance FiffSparseMatrix**:
    *   在 `FiffSparseMatrix` 中实现了 `toEigenSparse()`，支持 CRS/CCS 格式到 `Eigen::SparseMatrix` 的转换。
*   [x] **Task 15.2 Read MorphMap**:
    *   在 `MneMorphMap` 中实现了 `readMorphMap`，能够从 `.fif` 文件中解析 Morph Map。
    *   支持自动查找文件路径（`subjects_dir/morph-maps/`）。
*   [x] **Task 15.3 Morph Logic**:
    *   实现了 `MneMorphMap::morphSourceEstimate`。
    *   实现了基于 Eigen 稀疏矩阵乘法的 Morphing 逻辑。
    *   支持 LH/RH 自动分割与合并。
*   [x] **Task 15.4 Verification**:
    *   `test_morphing` 测试通过，验证了稀疏矩阵转换和 Morphing 计算逻辑的正确性。

### 2. 代码质量
*   **重用性**: 复用了 `FiffSparseMatrix` 和 `FiffStream`。
*   **健壮性**: 增加了对文件路径和文件内容的检查。
*   **兼容性**: 保持了与 Eigen 的良好集成，便于后续科学计算。

## 结论
Phase 15 任务完成。现在 MNE-CPP 具备了将源空间数据映射到标准空间的能力，为 Phase 16 的组统计分析扫清了障碍。
