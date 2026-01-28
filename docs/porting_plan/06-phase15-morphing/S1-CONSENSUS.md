# Phase 15: Source Estimate Morphing - S1 Consensus

## 1. 需求共识
确认实现源空间 Morphing 功能，支持从文件读取 Morph Map 并应用到源估计 (SourceEstimate) 上。

### 验收标准
1.  **Morph Map I/O**:
    *   能够从 `.fif` 文件中读取 Morph Map 数据 (通常包含 `MNE_MORPH_MAP` 标签)。
    *   `MneMorphMap` 类能够正确解析稀疏矩阵。
2.  **Morphing 操作**:
    *   实现 `morph_source_estimate` 函数（或类似方法）。
    *   输入：`MNESourceEstimate` (Subject A), `MneMorphMap`。
    *   输出：`MNESourceEstimate` (Subject B)。
    *   验证：变换后的 STC 数据维度应与 Subject B 的源空间一致。
3.  **集成测试**:
    *   使用测试数据验证 Morphing 过程是否跑通。

## 2. 技术方案概要

### MneMorphMap 类增强
目前 `MneMorphMap` 仅是一个数据容器。需要增加：
*   `static MneMorphMap::read_morph_map(const QString& subject_from, const QString& subject_to, const QString& subjects_dir)`: 查找并读取 morph map 文件。
*   `Eigen::SparseMatrix<double> toEigen() const`: 将内部的 `FiffSparseMatrix` 转换为 Eigen 稀疏矩阵以便计算。

### 稀疏矩阵转换
`FiffSparseMatrix` 存储的是原始的 CRS/CCS 格式 (`data`, `inds`, `ptrs`)。
需要编写转换逻辑：
```cpp
Eigen::SparseMatrix<double> mat(m, n);
// Fill from data, inds, ptrs based on 'coding' (FIFFTS_MC_CCS or FIFFTS_MC_RCS)
```

### Morphing 逻辑
```cpp
MNESourceEstimate morph(const MNESourceEstimate& stc, const MneMorphMap& mm) {
    Eigen::SparseMatrix<double> morph_mat = mm.toEigen();
    Eigen::MatrixXd data_to = morph_mat * stc.data;
    // Create new STC with data_to and subject_to's vertices
}
```

## 3. 风险与约束
*   **文件格式**: 需确认 MNE-Python 生成的 morph map `.fif` 文件的 tag 结构。通常是 `FIFFB_MNE_MORPH_MAP` block。
*   **Helper Functions**: `mne_morph_map.cpp` 目前为空，需要大量实现。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
