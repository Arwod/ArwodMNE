# Phase 15: Source Estimate Morphing - S2 Design

## 1. 模块架构

### 新增功能于 `MneMorphMap` (`mne/c/mne_morph_map.h`)
增强现有类以支持 I/O 和计算。

```cpp
class MNESHARED_EXPORT MneMorphMap
{
public:
    // ... existing members ...
    
    // New methods
    /**
     * @brief Read morph map from .fif file.
     * Searches for [subject_from]-[subject_to]-morph.fif in subjects_dir/morph-maps/
     */
    static bool readMorphMap(const QString& subject_from, 
                             const QString& subject_to, 
                             const QString& subjects_dir,
                             MneMorphMap::SPtr& pMorphMapLH,
                             MneMorphMap::SPtr& pMorphMapRH);

    /**
     * @brief Convert internal FiffSparseMatrix to Eigen::SparseMatrix.
     */
    Eigen::SparseMatrix<double> toEigen() const;
    
    /**
     * @brief Morph a SourceEstimate.
     * Helper function that handles splitting STC, applying maps, and merging.
     */
    static MNESourceEstimate morphSourceEstimate(const MNESourceEstimate& stc,
                                                 const QString& subject_from,
                                                 const QString& subject_to,
                                                 const QString& subjects_dir);
};
```

### 辅助类/函数
可能需要 `MneSurfaceOrVolume` 相关的几何计算（如果涉及实时计算），但 Phase 15 聚焦于 **读取预计算的 Morph Map**。

## 2. 核心算法细节

### 读取 Morph Map (.fif)
*   利用 `FiffStream` 遍历 Tag。
*   寻找 `FIFFB_MNE_MORPH_MAP` block。
*   读取 `FIFF_MNE_MORPH_MAP_FROM` 和 `FIFF_MNE_MORPH_MAP_TO` 确认 subject。
*   读取 `FIFF_MNE_MORPH_MAP` (Sparse Matrix)。
    *   通常文件中有两个 matrix，分别对应 LH 和 RH。
    *   可以通过 `FIFF_MNE_HEMI` tag 来区分吗？或者顺序？
    *   MNE-C 源码显示：通常按顺序存储，或者包含 hemisphere 信息。需要解析 Tag。

### 应用 Morphing
1.  **Split STC**: 检测 `stc.vertices` 中的索引下降点，将数据分为 LH 和 RH。
2.  **Apply Map**: 
    *   $Y_{lh} = M_{lh} \cdot X_{lh}$
    *   $Y_{rh} = M_{rh} \cdot X_{rh}$
3.  **Merge**: 
    *   $Y = [Y_{lh}; Y_{rh}]$
    *   $Vertices = [Vertices_{lh}; Vertices_{rh}]$ (来自 Morph Map 的行索引或目标 SourceSpace)。

### 稀疏矩阵转换
`FiffSparseMatrix` 到 `Eigen::SparseMatrix`:
*   CRS format: `ptr` 数组对应 `outerIndexPtr`, `inds` 对应 `innerIndexPtr`, `data` 对应 `valuePtr`.
*   Eigen 使用 CCS (缺省) 或 CRS (RowMajor)。需注意行列存储顺序。`FiffSparseMatrix` 通常是 CRS (Compressed Row Storage)。

## 3. 依赖关系
*   `mne_mne` 库依赖 `mne_fiff`。
*   需引用 `FiffStream`, `FiffTag`.

## 4. 接口变更
无破坏性变更。`MNESourceEstimate` 保持原样，仅在 `MneMorphMap` 中添加静态工具函数。
