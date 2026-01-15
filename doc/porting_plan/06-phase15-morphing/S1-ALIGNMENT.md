# Phase 15: Source Estimate Morphing - S1 Alignment

## 1. 项目背景
在 Phase 12 (DICS) 和 Phase 13 (LCMV) 中，我们实现了源定位算法，可以计算单个被试的源空间活动（Source Estimates）。然而，为了进行组分析（Group Analysis），我们需要将不同被试的源活动映射到一个共同的参考空间（通常是 `fsaverage`）。这个过程称为 "Morphing"。

## 2. 任务目标
### 核心需求
1.  **Morph Map 计算/加载**:
    *   能够读取预计算的 Morph Maps (通常由 FreeSurfer/MNE-Python 生成，存储在 `.fif` 或类似结构中)。
    *   或者，如果可行，实现即时的 Morph Map 计算（涉及网格插值，较复杂，优先考虑读取）。
2.  **应用 Morphing**:
    *   将 `MNESourceEstimate` (STC) 从 `subject_from` 变换到 `subject_to`。
    *   支持稀疏矩阵乘法：$Y = M \cdot X$，其中 $M$ 是 Morph Matrix，$X$ 是源数据。
3.  **支持数据类型**:
    *   `MNESourceEstimate` (Surface-based).
    *   (可选) `VolSourceEstimate` (Volume-based) - 暂不作为 P0。

### 对应 Python 功能
*   `mne.compute_source_morph`
*   `mne.read_source_morph`
*   `SourceMorph.apply()`

## 3. 现有代码分析
*   `src/libraries/mne/c/mne_morph_map.h/cpp` 已经存在。
*   它包含 `MneMorphMap` 类，成员包括 `FIFFLIB::FiffSparseMatrix* map`。
*   看起来这是对 MNE-C 的移植。
*   我们需要检查：
    1.  这个类是否完整可用？
    2.  是否支持从文件读取 Morph Map？(MNE-C 通常计算它并保存)
    3.  是否集成了 `MNESourceEstimate` 的变换逻辑？

## 4. 关键决策点

### Q1: 计算 vs 读取
*   **读取**: MNE-Python 通常在运行时计算 morphing (通过 `compute_source_morph`)，也可以保存。但底层的 morph map (vertices mapping) 往往依赖 FreeSurfer 的球面配准 (`lh.sphere.reg`)。
*   **MNE-CPP 现状**: `mne_morph_map` 似乎是用来存储 map 的。我们需要确认是否有计算逻辑。
*   **决策**: 优先实现 **读取** 现有的 Morph Map (如果 MNE-Python 能导出兼容格式)，或者实现基于 `sphere.reg` 的 **计算** (如果 `mne_mne` 库中已有相关几何算法)。
    *   *调查*: `mne_mne` 库中有 `mne_surface_or_volume.cpp`, `mne_surface.cpp`。需要搜索 `sphere.reg` 或 `nearest`。

### Q2: 稀疏矩阵运算
*   Morphing 本质是稀疏矩阵乘法。
*   `MneMorphMap` 使用 `FiffSparseMatrix`。需要确认该类是否支持与 `Eigen::MatrixXd` (STC data) 的乘法。

## 5. 下一步计划
1.  深入代码库 (`src/libraries/mne`)，了解现有的 Morphing 支持。
2.  编写 `S1-CONSENSUS.md`。
