# Phase 16: Non-Parametric Statistics - S1 Alignment

## 1. 项目背景
Phase 16 旨在为 MNE-CPP 引入组分析（Group Analysis）能力。在 Phase 15 中，我们实现了源空间 Morphing，使得将不同被试的数据映射到同一参考空间成为可能。下一步是实现统计推断，特别是神经科学中广泛使用的 **基于聚类的置换检验 (Cluster-based Permutation Tests)**。

## 2. 任务目标
### 核心需求
1.  **创建 `stats` 库**:
    *   位于 `src/libraries/stats`。
    *   提供非参数统计功能。
2.  **实现核心算法**:
    *   **t-test / F-test**: 计算样本间的统计量。
    *   **Clustering**: 基于邻接关系（Adjacency）将显著点聚类。
    *   **Permutation**: 通过随机重排标签生成零分布。
3.  **支持的数据结构**:
    *   **1D (Temporal)**: 仅时间维度。
    *   **2D (Spatio-Temporal)**: 空间（顶点/通道）x 时间。需支持空间邻接矩阵（Adjacency Matrix）。
4.  **接口设计**:
    *   类似 MNE-Python 的 `permutation_cluster_test` 和 `permutation_cluster_1samp_test`。

### 对应 Python 功能
*   `mne.stats.permutation_cluster_test`
*   `mne.stats.permutation_cluster_1samp_test`
*   `mne.stats.spatio_temporal_cluster_test`

## 3. 现有代码分析
*   目前 MNE-CPP 缺乏专门的统计库。
*   `utils/mnemath.h` 可能包含基础数学运算，但不足以支持复杂的置换检验。
*   `connectivity` 库中有网络分析相关的邻接处理，可借鉴但不可直接复用。
*   `mne_mne` 库中有 `MneSurface`，可用于计算源空间的邻接矩阵。

## 4. 关键决策点

### Q1: 邻接矩阵 (Adjacency Matrix)
*   对于源空间数据，邻接关系由网格（Mesh）定义。
*   MNE-Python 使用稀疏矩阵表示邻接。
*   **决策**: `stats` 库应接受 `Eigen::SparseMatrix` 作为邻接矩阵。如何生成该矩阵（从 `MneSurface`）可能需要辅助函数，可以放在 `mne` 库或 `stats` 库的 helper 中。

### Q2: 性能
*   置换检验涉及大量重复计算（通常 1000+ 次）。
*   **决策**: 必须使用并行计算 (`QtConcurrent` 或 OpenMP)。考虑到 MNE-CPP 已广泛使用 `QtConcurrent`，优先使用之。

### Q3: 泛型设计
*   数据维度可能是 1D, 2D, 甚至 3D (TFR)。
*   **决策**: 设计一个通用的 `ClusterPermutation` 类，或者针对不同维度提供不同的函数接口。考虑到 C++ 模板的复杂性，可能先针对最常用的 Spatio-Temporal (2D) 和 Temporal (1D) 实现具体类/函数。

## 5. 下一步计划
1.  创建 `src/libraries/stats` 目录结构。
2.  编写 `S1-CONSENSUS.md`。
