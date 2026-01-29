# Phase 16: Non-Parametric Statistics - S1 Consensus

## 1. 需求共识
确认创建 `stats` 库，并实现基于聚类的置换检验。

### 验收标准
1.  **库结构**: 创建 `src/libraries/stats`。
2.  **核心类**:
    *   `ClusterPermutation`: 通用管理类。
    *   `TTest`: 实现 t-test 统计量计算 (indep, rel, 1samp)。
3.  **功能验证**:
    *   **1D Clustering**: 能够对时间序列数据进行聚类置换检验，结果与 MNE-Python/SciPy 对齐。
    *   **2D Clustering**: 能够对 Source Estimate 数据（结合空间邻接矩阵）进行聚类，结果合理。
4.  **性能**:
    *   置换过程必须并行化。

## 2. 技术方案概要

### 目录结构
```
src/libraries/stats/
├── CMakeLists.txt
├── stats_global.h
├── clusterpermutation.h/cpp
└── statistics.h/cpp  (t-test, f-test helpers)
```

### 核心接口设计
```cpp
class STATSSHARED_EXPORT ClusterPermutation {
public:
    struct ClusterResult {
        std::vector<int> clusterIndices; // Mask of significant points
        double clusterStatistic;
        double pValue;
    };

    // 1D / General interface using Eigen Matrix
    // X: n_samples x n_features (e.g., subjects x timepoints)
    static QList<ClusterResult> permutationClusterOneSampleTest(
        const Eigen::MatrixXd& X, 
        double threshold, 
        int n_permutations = 1024,
        double tail = 0.05,
        const Eigen::SparseMatrix<double>& adjacency = Eigen::SparseMatrix<double>());
};
```

### 算法流程
1.  **Initial Stat**: 计算原始数据的 T-values。
2.  **Thresholding**: 将 T-values > threshold 的点标记为 candidate。
3.  **Clustering**:
    *   基于 `adjacency` 矩阵，使用 BFS/DFS 或 Union-Find 算法将 candidate 点聚类。
    *   计算每个 cluster 的 cluster-stat (e.g., sum(t)).
4.  **Permutation Loop** (Parallel):
    *   随机翻转部分样本的符号 (1-sample) 或 Shuffle 标签。
    *   重做 step 1-3。
    *   记录该次置换中最大的 cluster-stat (H0 distribution)。
5.  **P-value**:
    *   比较原始 cluster-stat 与 H0 分布。

## 3. 风险与约束
*   **Adjacency Matrix**: 对于 Source Space，邻接矩阵可能很大 (20k x 20k)，需确保使用稀疏矩阵。
*   **内存**: 并行时每个线程可能需要一份数据副本，需注意内存占用。对于大 Source Estimate，建议共享只读数据。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
