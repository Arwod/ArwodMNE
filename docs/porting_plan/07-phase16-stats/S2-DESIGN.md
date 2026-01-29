# Phase 16: Non-Parametric Statistics - S2 Design

## 1. 模块架构

### 新增库 `stats`
位于 `src/libraries/stats`。

### 核心类设计

#### 1. `Statistics` (`statistics.h`)
提供基础统计计算的静态函数库。
```cpp
class STATSSHARED_EXPORT Statistics {
public:
    // 计算单样本 T 检验
    // Input: n_subjects x n_features
    // Output: 1 x n_features (t-values)
    static Eigen::RowVectorXd ttest1Samp(const Eigen::MatrixXd& data, double sigma = 0.0);
    
    // 计算独立样本 T 检验
    static Eigen::RowVectorXd ttestIndep(const Eigen::MatrixXd& group1, const Eigen::MatrixXd& group2);
};
```

#### 2. `ClusterPermutation` (`clusterpermutation.h`)
执行聚类置换检验的核心逻辑。
```cpp
class STATSSHARED_EXPORT ClusterPermutation {
public:
    struct Cluster {
        QList<int> indices;   // Global indices in the feature space
        double clusterStat;   // Sum of t-values
        double pValue;
    };

    // Main entry point for 1-sample test
    static QList<Cluster> permutationClusterOneSampleTest(
        const Eigen::MatrixXd& data, // n_subjects x n_features
        double threshold,            // T-value threshold
        int n_permutations = 1024,
        const Eigen::SparseMatrix<double>& adjacency = Eigen::SparseMatrix<double>(), // Optional spatial connectivity
        int n_jobs = 1 // Number of parallel jobs (or -1 for all)
    );

private:
    // Helper to find clusters given a binary mask and adjacency
    static QList<Cluster> findClusters(
        const Eigen::RowVectorXd& stats,
        const Eigen::RowVectorXi& mask,
        const Eigen::SparseMatrix<double>& adjacency
    );
    
    // Helper for one permutation iteration
    static double runPermutationStep(
        const Eigen::MatrixXd& data,
        double threshold,
        const Eigen::SparseMatrix<double>& adjacency,
        unsigned int seed
    );
};
```

## 2. 核心算法细节

### Clustering Algorithm (BFS)
当提供了邻接矩阵时，使用 BFS 寻找连通分量。
如果未提供邻接矩阵（且数据是 1D），假设相邻索引即为邻接（Temporal adjacency）。
*   **Adjacency Check**: `adj.coeff(i, j) != 0` implies connection.
*   **Optimization**: 使用 `visited` 数组避免重复访问。

### Permutation (1-Sample)
*   零假设：数据分布对称于 0。
*   置换策略：随机翻转每个 subject 数据的符号 (sign flipping)。
    *   $X_{perm} = X \cdot \text{diag}(s)$, where $s_i \in \{-1, 1\}$.

### Parallelization
*   使用 `QtConcurrent::mappedReduced`。
*   Map: 执行一次（或一批）置换，返回该次置换下的最大 Cluster Statistic (H0 sample)。
*   Reduce: 收集所有 H0 samples，形成 Null Distribution。

## 3. 依赖关系
*   `Eigen`: 矩阵运算。
*   `QtConcurrent`: 并行化。
*   `mne_utils`: 可能用到其中的随机数生成或基础数学。

## 4. 接口变更
无现有接口变更。这是新增功能。
