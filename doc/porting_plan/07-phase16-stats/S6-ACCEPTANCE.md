# Phase 16: Non-Parametric Statistics - S6 Acceptance

## 验收结果

### 1. 功能完整性
*   [x] **Task 16.1 Skeleton**: 创建了 `stats` 库，配置了 CMake 和全局头文件。
*   [x] **Task 16.2 Basic Stats**: 实现了 `Statistics` 类，支持单样本和独立样本 T 检验。
*   [x] **Task 16.3 Clustering**: 实现了 `ClusterPermutation::findClusters`，支持 1D (Linear) 和 2D (Sparse Matrix) 聚类。
*   [x] **Task 16.4 Permutation**: 实现了 `permutationClusterOneSampleTest`，使用 `QtConcurrent` 进行并行置换检验，计算 Cluster-level P-values。
*   [x] **Task 16.5 Verification**: `test_stats` 测试通过，验证了 T 检验准确性和 1D/2D 聚类检测能力。

### 2. 代码质量
*   **并行化**: 利用 `QtConcurrent` 加速置换过程。
*   **泛型设计**: 接口使用 `Eigen::MatrixXd` 和 `Eigen::SparseMatrix`，适用于任意维度的特征数据。
*   **模块化**: 统计计算 (`Statistics`) 与置换逻辑 (`ClusterPermutation`) 分离。

## 结论
Phase 16 任务圆满完成。MNE-CPP 现已具备进行组水平统计推断的核心能力。
