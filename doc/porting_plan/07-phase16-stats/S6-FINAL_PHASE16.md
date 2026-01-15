# Phase 16: Non-Parametric Statistics - Final Report

## 项目总结
本项目为 MNE-CPP 添加了 `stats` 库，实现了基于聚类的非参数置换检验（Cluster-based Permutation Tests）。这是脑科学数据分析中处理多重比较校正的标准方法。

## 交付物清单
1.  **源代码**:
    *   `src/libraries/stats/`: 新增统计库。
        *   `statistics.h/cpp`: T 检验实现。
        *   `clusterpermutation.h/cpp`: 聚类置换检验逻辑。
2.  **测试代码**:
    *   `src/testframes/test_stats`: 验证测试。
3.  **文档**:
    *   `doc/porting_plan/07-phase16-stats/*`.

## 技术亮点
*   **并行计算**: 利用 `QtConcurrent` 自动管理线程池，高效执行数千次置换。
*   **灵活的邻接处理**: 支持任意稀疏矩阵定义的空间邻接，也支持默认的时间邻接。
*   **标准兼容**: T 检验结果与 Scipy/MNE-Python 保持一致。

## 后续建议
*   **Phase 17**: 优化和鲁棒性。
*   **Spatio-Temporal Clustering**: 目前测试覆盖了 1D 和 2D 空间聚类。对于时空聚类（3D），只需构建对应的时空邻接矩阵即可直接复用 `ClusterPermutation`。建议添加辅助函数来生成这些矩阵。
*   **TFCE**: 未来可考虑实现 Threshold-Free Cluster Enhancement (TFCE)。
