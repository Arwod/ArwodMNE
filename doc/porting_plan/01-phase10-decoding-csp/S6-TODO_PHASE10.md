# Phase 10: Decoding Foundation (CSP) - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **多分类支持**:
    *   目前 `fit` 函数中强制检查 `unique_labels.size() == 2`。
    *   **Action**: 实现 One-vs-Rest 策略，训练多个 CSP 模型。

2.  **正则化 (Regularization)**:
    *   目前假设协方差矩阵是可逆的。在小样本高维数据下可能不稳定。
    *   **Action**: 实现 Ledoit-Wolf 收缩或对角线加载（Diagonal Loading）。

3.  **Python 兼容性验证**:
    *   目前使用模拟数据验证。
    *   **Action**: 导出 MNE-Python 处理的真实数据（如 Motor Imagery），在 C++ 中加载并对比结果。

4.  **构建系统优化**:
    *   检查是否需要支持 `qmake` (.pro 文件)，目前仅添加了 `CMake` 支持。如果项目仍需维护 qmake 构建，则需要添加 `.pro` 文件。

5.  **安装配置**:
    *   确保 `make install` 能正确安装新库的头文件和二进制文件。
