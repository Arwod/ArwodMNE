# Phase 10: Decoding Foundation (CSP) - S1 Consensus

## 1. 需求共识
确认建立 `libraries/decoding` 模块，并实现基础的 CSP (Common Spatial Patterns) 算法。

### 验收标准
1.  **代码实现**:
    *   `src/libraries/decoding/csp.h` 和 `csp.cpp` 存在并编译通过。
    *   `CSP` 类包含 `fit` 和 `transform` 方法。
    *   支持二分类任务。
2.  **构建系统**:
    *   `CMakeLists.txt` 配置正确，能生成 `mne_decoding` 库。
3.  **测试验证**:
    *   提供单元测试（或验证测试），使用模拟数据验证 CSP 提取的特征。
    *   特征值分解结果正确（特征值和特征向量满足 CSP 约束：$\Sigma_1 w = \lambda \Sigma_2 w$ 或类似形式）。

## 2. 技术方案概要
*   **输入数据**: 使用 `std::vector<Eigen::MatrixXd>` 表示 Epochs 数据。
*   **核心依赖**: Eigen 3 库（用于 `SelfAdjointEigenSolver` 或 `GeneralizedSelfAdjointEigenSolver`）。
*   **算法流程**:
    1.  计算两类数据的平均协方差矩阵 $C_a, C_b$。
    2.  求解广义特征值问题 $C_a w = \lambda C_b w$ (或 $C_a w = \lambda (C_a + C_b) w$)。
    3.  排序特征向量，选取首尾各 $k$ 个作为空间滤波器。
    4.  特征提取：$f = \log(\text{var}(W^T X))$。

## 3. 风险与约束
*   **数据维度**: 假设所有 Epochs 具有相同的通道数和时间点数。
*   **数值稳定性**: 协方差矩阵求逆可能不稳定，需考虑是否加入正则化（本次暂不实现，但需注意潜在风险）。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
