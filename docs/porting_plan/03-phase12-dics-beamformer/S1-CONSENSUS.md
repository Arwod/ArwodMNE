# Phase 12: Frequency-Domain Beamformer (DICS) - S1 Consensus

## 1. 需求共识
确认在 `libraries/inverse` 中实现 DICS (Dynamic Imaging of Coherent Sources) 算法。

### 验收标准
1.  **代码实现**:
    *   `src/libraries/inverse/beamformer/dics.h/cpp` 存在。
    *   `DICS::compute_source_power` 能够接受 `CSD` 和 `Leadfield`，返回源功率。
2.  **功能特性**:
    *   支持 Tikhonov 正则化。
    *   支持实部滤波器 (`real_filter` 选项)。
    *   支持多频率计算（输出每个频率的功率）。
3.  **测试验证**:
    *   提供单元测试 `test_dics`。
    *   验证在模拟数据下（已知源位置和频率），DICS 能在正确位置重建功率峰值。

## 2. 技术方案概要
*   **输入**:
    *   `leadfield`: Eigen::MatrixXd (n_channels x n_sources * n_orient).
    *   `csd`: TFRLIB::CSD.
    *   `reg`: double (e.g. 0.05).
    *   `real_filter`: bool.
*   **算法流程**:
    1.  遍历 CSD 中的每个频率 $f$。
    2.  获取 $C(f)$。如果是 `real_filter`，取实部；否则取复数。
    3.  计算逆矩阵 $C_{inv} = (C(f) + \lambda I)^{-1}$。
    4.  对每个源点 $i$ (假设 Leadfield 已分块):
        *   $L_i$ 是该源点的 Leadfield 部分。
        *   $G = (L_i^T C_{inv} L_i)^{-1}$。
        *   $W_i = G L_i^T C_{inv}$。
        *   功率 $P_i(f) = \text{trace}(W_i C(f) W_i^{*T})$。
            *   注：如果是 "vector" beamformer，通常取 trace 或最大特征值作为该点的功率。我们先实现 trace (sum of power in 3 orientations)。
5.  **输出**:
    *   `Eigen::MatrixXd` (n_sources x n_freqs)。

## 3. 风险与约束
*   **计算量**: 对每个频率都要做矩阵求逆。虽然 CSD 维度 (Channels x Channels) 不大 (e.g. 64x64)，但源点数量可能很大 (e.g. 5000)。
*   **数值稳定性**: 求逆可能失败，必须加正则化。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
