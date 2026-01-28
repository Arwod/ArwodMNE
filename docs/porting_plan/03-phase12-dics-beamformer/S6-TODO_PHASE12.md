# Phase 12: Frequency-Domain Beamformer (DICS) - TODO List

## 待办事项 (Technical Debt & Future Work)

1.  **相干性计算 (Coherence Mapping)**:
    *   目前的 DICS 实现仅输出源功率 (`power_map`)。
    *   **Action**: 扩展 `DICS` 类，支持计算参考源与全脑的相干性 (Source-Source Coherence)。
        *   公式: $Coh(i, j) = |W_i C W_j^{*T}| / \sqrt{P_i P_j}$。

2.  **方向优化 (Orientation Optimization)**:
    *   目前支持 "vector" (保留所有方向) 或 "normal" (如果 Leadfield 是一维的)。
    *   **Action**: 实现 "max-power" 方向选择，即找到使输出功率最大的方向 $v_{max}$，这需要求解广义特征值问题。

3.  **MNEForwardSolution 集成**:
    *   目前的接口接受原始的 `Eigen::MatrixXd` Leadfield。
    *   **Action**: 提供接受 `MNEForwardSolution` 对象的重载函数，自动提取 Leadfield 和源空间信息。

4.  **GUI 集成**:
    *   在 `mne_analyze` 或 `mne_scan` 中添加 DICS 插件。
