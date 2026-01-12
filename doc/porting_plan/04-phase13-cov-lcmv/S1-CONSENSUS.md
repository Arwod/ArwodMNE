# Phase 13: Covariance & LCMV Refactoring - S1 Consensus

## 1. 需求共识
确认重构 `Covariance` 和 `LCMV` 模块。

### 验收标准
1.  **Covariance 类**:
    *   能够存储协方差矩阵及元数据。
    *   `compute_covariance` 支持从 Epochs 计算。
    *   支持基本的正则化 (Diagonal loading)。
2.  **LCMV 重构**:
    *   `make_lcmv` 返回包含滤波器的对象。
    *   支持 `pick_ori="max-power"` (基于特征值分解选择主方向)。
    *   支持 `weight_norm="unit-noise-gain"` (UNG)。
    *   能够正确处理 `Covariance` 对象（包括 Whitening）。
3.  **测试**:
    *   `test_covariance`: 验证从模拟数据计算的协方差。
    *   `test_lcmv_refactored`: 验证 Max-Power 和 UNG 功能。

## 2. 技术方案概要

### Covariance
```cpp
class Covariance {
public:
    Eigen::MatrixXd data;
    std::vector<std::string> names;
    // ...
    static Covariance compute_empirical(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<std::string>& names);
};
```

### LCMV
```cpp
struct Beamformer {
    Eigen::MatrixXd weights;
    // ...
};

class LCMV {
public:
    static Beamformer make_lcmv(const Eigen::MatrixXd& leadfield, const Covariance& noise_cov, const Covariance& data_cov, ...);
};
```

## 3. 风险与约束
*   **兼容性**: 此次修改可能会破坏现有的 `LCMV::compute_weights` 接口（如果有其他模块调用）。
    *   *Check*: `mne_inverse` 库内部是否有调用？`inverse_global.cpp` 似乎没有。如果有外部调用，需注意。
    *   我们将保留旧接口标记为 `[[deprecated]]` 或直接重构（如果仅为内部测试使用）。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
