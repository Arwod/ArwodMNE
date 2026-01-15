# Phase 10: Decoding Foundation (CSP) - S2 Design

## 1. 模块架构
新建 `decoding` 库，作为 MNE-CPP 的核心库之一。

### 目录结构
```
src/libraries/decoding/
├── CMakeLists.txt          # 构建配置
├── decoding_global.h       # 库导出宏定义
├── csp.h                   # CSP 类声明
└── csp.cpp                 # CSP 类实现
```

### 依赖关系
*   **MNE Libraries**: `utils` (矩阵操作辅助), `fiff` (可能用到 Fiff 常量，暂时尽量解耦)。
*   **External**: `Eigen` (核心计算)。

## 2. 接口设计 (C++)

### CSP 类 (`csp.h`)

```cpp
#ifndef CSP_H
#define CSP_H

#include "decoding_global.h"
#include <Eigen/Core>
#include <vector>

namespace DECODINGLIB
{

class DECODINGSHARED_EXPORT CSP
{
public:
    explicit CSP(int n_components = 4, bool norm_trace = false, bool log = true, bool cov_est = false);
    ~CSP() = default;

    /**
     * @brief Fit CSP filters from epochs.
     * 
     * @param epochs Vector of epochs data (Channels x Time).
     * @param labels Vector of labels corresponding to each epoch (0 or 1).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels);

    /**
     * @brief Apply CSP filters to transform data.
     * 
     * @param epochs Vector of epochs data.
     * @return Feature matrix (n_epochs x n_components).
     */
    Eigen::MatrixXd transform(const std::vector<Eigen::MatrixXd>& epochs) const;

    // Getters
    Eigen::MatrixXd getFilters() const { return m_matFilters; }
    Eigen::MatrixXd getPatterns() const { return m_matPatterns; }
    Eigen::VectorXd getEigenValues() const { return m_vecEigenValues; }

private:
    // Parameters
    int m_iNComponents;
    bool m_bNormTrace;
    bool m_bLog;
    bool m_bCovEst; // Use covariance estimator (e.g., Ledoit-Wolf) - Future Work

    // Model attributes
    Eigen::MatrixXd m_matFilters;  // Spatial filters (W)
    Eigen::MatrixXd m_matPatterns; // Spatial patterns (A = W^-1)
    Eigen::VectorXd m_vecEigenValues;
    std::vector<std::string> m_vecClassNames;
};

} // namespace DECODINGLIB

#endif // CSP_H
```

## 3. 核心算法逻辑

### Fit 流程
1.  **输入检查**: 检查 `epochs` 和 `labels` 维度匹配，检查是否为二分类。
2.  **计算协方差**:
    *   遍历所有 Epochs，按 Label 分组。
    *   对每个 Epoch 计算协方差矩阵 $C = \frac{1}{N-1} X X^T$ (假设已去均值)。
    *   计算每类的平均协方差矩阵 $\bar{C}_0, \bar{C}_1$。
3.  **特征值分解**:
    *   解决广义特征值问题: $\bar{C}_0 w = \lambda \bar{C}_1 w$。
    *   或者使用联合对角化 (Simultaneous Diagonalization):
        *   $C_{sum} = \bar{C}_0 + \bar{C}_1$
        *   分解 $C_{sum} = U \Lambda U^T$
        *   白化变换 $P = \Lambda^{-1/2} U^T$
        *   变换后协方差 $S_0 = P \bar{C}_0 P^T$
        *   分解 $S_0 = B \lambda B^T$
        *   总滤波器 $W = P^T B$
4.  **排序与截断**:
    *   按特征值降序排序。
    *   选择前 `n_components/2` 和后 `n_components/2` 个滤波器。

### Transform 流程
1.  对每个 Epoch $X$:
    *   投影: $Z = W^T X$ (结果维度: Components x Time)
    *   计算方差: $v_i = \text{var}(Z_i)$
    *   取对数 (可选): $f_i = \log(v_i)$
2.  组合成特征矩阵。

## 4. 异常处理
*   如果数据维度不一致，抛出异常或返回 false。
*   如果协方差矩阵奇异，可能需要处理（如加正则化项）。

## 5. 数据流向
`Epochs Data` -> `CSP::fit` -> `Covariance Matrices` -> `Eigen Decomposition` -> `Spatial Filters` -> `CSP::transform` -> `Features`

