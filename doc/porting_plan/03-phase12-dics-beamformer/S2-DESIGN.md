# Phase 12: Frequency-Domain Beamformer (DICS) - S2 Design

## 1. 模块架构
在 `src/libraries/inverse/beamformer` 中添加 `dics.h` 和 `dics.cpp`。

### 依赖关系
*   `TFRLIB`: 引用 `csd.h`。
*   `Eigen`: 矩阵运算。

## 2. 接口设计 (C++)

```cpp
#ifndef DICS_H
#define DICS_H

#include "../inverse_global.h"
#include <tfr/csd.h>
#include <Eigen/Core>
#include <vector>

namespace INVERSELIB {

class INVERSESHARED_EXPORT DICS
{
public:
    /**
     * @brief Compute Source Power using DICS beamformer.
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     *                  Assuming sources are grouped (e.g. 3 cols per source if n_ori=3).
     *                  If n_ori is 1 (fixed or normal), step is 1.
     * @param csd       Cross-Spectral Density object.
     * @param reg       Regularization parameter (e.g. 0.05).
     * @param n_ori     Number of orientations per source (default 1).
     * @param real_filter Whether to use real part of CSD to compute filters (default true).
     * @return Source Power Matrix (n_sources x n_freqs).
     *         n_sources = leadfield.cols() / n_ori.
     */
    static Eigen::MatrixXd compute_source_power(const Eigen::MatrixXd& leadfield, 
                                                const TFRLIB::CSD& csd, 
                                                double reg = 0.05,
                                                int n_ori = 1,
                                                bool real_filter = true);
                                                
    // Potential future method: compute_coherence
};

} // NAMESPACE

#endif // DICS_H
```

## 3. 核心算法逻辑

### compute_source_power 流程
1.  **维度检查**: 检查 `leadfield.rows()` 是否等于 `csd.n_channels`。
2.  **准备输出**: `MatrixXd power(n_sources, csd.freqs.size())`.
3.  **频率循环**:
    *   For each freq index `i`:
        *   Get $C = csd.data[i]$.
        *   If `real_filter`: $C_{filt} = C.real()$. Else $C_{filt} = C$.
        *   Regularize: $C_{inv} = (C_{filt} + \lambda I)^{-1}$.
        *   **源循环**:
            *   Extract $L_{src}$ (n_channels x n_ori).
            *   Compute Filter Denominator: $D = (L_{src}^T C_{inv} L_{src})^{-1}$.
            *   Compute Filter: $W = D L_{src}^T C_{inv}$.
            *   Compute Power: $P = W C W^H$.
                *   Note: $C$ here should be the original CSD (complex), even if filter was computed using real part.
                *   Power is Trace(P).real().
            *   Store in output.

## 4. 优化
*   对于大量源点，循环内的矩阵乘法 ($L^T C_{inv}$) 是瓶颈。
*   $C_{inv}$ 对所有源点是共用的。
*   $L_{src}^T C_{inv} L_{src}$ 计算量小 (3xCh * ChxCh * Chx3 -> 3xCh * Chx3 -> 3x3)。
*   OpenMP 可以并行化源点循环。

