# Phase 11: Advanced Time-Frequency (Multitaper & CSD) - S2 Design

## 1. 模块架构
在 `libraries/tfr` 中添加新文件：
*   `csd.h/cpp`: CSD 类及相关计算函数。

修改现有文件：
*   `tfr_utils.h/cpp`: 添加 `dpss_windows`。
*   `psd.h/cpp`: 添加 `psd_multitaper`。

### 依赖关系
*   `Eigen`: 矩阵运算、特征值分解、FFT (`unsupported/Eigen/FFT`)。
*   `mne_utils`: 数学辅助函数。

## 2. 接口设计 (C++)

### TFRUtils (`tfr_utils.h`)

```cpp
    /**
     * @brief Compute DPSS (Slepian) windows.
     * 
     * @param N Sequence length.
     * @param nw Time-bandwidth product.
     * @param k_max Number of windows to return.
     * @return Pair of (Windows [N x k_max], Eigenvalues [k_max]).
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> dpss_windows(int N, double nw, int k_max);
```

### PSD (`psd.h`)

```cpp
    /**
     * @brief Compute PSD using multitaper method.
     * 
     * @param data Input data (n_channels x n_times).
     * @param sfreq Sampling frequency.
     * @param bandwidth Frequency bandwidth (default: 4.0 / n_times * sfreq). 
     *                  Determines NW = bandwidth * n_times / 2.
     * @param adaptive Use adaptive weighting (default: false for now, implement basic mean first).
     * @return Pair of (PSDs [n_channels x n_freqs], Freqs).
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> psd_multitaper(
        const Eigen::MatrixXd& data, double sfreq, 
        double bandwidth = 0.0, bool adaptive = false, 
        bool low_bias = true, int n_jobs = 1);
```

### CSD (`csd.h`)

```cpp
#ifndef CSD_H
#define CSD_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <complex>

namespace TFRLIB {

class TFRSHARED_EXPORT CSD {
public:
    CSD();
    ~CSD() = default;

    // Data Holders
    std::vector<double> freqs; // Frequencies
    std::vector<Eigen::MatrixXcd> data; // Vector of CSD matrices (n_chan x n_chan) per freq
    std::vector<std::string> ch_names;
    int n_fft;

    /**
     * @brief Compute CSD using multitaper.
     * 
     * @param epochs Input epochs (std::vector<MatrixXd>).
     * @param sfreq Sampling frequency.
     * @param tmin Time start (relative to epoch).
     * @param tmax Time end.
     * @param fmin Min frequency.
     * @param fmax Max frequency.
     * @param bandwidth Bandwidth for multitaper.
     * @return CSD object.
     */
    static CSD compute_multitaper(
        const std::vector<Eigen::MatrixXd>& epochs, 
        double sfreq, 
        double tmin = 0.0, double tmax = 0.0,
        double fmin = 0.0, double fmax = 100.0,
        double bandwidth = 0.0);
        
    // Helper to get matrix at specific freq (nearest)
    Eigen::MatrixXcd get_data(double freq) const;
};

} // NAMESPACE

#endif // CSD_H
```

## 3. 核心算法逻辑

### DPSS 生成
1.  构造三对角矩阵 $T$ (N x N)。
    *   $T_{k,k} = ((\frac{N-1}{2})^2 - (k - \frac{N-1}{2})^2) \cos(2\pi W)$，其中 $W = NW/N$。
    *   $T_{k, k+1} = T_{k+1, k} = \frac{1}{2} (k+1) (N-1-k)$。
2.  求解特征值问题 $T v = \lambda v$。
3.  特征值最大的对应的特征向量即为 DPSS 窗。
4.  注意：Eigen 排序通常是升序，需取最后 $K$ 个。

### Multitaper CSD
1.  确定 $N$ (时间点数) 和 $NW$，生成 $K$ 个 DPSS 窗。
2.  对每个 Epoch $X$ (Ch x Time):
    *   对每个 Taper $k$:
        *   加窗: $X_k = X \cdot w_k$ (逐行点乘)。
        *   FFT: $F_k = \text{FFT}(X_k)$ (Ch x Freqs)。
    *   计算 Cross-Spectral Matrix for Taper $k$: $C_k(f) = F_k(f) F_k^H(f)$ (Outer product of column vectors at freq $f$)。
    *   注：实际上不需要显式存储所有 $C_k$，可以累加。
    *   平均 Tapers: $C_{epoch}(f) = \frac{1}{K} \sum C_k(f)$。
3.  平均 Epochs: $C_{final}(f) = \frac{1}{N_{epochs}} \sum C_{epoch}(f)$。

## 4. 数据流向
`Epochs` -> `CSD::compute_multitaper` -> `DPSS` -> `FFT` -> `Accumulate` -> `CSD Object`.

