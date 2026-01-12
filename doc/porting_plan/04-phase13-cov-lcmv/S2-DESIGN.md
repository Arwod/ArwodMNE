# Phase 13: Covariance & LCMV Refactoring - S2 Design

## 1. 模块架构

### Covariance (`covariance.h/cpp`)
将纯静态类改为实体类。

```cpp
class INVERSESHARED_EXPORT Covariance {
public:
    Covariance();
    Covariance(const Eigen::MatrixXd& data, const std::vector<std::string>& names);

    Eigen::MatrixXd data; // The covariance matrix
    std::vector<std::string> names; // Channel names
    int nfree; // Degrees of freedom
    bool is_diagonal; // Optimization flag

    // Static computation methods
    static Covariance compute_empirical(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<std::string>& names);
    
    // Regularization
    Covariance regularize(const Eigen::MatrixXd& info, double grad, double mag, double eeg, double proj, bool exclude_bads) const;
    // Simple regularization for now
    Covariance regularize(double lambda) const;
};
```

### LCMV (`lcmv.h/cpp`)

```cpp
struct INVERSESHARED_EXPORT BeamformerWeights {
    Eigen::MatrixXd weights; // (n_sources x n_channels)
    std::vector<std::string> ch_names;
    std::string pick_ori;
    bool weight_norm;
};

class INVERSESHARED_EXPORT LCMV {
public:
    /**
     * @brief Compute LCMV beamformer.
     * 
     * @param leadfield (n_channels x n_sources*n_ori)
     * @param info_cov Noise covariance (for whitening/inversion).
     * @param data_cov Data covariance (optional, for some variants, usually LCMV uses data_cov as the C matrix).
     *                 Standard LCMV uses Data Covariance as C.
     *                 Wait, standard LCMV minimizes variance of output subject to unit gain.
     *                 The C matrix in (L^T C^-1 L)^-1 L^T C^-1 is the Data Covariance.
     *                 Noise covariance is used for whitening or if we do 'Unit Noise Gain' normalization accurately.
     * @param reg Regularization factor.
     * @param pick_ori "vector", "max-power", "normal".
     * @param weight_norm "unit-noise-gain", "none".
     */
    static BeamformerWeights make_lcmv(
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov = Covariance(), // Optional, used for weight norm
        double reg = 0.05,
        const std::string& pick_ori = "vector",
        const std::string& weight_norm = "none",
        int n_ori = 3 // Orientations per source
    );

    static Eigen::MatrixXd apply(const BeamformerWeights& weights, const Eigen::MatrixXd& data);
};
```

## 2. 核心算法细节

### Max-Power Orientation
对于每个源位置 $i$：
1.  计算 $W_{vector} (3 \times Ch)$。
2.  计算源协方差 $P = W_{vector} C_{data} W_{vector}^T (3 \times 3)$。
3.  对 $P$ 进行奇异值分解 (SVD) 或特征值分解。
4.  最大特征值对应的特征向量即为 $v_{max}$。
5.  新权重 $w_{opt} = v_{max}^T W_{vector} (1 \times Ch)$。

### Unit-Noise-Gain (UNG)
1.  计算未归一化权重 $W$。
2.  如果提供 `noise_cov` ($C_{noise}$)，则 $N = C_{noise}$。否则假设 $N = I$。
3.  计算噪声功率 $P_{noise} = W N W^T$ (diagonal elements if vector, or scalar).
4.  归一化因子 $g = 1 / \sqrt{P_{noise}}$。
5.  $W_{ung} = g \cdot W$.

## 3. 兼容性处理
保留旧的 `compute_weights` 函数，但在内部调用新的逻辑，或者标记为 Deprecated。为了代码整洁，直接修改旧文件，因为这是内部库。

