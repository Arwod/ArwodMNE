# Phase 14: Connectivity (Spectral Connectivity) - S2 Design

## 1. 模块架构

### 新增类 `AbstractSpectralMetric` (`metrics/abstractspectralmetric.h`)
继承自 `AbstractMetric`。

```cpp
class CONNECTIVITYSHARED_EXPORT AbstractSpectralMetric : public AbstractMetric
{
public:
    // Common spectral computation helpers
    
    // Compute Tapered Spectra for a single trial
    // Returns vector of matrices (n_tapers x n_freqs) for each channel? 
    // Or (n_channels x n_freqs) if averaged? No, we need tapers for CSD.
    // Return: vector of (n_channels x n_freqs) ? No.
    // Return: MatrixXcd (n_tapers * n_channels, n_freqs)? 
    // Current implementation: vecTapSpectra is a list of matrices (one per channel), each (n_tapers x n_freqs).
    static QList<Eigen::MatrixXcd> computeTaperedSpectra(
        const Eigen::MatrixXd& trialData, 
        const Eigen::MatrixXd& tapers, // (n_tapers x n_times)
        int n_fft);

    // Compute CSD from Tapered Spectra
    // Returns CSD matrix (n_channels x n_channels) for specific bins? 
    // Or full (n_channels x n_channels x n_freqs)?
    // Current implementation computes CSD row-wise to save memory?
    // It returns QVector<QPair<int, MatrixXcd>> vecPairCsd (sparse/compressed).
    
protected:
    // Helper to generate tapers using mne_tfr
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> generateTapers(
        int n_times, const QString& windowType, int n_tapers = -1, double nw = 4.0);
};
```

### Refactoring `Coherence` and `PLI`
*   They will use `AbstractSpectralMetric::generateTapers` and `computeTaperedSpectra`.
*   They will keep their specific accumulation loops (CSD vs ImagSign).

## 2. 核心算法细节

### Taper Generation
使用 `TFRLIB::TFRUtils::dpss_windows`。
*   如果 `windowType` 是 "hanning", "hamming"，则生成单 taper。
*   如果 "multitaper"，使用 DPSS。

### FFT
使用 `Eigen::FFT` (复用现有逻辑，或者封装)。

## 3. 兼容性
*   保持 `Network calculate(ConnectivitySettings&)` 接口不变。

