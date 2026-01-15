# Phase 14: Connectivity (Spectral Connectivity) - S1 Consensus

## 1. 需求共识
确认重构 `connectivity` 库以消除冗余并利用 `mne_tfr`。

### 验收标准
1.  **架构重构**:
    *   引入 `AbstractSpectralMetric` 类。
    *   `Coherence` 和 `PhaseLagIndex` 及其变体继承自该类，不再包含重复的 FFT/CSD 循环代码。
2.  **功能一致性**:
    *   重构后的代码应产生与重构前相同（或更精确）的结果。
    *   必须通过 `test_connectivity` 验证。
3.  **依赖更新**:
    *   `connectivity` 库链接 `mne_tfr`。
    *   移除对 `utils/spectral.h` 的依赖（如果完全替代）。

## 2. 技术方案概要

### AbstractSpectralMetric
```cpp
class AbstractSpectralMetric : public AbstractMetric {
protected:
    // Template method pattern
    void computeSpectral(ConnectivitySettings& settings);
    
    // Hooks for subclasses
    virtual void accumulate(int trial, const std::vector<Eigen::MatrixXcd>& tapSpectra) = 0;
    virtual void finalize(Network& network) = 0;
    
    // Helper to compute Tapered Spectra for one trial
    std::vector<Eigen::MatrixXcd> computeTaperedSpectra(const Eigen::MatrixXd& trialData, ...);
};
```
*   Wait, the current implementation accumulates sums (`vecPairCsdSum`) across threads.
*   The `accumulate` hook needs to be thread-safe or return a partial result that is reduced later.
*   Current `Coherency` uses `QtConcurrent::map` with a lambda that writes to `vecPairCsdSum` protected by `QMutex`. This is inefficient (lock contention).
*   **Better approach**: `QtConcurrent::mappedReduced`. Map computes trial CSD/Stats. Reduce sums them up.
*   However, for this Phase, I will stick to the existing `mutex` approach to minimize risk, but move the logic to the base class. Or simpler: The base class provides the `computeTrialSpectra` function, and the subclasses use `QtConcurrent::map` with their specific lambda.

### Refined Plan
1.  Create `SpectralMetric` base class.
2.  Implement `computeTaperedSpectra` in base class (using `mne_tfr::TFRUtils::dpss_windows`).
3.  Subclasses (`Coherence`, `PLI`) implement `calculate` by calling `QtConcurrent::map` using `computeTaperedSpectra` and their specific accumulation logic.

## 3. 风险与约束
*   **GUI 耦合**: `ConnectivitySettings` 包含 GUI 相关的数据结构。需小心不要破坏。
*   **性能**: 使用 `mne_tfr` 的 DPSS 可能比旧的实现快或慢，需关注。

## 4. 确认状态
*   [x] 需求清晰
*   [x] 技术路径明确
*   [x] 边界已锁定

进入 **Phase 2: Architect**。
