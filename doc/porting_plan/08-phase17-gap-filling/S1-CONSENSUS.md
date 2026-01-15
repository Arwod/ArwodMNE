# Phase 17: Gap Filling & Robustness (Consensus)

## 1. Goal
Address critical technical debts accumulated in previous phases to improve library stability, interoperability, and performance.

## 2. Scope

### 2.1. Mathematical Utilities
*   **Feature**: Ledoit-Wolf Shrinkage Regularization.
*   **Implementation**: Add `ledoit_wolf_shrinkage` function to `src/libraries/utils/mnemath.h`.
*   **Usage**: Used in `FiffCov::regularize` (to be updated) and future CSP improvements.

### 2.2. Data Persistence (I/O)
*   **Feature**: Write support for `FiffCov`.
*   **Implementation**: Add `FiffCov::write(QIODevice& device)` method.
*   **Format**: Standard FIFF format (Block: `FIFFB_MNE_COV`, Tags: `FIFF_MNE_COV_KIND`, `FIFF_MNE_COV_DIM`, `FIFF_MNE_COV`, etc.).
*   **Note**: CSD I/O is **postponed** due to lack of HDF5 support in the current build system.

### 2.3. Optimization
*   **Feature**: Parallel computation of CSD.
*   **Implementation**: Use `QtConcurrent::mapped` in `CSD::compute_multitaper` to parallelize over epochs or tapers.

## 3. Technical Design

### 3.1. Ledoit-Wolf
```cpp
// src/libraries/utils/mnemath.h
namespace MNEMATHLIB {
    // Computes the Ledoit-Wolf optimal shrinkage coefficient
    static double ledoit_wolf_shrinkage(const Eigen::MatrixXd& data);
    // Returns the regularized covariance matrix
    static Eigen::MatrixXd ledoit_wolf(const Eigen::MatrixXd& data);
}
```

### 3.2. FiffCov Write
```cpp
// src/libraries/fiff/fiff_cov.h
void write(QIODevice &p_IODevice);

// src/libraries/fiff/fiff_cov.cpp
void FiffCov::write(QIODevice &p_IODevice) {
    FiffStream::SPtr t_pStream(new FiffStream(&p_IODevice));
    t_pStream->start_block(FIFFB_MNE_COV);
    // ... write kind, dim, names, data ...
    t_pStream->end_block(FIFFB_MNE_COV);
}
```

## 4. Acceptance Criteria
1.  **Ledoit-Wolf**:
    *   Unit test compares results against a known hardcoded example (or Python output).
2.  **FiffCov Write**:
    *   Write a computed `FiffCov` to a file.
    *   Read it back using existing `FiffCov(QIODevice)` constructor.
    *   Verify data equality.
3.  **CSD Optimization**:
    *   Verify `CSD::compute_multitaper` produces same results.
    *   Benchmark execution time (should be faster on multi-core).

## 5. Next Steps
Proceed to **Phase 2: Architect** (Create S2-DESIGN.md).
