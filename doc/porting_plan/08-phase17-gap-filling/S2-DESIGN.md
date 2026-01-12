# Phase 17: Gap Filling - System Design

## 1. Module Design

### 1.1. Math Utilities (`mnemath`)
*   **Location**: `src/libraries/utils/mnemath.h/cpp`
*   **Responsibilities**: Provide statistical and linear algebra utilities.
*   **New Components**:
    *   `ledoit_wolf_shrinkage(data)`: Computes optimal shrinkage.
    *   `ledoit_wolf(data)`: Computes regularized covariance.
*   **Dependencies**: `Eigen`, `FiffCov` (indirectly used by).

### 1.2. FiffCov I/O
*   **Location**: `src/libraries/fiff/fiff_cov.cpp`
*   **Responsibilities**: Serialize `FiffCov` object to FIFF file.
*   **Key Logic**:
    *   Open `FiffStream`.
    *   Start Block `FIFFB_MNE_COV`.
    *   Write Metadata: `Kind`, `Dim`, `Nfree`.
    *   Write Channels: `RowNames`, `BadChannels`.
    *   Write Data:
        *   If `diag`: Write `FIFF_MNE_COV_DIAG`.
        *   If `!diag`: Write `FIFF_MNE_COV` (Packed lower triangle). *Note: FiffCov stores full matrix, need to pack it.*
    *   Write Eig: `Eigenvalues`, `Eigenvectors`.
    *   Write Projections: `Projs`.
    *   End Block.

### 1.3. CSD Optimization
*   **Location**: `src/libraries/tfr/csd.cpp`
*   **Responsibilities**: Compute Cross-Spectral Density efficiently.
*   **Strategy**:
    *   Input: `vector<MatrixXd>` (epochs).
    *   Map Step: For each epoch, compute FFT and cross-spectral products (accumulate to a thread-local sum or return partial CSD).
    *   Reduce Step: Sum all partial CSDs and divide by N.
    *   Implementation: `QtConcurrent::mappedReduced`.

## 2. Interface Definition

### 2.1. Math
```cpp
namespace MNEMATHLIB {
    // data: (n_samples, n_features) - Note: standard is usually (n_features, n_samples) in MNE-CPP?
    // MNE-Python ledoit_wolf takes (n_samples, n_features).
    // Eigen default is col-major. We need to be careful about dimensions.
    // Let's assume input is (n_channels, n_times) like everywhere else in MNE-CPP.
    // So we transpose inside if needed.
    double ledoit_wolf_shrinkage(const Eigen::MatrixXd& data); 
}
```

### 2.2. FiffCov
```cpp
void FiffCov::write(QIODevice &p_IODevice) {
    // implementation
}
```

## 3. Data Flow
1.  **Covariance Write**: `FiffCov Object` -> `FiffStream` -> `QIODevice` -> `File`.
2.  **CSD Compute**: `Epochs` -> `Parallel Workers (FFT)` -> `Partial CSDs` -> `Main Thread (Sum)` -> `CSD Object`.

## 4. Verification Strategy
*   **Ledoit-Wolf**: Test with small random matrix, compare with Scikit-Learn result (hardcoded in test).
*   **Write**: Round-trip test (Write -> Read -> Compare).
*   **Optimization**: Check correctness vs sequential version.
