# Phase 17: Gap Filling & Robustness (Alignment)

## 1. Context Analysis
The project has successfully completed Phases 10 through 16, implementing core functionalities including Decoding (CSP), Time-Frequency Analysis (Multitaper, CSD), Beamforming (DICS, LCMV), Connectivity, and Statistics (Cluster Permutation).

However, during these phases, several "technical debts" and missing features were identified, primarily related to:
1.  **Numerical Stability**: Lack of regularization (e.g., Ledoit-Wolf) in Covariance and CSP estimation.
2.  **Interoperability**: Missing I/O capabilities for intermediate objects like `CSD` and `FiffCov` (writing).
3.  **Performance**: Opportunity for parallelization in CSD computation.

This phase aims to address these gaps to make the library production-ready and fully compatible with `mne-python` workflows for validation.

## 2. Requirement Understanding

### 2.1. Regularization (Ledoit-Wolf)
*   **Requirement**: Implement Ledoit-Wolf shrinkage for covariance estimation.
*   **Target**: A shared utility in `libraries/utils` or `libraries/fiff` (since `FiffCov` needs it).
*   **Reference**: `sklearn.covariance.ledoit_wolf`, `mne.cov.compute_covariance`.
*   **Usage**: Should be usable by `FiffCov::compute_cov` and `CSP`.

### 2.2. Data Persistence (I/O)
*   **Requirement A**: Implement writing capability for `FiffCov`.
    *   Existing: `FiffCov` has a read constructor but no `write` method.
    *   Target: `FiffCov::write(QIODevice& device)`.
*   **Requirement B**: Implement I/O for `CSD`.
    *   Existing: `CSD` class exists but has no I/O.
    *   Target: `CSD::write` and `CSD::read` (support .h5 or .fif). Since `mne-python` saves CSD as .h5, we should probably support that or define a custom .fif tag if appropriate. *Correction*: MNE-Python saves CSDs to HDF5 (`.h5`) usually. We need to check if we have HDF5 support. If not, maybe stick to `.fif` or simple binary for now, but `.h5` is preferred for CSD.
    *   *Constraint*: Check if HDF5 library is available in the project. If not, might need to rely on FIFF.

### 2.3. Optimization
*   **Requirement**: Parallelize CSD computation.
*   **Target**: `CSD::compute_multitaper`.
*   **Method**: Use `QtConcurrent` or `OpenMP` (existing project uses QtConcurrent in `stats`).

## 3. Decision Strategy (Smart Decisions)

### Q1: Where to implement Ledoit-Wolf?
*   **Decision**: `src/libraries/utils/mnemath.h` or similar. It's a general linear algebra operation.
*   *Verification*: Check where other math helpers are. `src/libraries/utils` seems appropriate.

### Q2: CSD I/O Format?
*   **Analysis**: MNE-Python uses `.h5` for CSD.
*   **Constraint**: Does `ArwodMNE` link against HDF5?
*   **Action**: Check `CMakeLists.txt` for HDF5 dependency. If missing, adding it might be too heavy.
*   **Alternative**: Save as `.fif` using a custom structure or `FiffGeneric` tags? Or just `.fif` since CSD is basically matrices + frequencies.
*   *Proposal*: If no HDF5, implement a `.fif` serialization for CSD (similar to how Covariance is stored).

### Q3: FiffCov Write Location?
*   **Decision**: Add `write` method to `FiffCov` class in `fiff_cov.cpp`.

## 4. Key Questions (Interruption Points)
1.  Do we have HDF5 support? (I will check CMakeLists.txt).
2.  Should Ledoit-Wolf be a standalone function or part of a class? (Standalone function in a namespace is better).

## 5. Next Steps
1.  Check HDF5 availability.
2.  Finalize Design.
3.  Create S1-CONSENSUS.md.
