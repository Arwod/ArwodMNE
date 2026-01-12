# MNE Porting Project - Revised Roadmap (Post-Phase 14)

## 1. Current Status Overview
We have successfully completed Phases 10 through 14, establishing core capabilities for Decoding, Time-Frequency Analysis, Beamforming, and Connectivity.

| Phase | Feature | Status | Key Achievements |
| :--- | :--- | :--- | :--- |
| **10** | **Decoding (CSP)** | ✅ Done | CSP implementation, sklearn-like API. |
| **11** | **TFR (Multitaper/CSD)** | ✅ Done | DPSS windows, Multitaper PSD, CSD computation. |
| **12** | **DICS Beamformer** | ✅ Done | Frequency-domain beamformer, source power mapping. |
| **13** | **Covariance & LCMV** | ✅ Done | Covariance estimation, LCMV beamformer refactoring. |
| **14** | **Connectivity** | ✅ Done | Spectral connectivity (Coh, PLI) refactoring using `mne_tfr`. |

## 2. Accumulated Technical Debt (TODOs from Phase 10-14)
The following tasks have been identified as necessary improvements but were out of scope for the initial phases.

### A. Mathematics & Algorithms
1.  **Regularization (Ledoit-Wolf)**:
    *   *Source*: Phase 10 (CSP), Phase 13 (Covariance).
    *   *Impact*: Critical for stability on high-dimensional/small-sample data.
2.  **Adaptive Weighting (Multitaper)**:
    *   *Source*: Phase 11.
    *   *Impact*: Reduces spectral leakage.
3.  **Advanced Beamforming**:
    *   *Source*: Phase 12 (DICS Orientation Optimization, Coherence Map), Phase 13 (Auto Rank).
    *   *Impact*: Improves localization accuracy.

### B. I/O & Interoperability
1.  **Data Persistence**:
    *   *Source*: Phase 11 (CSD .h5/.fif), Phase 13 (Covariance .fif).
    *   *Impact*: Allows saving intermediate results and validation against MNE-Python.

### C. Performance & Optimization
1.  **Parallelization**:
    *   *Source*: Phase 11 (CSD computation).
    *   *Impact*: Speed up connectivity analysis.
2.  **Memory Efficiency**:
    *   *Source*: Phase 14 (Streaming computation).
    *   *Impact*: Enable processing of long/high-density recordings.

## 3. Future Development Plan

### Phase 15: Source Estimate Morphing (Current)
To enable group-level analysis (which is the next logical step after individual source localization), we need to map source estimates from individual subject space to a common reference (e.g., fsaverage).
*   **Goal**: Implement morphing of SourceEstimates.
*   **Key Tasks**:
    *   Compute/Read Morph Maps (sparse matrices).
    *   Apply morphing to `MNEHemisphere` / `MNESourceEstimate`.
    *   Support surface-based morphing.

### Phase 16: Non-Parametric Statistics
With morphed data available, we can implement group statistics.
*   **Goal**: Cluster-level permutation tests.
*   **Key Tasks**:
    *   Clustering algorithms (temporal, spatial, spatio-temporal).
    *   Permutation testing logic.

### Phase 17: Robustness & Optimization (Gap Filling)
Address the accumulated technical debts to make the library production-ready.
*   **Goal**: Implement Regularization, I/O, and Optimizations.
*   **Key Tasks**:
    *   Implement Ledoit-Wolf shrinkage (shared utility).
    *   Implement CSD/Covariance I/O.
    *   Optimize CSD/Connectivity computation.

## 4. Execution Strategy for Phase 15
We will proceed with **Phase 15: Source Estimate Morphing**. This is a prerequisite for Phase 16 (Stats) and closes the loop on source analysis.

### Why Morphing now?
*   We have Source Estimates (LCMV, DICS, MNE).
*   We cannot do Group Analysis without Morphing.
*   It is a distinct, self-contained module (`mne_morph_map`).
