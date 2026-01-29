# Phase 3: Preprocessing (ICA) Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 3: Independent Component Analysis (ICA).

## Overview
Port the ICA functionality (specifically FastICA) from `mne-python`/`sklearn` to C++. This involves Principal Component Analysis (PCA) for whitening and the FastICA algorithm for source separation.

## Tasks

### Task 3.1: PCA & Whitening
*   **Goal**: Implement PCA for data whitening, required before ICA.
*   **Target**: `libraries/preprocessing/pca.h` (New)
*   **Details**:
    *   Compute SVD of covariance matrix or data matrix.
    *   Whitening matrix computation.
    *   Inverse whitening.
- [x] Implement PCA class/functions
- [x] Verify whitening against `sklearn.decomposition.PCA`

### Task 3.2: FastICA Algorithm
*   **Goal**: Implement the FastICA algorithm (Parallel or Deflation).
*   **Target**: `libraries/preprocessing/fastica.h` (New)
*   **Details**:
    *   Implement `_sym_decorrelation` (Symmetric decorrelation).
    *   Implement `_ica_par` (Parallel FastICA loop).
    *   Support `logcosh` and `exp` negentropy functions (g and g').
- [x] Implement FastICA core algorithm
- [x] Verify mixing/unmixing matrices against `sklearn.decomposition.FastICA`

### Task 3.3: ICA Class
*   **Goal**: Create a high-level `ICA` class similar to `mne.preprocessing.ICA`.
*   **Target**: `libraries/preprocessing/ica.h` (New)
*   **Details**:
    *   `fit(Raw)`: Run PCA then FastICA.
    *   `apply(Raw)`: Remove selected components and reconstruct signal.
    *   `plot_sources`: (Optional/Later - maybe just export data).
- [x] Define `ICA` class
- [x] Implement `fit`
- [x] Implement `apply` (exclude components)

## Verification Instructions

1.  **Generate Data**: Update `tests/verification_data/gen_test_data.py` to generate:
    *   Mixed signals (sources * mixing matrix).
    *   Expected whitening matrix.
    *   Expected unmixing matrix (ICA result).
    *   Expected sources (reconstructed).
2.  **Run Tests**:
    *   `test_verification` should load mixed signals.
    *   Run C++ PCA/ICA.
    *   Compare unmixing matrix and sources with Python ground truth.
    *   Note: ICA sources can have sign flips and permutation differences. Verification logic must account for this (correlation matching).

## Definition of Done
*   [x] `libraries/preprocessing` created and compiled.
*   [x] PCA/Whitening implemented and verified.
*   [x] FastICA implemented and verified.
*   [x] `ICA::fit` and `ICA::apply` working.
*   [x] `test_verification` passes for ICA scenarios.
