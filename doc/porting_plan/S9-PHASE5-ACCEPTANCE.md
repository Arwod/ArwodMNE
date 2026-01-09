# Phase 5: Beamformers & Advanced Inverse Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 5: Beamformers.

## Overview
Implement the LCMV (Linearly Constrained Minimum Variance) beamformer. This technique reconstructs source activity by constructing a spatial filter that minimizes variance while passing activity from a target location.

## Tasks

### Task 5.1: Covariance Matrix Computation
*   **Goal**: Compute data covariance and noise covariance matrices.
*   **Target**: `libraries/inverse/beamformer/covariance.h` (New)
*   **Details**:
    *   Compute empirical covariance: $C = \frac{1}{N-1} \sum (x - \bar{x})(x - \bar{x})^T$
    *   Regularization (shrinkage) support is desirable but optional for first pass.
- [x] Implement `compute_covariance`

### Task 5.2: LCMV Beamformer
*   **Goal**: Implement LCMV spatial filter.
*   **Target**: `libraries/inverse/beamformer/lcmv.h` (New)
*   **Details**:
    *   Formula: $W = \frac{C^{-1} L}{L^T C^{-1} L}$
    *   where $L$ is the leadfield for a specific source location.
    *   Usually done for many source points.
    *   Input: `ForwardSolution` (simulated leadfield), `Covariance` (Data/Noise).
    *   Output: `SourceEstimate` (or just source time courses).
- [x] Implement `make_lcmv` (computes filters)
- [x] Implement `apply_lcmv` (applies to data)

## Verification Instructions

1.  **Generate Data**: Update `tests/verification_data/gen_test_data.py`:
    *   Simulate a leadfield matrix (L) for a few dipoles.
    *   Simulate source activity (S).
    *   Generate data $X = L S + Noise$.
    *   Compute Covariance of X.
    *   Compute expected LCMV weights and source reconstruction using `mne.beamformer.make_lcmv` (if feasible) or just direct numpy calculation in the script.
2.  **Run Tests**:
    *   `test_verification` runs C++ LCMV.
    *   Compare reconstructed sources with ground truth.

## Definition of Done
*   [x] `libraries/inverse/beamformer` created.
*   [x] Covariance computation implemented.
*   [x] LCMV implemented.
*   [x] `test_verification` passes for LCMV scenarios.
