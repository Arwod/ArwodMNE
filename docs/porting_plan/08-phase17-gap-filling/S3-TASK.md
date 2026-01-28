# Phase 17: Gap Filling - Tasks

## Task 1: Ledoit-Wolf Regularization
*   **Goal**: Implement Ledoit-Wolf shrinkage algorithm.
*   **Input**: `Eigen::MatrixXd` (data).
*   **Output**: `Eigen::MatrixXd` (covariance), `double` (shrinkage).
*   **File**: `src/libraries/utils/mnemath.h/cpp`.
*   **Verification**: Unit test in `src/testframes/test_utils`.

## Task 2: FiffCov Write Support
*   **Goal**: Enable saving `FiffCov` to disk.
*   **Input**: `FiffCov` object, `QIODevice`.
*   **Output**: Binary FIFF data in stream.
*   **File**: `src/libraries/fiff/fiff_cov.cpp`.
*   **Verification**: Write a covariance matrix and read it back.

## Task 3: CSD Parallelization
*   **Goal**: Speed up CSD computation.
*   **Input**: `vector<MatrixXd>` (epochs).
*   **Output**: `CSD` object (identical to sequential).
*   **File**: `src/libraries/tfr/csd.cpp`.
*   **Method**: Use `QtConcurrent` to parallelize loop over epochs.
*   **Verification**: Compare output with sequential version.

## Task 4: Final Acceptance
*   **Goal**: Run all verifications.
*   **File**: `doc/porting_plan/08-phase17-gap-filling/S6-ACCEPTANCE.md`.
