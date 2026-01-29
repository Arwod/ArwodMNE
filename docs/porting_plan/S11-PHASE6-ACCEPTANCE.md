# Phase 6: Minimum Norm Estimates Verification Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 6: Minimum Norm Estimates (Verification).

## Overview
Verify the existing `MinimumNorm` implementation in `libraries/inverse/minimumNorm` against `mne-python`. This module implements MNE, dSPM, and sLORETA inverse solutions.

## Tasks

### Task 6.1: Data Generation (Python)
*   **Goal**: Generate ground truth data using `mne-python`.
*   **Target**: `tests/verification_data/gen_test_data.py`
*   **Details**:
    *   Create a simulated Evoked dataset (using `mne.simulation`).
    *   Compute Forward solution (or use sample data if simulation is too complex to save as full FIF).
        *   Actually, generating a full Forward solution and saving as FIF might be heavy.
        *   Alternative: Use `mne-python` sample data (truncated) if available, or construct minimal objects.
        *   Let's try to generate a minimal Forward/Inverse setup.
        *   We need to save:
            1.  `evoked.fif`: The evoked data.
            2.  `inv.fif`: The inverse operator.
            3.  `stc_mne.txt` / `stc_dspm.txt`: The expected source time courses (for verification).
    *   The `gen_test_data.py` should use `mne.minimum_norm.make_inverse_operator` and `apply_inverse`.

### Task 6.2: C++ Verification Test
*   **Goal**: Run C++ implementation and compare with Python output.
*   **Target**: `src/testframes/test_verification/test_verification.cpp`
*   **Details**:
    *   Load `evoked.fif` using `FIFFLIB::FiffEvoked`.
    *   Load `inv.fif` using `MNELIB::MNEInverseOperator`.
    *   Instantiate `INVERSELIB::MinimumNorm`.
    *   Run `calculateInverse` for "MNE" and "dSPM".
    *   Compare resulting `MNESourceEstimate` data with `stc_*.txt`.

## Definition of Done
*   [ ] `gen_test_data.py` updated to produce FIF files and STC ground truth.
*   [ ] `test_verification` updated to load FIF files and verify Minimum Norm results.
*   [ ] MNE and dSPM results match within tolerance (e.g., corr > 0.99 or rel_error < 1e-2).
