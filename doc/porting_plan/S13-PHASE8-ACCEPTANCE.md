# Phase 8: Dipole Fitting Verification Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 8: Dipole Fitting Verification.

## Overview
Verify the existing `DipoleFit` implementation in `libraries/inverse/dipoleFit` against simulated data.

## Tasks

### Task 8.1: Data Generation (Python)
*   **Goal**: Generate simulated Evoked data for a single dipole source.
*   **Target**: `tests/verification_data/gen_test_data.py`
*   **Details**:
    *   Simulate a dipole at `[0, 0.05, 0]` m with orientation `[1, 0, 0]`.
    *   Use Sphere Model (r=0.09m).
    *   Generate Forward Solution for 6-channel Magnetometer setup.
    *   Generate Evoked data (constant amplitude).
    *   Save:
        *   `df_evoked-ave.fif`: Measurement data (with `dev_head_t` identity).
        *   `df_noise-cov.fif`: Identity noise covariance.
        *   `df_true_pos.txt`, `df_true_ori.txt`, `df_true_amp.txt`: Ground truth.

### Task 8.2: C++ Verification Test
*   **Goal**: Run C++ `DipoleFit::calculateFit` and compare.
*   **Target**: `src/testframes/test_verification/test_verification.cpp`
*   **Details**:
    *   Implement `verifyDipoleFit`.
    *   Configure `DipoleFitSettings`:
        *   Sphere model (r0=[0,0,0]).
        *   Meas/Noise files.
        *   Guess grid settings (10mm).
    *   Run `calculateFit`.
    *   Verify:
        *   GOF > 0.90.
        *   Position Error < 1cm.
        *   Orientation Error < 0.1 (rad?).
        *   Amplitude Error < 1 nAm.

## Definition of Done
*   [x] `gen_test_data.py` generates dipole fit data.
*   [x] `test_verification` implements `verifyDipoleFit`.
*   [x] Dipole fitting results match ground truth within tolerance.
