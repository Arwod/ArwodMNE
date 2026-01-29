# Phase 7: Connectivity Verification Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 7: Connectivity Verification.

## Overview
Verify the existing `Connectivity` implementation in `libraries/connectivity` against `mne-connectivity` (Python).

## Tasks

### Task 7.1: Data Generation (Python)
*   **Goal**: Generate multi-channel data with known connectivity.
*   **Target**: `tests/verification_data/gen_test_data.py`
*   **Details**:
    *   Simulate 2 signals with strong connectivity at 10 Hz (e.g., phase-locked sines).
    *   Generate ~10-20 epochs to ensure stable estimation.
    *   Compute Ground Truth using `mne_connectivity.spectral_connectivity_epochs`:
        *   Methods: `coh`, `plv`, `pli`.
        *   Freqs: 10 Hz (or broad band, checking peak).
    *   Save:
        *   `con_data_trials.txt`: Concatenated trials or separate files? 
            *   `ConnectivitySettings` accepts a list of matrices.
            *   We can save as `con_trial_0.txt`, `con_trial_1.txt`, ...
            *   Or one big matrix `(n_epochs * n_channels) x n_times` and reshape in C++.
            *   Simpler: Save one `con_data.txt` which is `n_channels x (n_times * n_epochs)` (concatenated in time) might be confusing.
            *   Let's save as `con_trial_X.txt` for 5 trials.
        *   `con_res_coh.txt`, `con_res_plv.txt`, `con_res_pli.txt`: The connectivity matrices (n_nodes x n_nodes x n_freqs).

### Task 7.2: C++ Verification Test
*   **Goal**: Run C++ `Connectivity::calculate` and compare.
*   **Target**: `src/testframes/test_verification/test_verification.cpp`
*   **Details**:
    *   Load trial data from text files.
    *   Configure `ConnectivitySettings`:
        *   Methods: "COH", "PLV", "PLI".
        *   SFreq, window type (Hanning).
    *   Run `Connectivity::calculate`.
    *   Extract results from `Network` objects.
    *   Compare with Python ground truth.

## Definition of Done
*   [x] `gen_test_data.py` generates connectivity data and ground truth.
*   [x] `test_verification` implements `verifyConnectivity`.
*   [x] COH, PLV, PLI results match `mne-connectivity` within tolerance (e.g., < 0.05 difference).
