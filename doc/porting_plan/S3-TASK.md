# MNE Porting Project - Detailed Task Plan

This document outlines the detailed plan to port functionality from `mne-python` to `ArwodMNE`.

## Phase 1: Foundation & Verification

### Task 1.1: Verification Framework
*   **Goal**: Ensure C++ implementation matches Python numerically.
*   **Input**: Python scripts to generate synthetic data and expected outputs (intermediate results).
*   **Output**: C++ unit tests (using QTest or similar) that read inputs and assert outputs.
*   **Sub-tasks**:
    *   Create `tests/verification_data/` directory.
    *   Write `gen_test_data.py`: Generates Raw, Epochs, and simple filtered data.
    *   Create C++ test harness to read `.npy` or `.fif` debug data.

### Task 1.2: Signal Processing Primitives (`libraries/utils`)
*   **Goal**: Implement missing signal processing basics required for TFR and ICA.
*   **Sub-tasks**:
    *   **Hilbert Transform**: Implement `hilbert()` using FFTW/Eigen.
    *   **Convolution**: Efficient 1D convolution (FFT-based).
    *   **Window Functions**: Hamming, Hanning, Blackman (if missing).
    *   **Padding**: Reflection/Mirror padding for filtering.

## Phase 2: Time-Frequency (TFR)
*   **Target**: `libraries/tfr` (New Library)
*   **Dependencies**: `libraries/utils` (FFT, Convolution).
*   **Sub-tasks**:
    *   **Task 2.1: Morlet Wavelet**: Implement `mne.time_frequency.tfr.morlet`.
    *   **Task 2.2: TFR Class**: Create `TimeFrequency` data container (similar to `AverageTFR`).
    *   **Task 2.3: TFR Computation**: Implement `tfr_morlet` function (convolution of signal with wavelets).
    *   **Task 2.4: PSD**: Implement `psd_welch` (Welch's method).

## Phase 3: Preprocessing (ICA)
*   **Target**: `libraries/preprocessing` (New Library or expand `mne`)
*   **Dependencies**: `libraries/utils` (Whitening, PCA).
*   **Sub-tasks**:
    *   **Task 3.1: PCA**: Ensure robust PCA implementation (Eigen SVD).
    *   **Task 3.2: FastICA**: Implement FastICA algorithm (port from `sklearn` or `mne-python`'s internal version).
    *   **Task 3.3: ICA Class**: Create `ICA` class (fit, transform, apply).
    *   **Task 3.4: Artifact Detection**: Implement correlation-based EOG/ECG detection.

## Phase 4: Statistics (Basic)
*   **Target**: `libraries/stats` (New Library)
*   **Sub-tasks**:
    *   **Task 4.1**: T-test (1-sample, 2-sample).
    *   **Task 4.2**: Bonferroni/FDR correction.

## Phase 5: Beamformers & Advanced Inverse
*   **Target**: `libraries/inverse`
*   **Sub-tasks**:
    *   **Task 5.1**: Review existing LCMV/DICS in `mne-python`.
    *   **Task 5.2**: Port missing beamformer types.

## Execution Order
1.  **Task 1.1** (Verification) is prerequisite for everything.
2.  **Phase 2** (TFR) is self-contained and a good starting point for "new library" creation.
3.  **Phase 3** (ICA) is complex mathematically but critical for cleaning data.

## Technical Constraints
*   **Language**: C++17 (or match existing standard).
*   **Math**: Use `Eigen` for matrix ops.
*   **GUI**: Keep logic separate from GUI (`libraries` vs `applications`).
