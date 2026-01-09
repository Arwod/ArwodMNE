# Phase 2: Time-Frequency (TFR) - Acceptance Log

## Branch Information
*   **Branch**: `feature/phase2-tfr`
*   **Parent**: `dev`

## Tasks Status

### Task 2.1: Morlet Wavelet
*   **Goal**: Implement `morlet` wavelet generation compatible with MNE-Python.
*   **Target**: `libraries/tfr/tfr_utils.h` (New)
- [x] Implement `morlet(sfreq, freqs, n_cycles, sigma)`
- [x] Verify against `mne.time_frequency.tfr.morlet`

### Task 2.2: TFR Data Structure
*   **Goal**: Create a container for TFR results.
*   **Target**: `libraries/tfr/timefrequency.h` (New)
- [x] Define `TimeFrequency` class
- [x] Support data storage (epochs x channels x freqs x times)

### Task 2.3: TFR Computation (Morlet)
*   **Goal**: Implement `tfr_morlet` (convolution-based).
*   **Target**: `libraries/tfr/tfr_compute.h` (New)
- [x] Implement `tfr_morlet` using `MNEMath::convolve`
- [x] Support `use_fft=True` (default)
- [x] Verify against `mne.time_frequency.tfr_morlet`

### Task 2.4: PSD (Welch)
*   **Goal**: Implement Power Spectral Density using Welch's method.
*   **Target**: `libraries/tfr/psd.h` (New)
- [x] Implement `psd_welch`
- [x] Verify against `mne.time_frequency.psd_welch`

## Verification Instructions
1.  **Generate Phase 2 Data**:
    ```bash
    cd ArwodMNE
    # (Update gen_test_data.py first to include TFR cases)
    python3 tests/verification_data/gen_test_data.py
    ```

2.  **Build**:
    ```bash
    cd build
    cmake -S ../src -B . -DBUILD_TESTS=ON
    cmake --build . --target test_verification
    ```
