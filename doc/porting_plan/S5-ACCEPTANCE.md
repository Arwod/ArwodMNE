# Phase 1: Foundation & Verification - Acceptance Log

## Branch Information
*   **Branch**: `feature/phase1-verification`
*   **Parent**: `dev`

## Tasks Status

### Task 1.1: Verification Framework
- [x] Create feature branch.
- [x] Create `tests/verification_data/` directory.
- [x] Implement `gen_test_data.py` (Python).
- [x] Implement C++ test runner (`src/testframes/test_verification`).

### Task 1.2: Signal Processing Primitives
- [x] Implement `hilbert()` in C++ (Migrated to `MNEMath` in `mnemath.h`).
- [x] Verify `hilbert()` against Python output.
- [x] Implement `convolve()` (FFT-based) in `MNEMath`.
- [x] Verify `convolve()` against Python output.
- [x] Implement Window functions (`hanning`, `hamming`, `blackman`) in `MNEMath`.
- [x] Verify Window functions against Python output.

## Verification Instructions
1.  **Generate Data**:
    ```bash
    cd ArwodMNE
    python3 tests/verification_data/gen_test_data.py
    ```
    *(Generates signal_raw, signal_hilbert_abs, signal_conv_ma50, window_* files)*

2.  **Build C++**:
    ```bash
    mkdir build && cd build
    cmake -S ../src -B . -DBUILD_TESTS=ON
    cmake --build . --target test_verification
    ```
3.  **Run Test**:
    ```bash
    # 如果在 ArwodMNE/build 目录下
    ../out/Release/tests/test_verification
    ```
