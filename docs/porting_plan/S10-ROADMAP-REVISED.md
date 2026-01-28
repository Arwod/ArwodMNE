# MNE Porting Project - Revised Roadmap (Post-Phase 5)

## Status Update
As of Phase 5, we have successfully implemented and verified the following using a modern C++17/Eigen approach:
*   **Foundation**: Verification framework, signal processing primitives.
*   **TFR**: Morlet wavelets, PSD (Welch).
*   **Preprocessing**: PCA, FastICA.
*   **Stats**: T-Tests, Bonferroni/FDR correction.
*   **Inverse (Beamformer)**: LCMV, Covariance computation.

## Gap Analysis: Existing vs. New Code
The `ArwodMNE` repository contains a significant amount of legacy/ported code from MNE-C (in `src/libraries/mne`, `src/libraries/inverse`, `src/libraries/connectivity`). 
*   **Strengths**: Many complex algorithms (MNE, dSPM, Dipole Fit, Connectivity) are already implemented.
*   **Weaknesses**: 
    *   **Style**: Mixed C-style and C++98 style.
    *   **Verification**: Unclear if these modules strictly match `mne-python` numerically.
    *   **Integration**: New modules (like `mne_preprocessing`) use `Eigen::MatrixXd` directly, while legacy modules often use `FiffMatrix` or custom MNE structures.

**Strategy**: Instead of rewriting everything from scratch, we will **verify and adapt** the existing legacy modules. If a module is broken or too difficult to use, we will wrap or refactor it.

## Revised Roadmap

### Phase 6: Minimum Norm Estimates (Verification)
*   **Goal**: Verify and enable MNE, dSPM, and sLORETA using the existing `libraries/inverse/minimumNorm` module.
*   **Tasks**:
    *   **Data Generation**: Generate `InverseOperator` and `SourceEstimate` (MNE/dSPM) in Python.
    *   **Verification Test**: Create a test in `test_verification` that loads an InverseOperator (via `MNEInverseOperator` class) and applies it to Evoked data.
    *   **Refinement**: If the existing API is cumbersome, create a modern `Eigen`-based wrapper in a new header (e.g., `libraries/inverse/minimumnorm_modern.h`).

### Phase 7: Connectivity (Verification)
*   **Goal**: Verify functional connectivity metrics.
*   **Target**: `libraries/connectivity`.
*   **Tasks**:
    *   **Data Generation**: Generate multi-channel data with known coherence/PLV.
    *   **Verification Test**: Compare C++ `Coherence`, `PLV`, `PLI` results against `mne-connectivity` (Python).

### Phase 8: Dipole Fitting (Verification)
*   **Goal**: Verify dipole fitting algorithms.
*   **Target**: `libraries/inverse/dipoleFit`.
*   **Tasks**:
    *   **Data Generation**: Simulate dipole sources and forward fields.
    *   **Verification Test**: Run `DipoleFit::fit_dipoles` and check location/moment accuracy.

### Phase 9: Real-Time Pipeline Integration
*   **Goal**: Integrate the new/verified modules into a real-time processing pipeline.
*   **Target**: `libraries/rtprocessing` (or similar).
*   **Tasks**:
    *   Create a pipeline that streams data -> Filter -> ICA -> TFR/Inverse -> Display.

## Next Step
Proceed with **Phase 6: Minimum Norm Estimates**.
