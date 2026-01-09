# MNE Porting Project - Gap Filling Roadmap (S15)

Based on a comprehensive comparison between `mne-python` and `ArwodMNE` (C++), this roadmap outlines the plan to bridge the feature gaps. The focus is on enabling advanced analysis capabilities (Decoding, Advanced Stats) and completing the frequency-domain analysis stack (Multitaper, CSD, DICS).

## Gap Analysis Summary

| Domain | Python Feature | C++ Status | Priority |
| :--- | :--- | :--- | :--- |
| **Decoding** | **CSP (Common Spatial Patterns)** | Missing | **High** (Crucial for BCI) |
| **TFR** | **Multitaper**, **CSD** | Missing | **High** (Prerequisite for DICS) |
| **Inverse** | **DICS** (Beamformer) | Missing | **Medium** |
| **Inverse** | sLORETA / eLORETA | Missing | Medium |
| **Stats** | **Cluster-level Permutation** | Missing | **High** (Standard for research) |
| **Preproc** | Maxwell Filtering (SSS) | Missing | Low (High complexity, MEG specific) |
| **I/O** | Direct EDF/BrainVision Read | Indirect (via App) | Medium |

## Phase 10: Decoding Foundation (CSP)
Establish the `decoding` library to support BCI and machine learning tasks.

*   **Goal**: Implement Common Spatial Patterns (CSP).
*   **Target**: `src/libraries/decoding` (New Library).
*   **Tasks**:
    *   Create `libraries/decoding` structure.
    *   Implement `CSP` class:
        *   `fit()`: Compute spatial filters from covariance matrices of two classes.
        *   `transform()`: Apply filters and compute log-variance features.
    *   **Verification**: Compare CSP patterns and features against `mne.decoding.CSP` using Motor Imagery data.

## Phase 11: Advanced Time-Frequency (Multitaper & CSD)
Enhance `tfr` library to support high-precision frequency analysis and connectivity prerequisites.

*   **Goal**: Implement Multitaper TFR/PSD and Cross-Spectral Density (CSD).
*   **Target**: `src/libraries/tfr`.
*   **Tasks**:
    *   Implement **DPSS (Slepian) Windows**: Generate sequences for multitaper.
    *   Implement `psd_multitaper`: PSD estimation using multitapers.
    *   Implement `compute_csd`: Compute Cross-Spectral Density matrices (sensor x sensor) across frequencies.
    *   **Verification**: Compare PSD and CSD matrices with `mne-python`.

## Phase 12: Frequency-Domain Beamformer (DICS)
Enable source localization of oscillatory sources.

*   **Goal**: Implement DICS (Dynamic Imaging of Coherent Sources).
*   **Target**: `src/libraries/inverse/beamformer`.
*   **Prerequisites**: Phase 11 (CSD).
*   **Tasks**:
    *   Implement `DICS` class in `beamformer`.
    *   Compute weights using CSD and Forward solution.
    *   **Verification**: Verify source localization of simulated oscillatory sources against `mne.beamformer.make_dics`.

## Phase 13: Non-Parametric Statistics (Cluster Permutation)
Implement the gold standard for neuroimaging statistics.

*   **Goal**: Implement Cluster-level Permutation Tests.
*   **Target**: `src/libraries/stats`.
*   **Tasks**:
    *   Implement **Clustering Algorithm**: Find connected components in 1D (time), 2D (time-freq), or 3D (source space) adjacency.
    *   Implement **Permutation Logic**: Randomly shuffle condition labels and re-compute statistics.
    *   **Verification**: Compare clusters and p-values with `mne.stats.permutation_cluster_test`.

## Phase 14: Direct I/O Expansion
Make the library more versatile by reading common EEG formats directly.

*   **Goal**: Native readers for EDF/BDF and BrainVision.
*   **Target**: `src/libraries/io` (New or extend `fiff`).
*   **Tasks**:
    *   Port `edf` reading logic from `mne_edf2fiff` into a reusable library class.
    *   Implement basic BrainVision (`.vhdr`, `.eeg`) reader.
    *   Ensure these populate `FiffRawData` structures directly.

## Next Step
Proceed with **Phase 10: Decoding Foundation (CSP)**.
