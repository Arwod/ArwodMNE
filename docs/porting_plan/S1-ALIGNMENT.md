# MNE Porting Project - Alignment Document

## 1. Project Overview

The goal of this project is to port functionality from the official Python-based MNE library (`mne-python`) to the C++ fork (`ArwodMNE`, originally `mne-cpp`). This is a large-scale engineering effort to bring the C++ version to feature parity with the Python version, enabling high-performance and real-time applications.

### Repositories

*   **Source (Reference)**: `mne-python` (Python)
    *   Status: Mature, feature-rich, "Gold Standard".
    *   Key Modules: `io`, `preprocessing`, `time_frequency`, `stats`, `decoding`, `viz`, `inverse`, `forward`.
*   **Target**: `ArwodMNE` (C++)
    *   Status: Strong in I/O (FIFF), Real-Time (RT Server), and Core Inverse/Forward. Missing advanced analysis features.
    *   Key Libraries: `fiff` (I/O), `mne` (Core), `disp/disp3D` (Viz), `connectivity`, `communication` (Real-time).

## 2. Current Status & Gap Analysis

| Feature Area | MNE-Python Status | ArwodMNE (C++) Status | Gap / Action |
| :--- | :--- | :--- | :--- |
| **File I/O** | Comprehensive (`.fif`, `.edf`, `.bdf`, etc.) | Strong (`fiff` library). Supports FIF, EDF (via app). | Maintain. Check for newer FIF tags/formats. |
| **Preprocessing** | ICA, SSP, Maxwell Filter, Artifact Repair | Basic SSP (`mne` lib). **No ICA found.** | **High Priority**: Implement ICA and advanced filtering. |
| **Forward/Inverse** | BEM, Source Spaces, MNE, dSPM, sLORETA, Beamformers | Supported (`mne` lib: `MNEForwardSolution`, `MNEInverseOperator`). | Verify parity of Beamformers (LCMV/DICS). |
| **Time-Frequency** | Morlet, Multitaper, TFR, PSD | **Missing** (No Morlet/TFR found). | **High Priority**: Port `time_frequency` module. |
| **Statistics** | Permutation tests, Cluster-level stats | **Missing**. | **Medium Priority**: Port basic stats. |
| **Connectivity** | Spectral, Phase-based, Envelope | Supported (`connectivity` lib). | Review for completeness. |
| **Visualization** | 2D Topomaps, 3D Brain, Interactive | Supported (`disp`, `disp3D` using Qt3D). | Good foundation. Expand as needed for new features. |
| **Real-Time** | `mne-realtime` (separate) | **Strong Core** (`mne_rt_server`, `mne_scan`). | C++ is leading here. |
| **Decoding** | CSP, classifiers (sklearn) | **Missing** (No `decoding` lib). | **Medium Priority**: Basic decoding (CSP). |

## 3. Goals and Boundaries

*   **Primary Goal**: Establish a roadmap to port missing critical modules: Preprocessing (ICA), Time-Frequency, and Statistics.
*   **Constraint**: Must strictly follow existing C++ patterns (Qt, Eigen).
*   **Approach**: Atomic tasks. Start with foundational math/signal processing before building complex flows.

## 4. Key Questions & Risks

*   **Dependencies**: MNE-Python uses `scipy` and `sklearn` heavily. We need to find C++ equivalents or implement algorithms from scratch (using Eigen).
*   **Testing**: How to verify parity? We should generate test data in Python and verify C++ output matches.

## 5. Next Steps

1.  Create a detailed task list (S3-TASK).
2.  Prioritize **Time-Frequency** or **ICA** as the first major porting target.
3.  Set up a verification framework (Python generates data -> C++ processes -> Verify).
