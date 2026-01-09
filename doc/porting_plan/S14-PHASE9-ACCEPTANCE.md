# Phase 9: Real-Time Pipeline Integration Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 9: Real-Time Pipeline Integration.

## Overview
Verify the integration of real-time processing modules (`libraries/rtprocessing`) to form a processing pipeline. Specifically, focus on the online covariance estimation (`RtCov`) and online inverse operator computation (`RtInvOp`).

## Tasks

### Task 9.1: Fix Legacy Bugs
*   **Goal**: Fix crash in `MNEHemisphere::add_geometry_info` when handling discrete source spaces (empty triangle list).
*   **Target**: `src/libraries/mne/mne_hemisphere.cpp`
*   **Details**:
    *   Ensure `neighbor_tri` vector is resized to `np` (number of points) instead of `tris.rows()` (number of triangles), preventing out-of-bounds access for discrete source spaces.

### Task 9.2: Data Generation
*   **Goal**: Ensure necessary data for RT verification is available.
*   **Target**: `tests/verification_data/gen_test_data.py`
*   **Details**:
    *   Save the forward solution (`df_fwd.fif`) generated in Phase 8/6.
    *   Ensure it is compatible with MNE-CPP readers (discrete source space).

### Task 9.3: RT Pipeline Verification Test
*   **Goal**: Verify `RtCov` -> `RtInvOp` pipeline.
*   **Target**: `src/testframes/test_rt_pipeline_integration/test_rt_pipeline_integration.cpp`
*   **Details**:
    *   **Setup**: Load `FiffInfo` and `MNEForwardSolution`.
    *   **RtCov**: Feed simulated data buffers (random noise) to `RtCov`. Verify it computes a valid `FiffCov` after accumulating enough samples.
    *   **RtInvOp**: Pass the computed covariance to `RtInvOp`. Verify it asynchronously computes a valid `MNEInverseOperator`.
    *   **Integration**: Ensure the data flows from `RtCov` to `RtInvOp` and produces a result.

## Definition of Done
*   [x] `MNEHemisphere` bug fixed.
*   [x] `gen_test_data.py` saves `df_fwd.fif`.
*   [x] `test_rt_pipeline_integration` implemented and passing.
*   [x] `RtCov` and `RtInvOp` verified to work together.
