# Phase 4: Statistics Acceptance Criteria

This document defines the tasks and acceptance criteria for Phase 4: Basic Statistics.

## Overview
Implement basic statistical functions commonly used in EEG/MEG analysis, such as T-tests and Multiple Comparison Corrections (Bonferroni, FDR).

## Tasks

### Task 4.1: T-Test
*   **Goal**: Implement 1-sample and 2-sample (independent/related) T-tests.
*   **Target**: `libraries/stats/ttest.h` (New Library: `mne_stats`)
*   **Details**:
    *   `ttest_1samp`: Test if mean of sample is different from popmean (usually 0).
    *   `ttest_ind`: Test if means of two independent samples are different.
    *   Return T-values and P-values (using Student's t-distribution CDF).
    *   *Note*: P-value calculation might require `boost::math` or a simple implementation of the CDF/Beta function if we want to avoid Boost. For now, simple approximation or lookup might suffice, or using `std::beta` if available (C++17 has `std::beta`? No, C++17 has `std::beta` in `<cmath>` but it's Beta function, not distribution).
    *   Actually, let's implement a simple T-distribution CDF or use an existing lightweight math lib if needed. Or just return T-values first.
- [x] Implement `ttest_1samp` (T-value computation)
- [x] Implement `ttest_ind` (T-value computation)
- [x] (Optional) P-value computation

### Task 4.2: Multiple Comparison Correction
*   **Goal**: Implement Bonferroni and FDR (False Discovery Rate) correction.
*   **Target**: `libraries/stats/correction.h` (New)
*   **Details**:
    *   `bonferroni_correction`: $p_{corrected} = p * n$
    *   `fdr_correction`: Benjamini-Hochberg procedure.
- [x] Implement `bonferroni_correction`
- [x] Implement `fdr_correction`

## Verification Instructions

1.  **Generate Data**: Update `tests/verification_data/gen_test_data.py`:
    *   Generate two groups of random data with known mean difference.
    *   Compute T-values and P-values using `scipy.stats`.
    *   Save results.
2.  **Run Tests**:
    *   `test_verification` runs C++ T-test.
    *   Compare T-values against SciPy.

## Definition of Done
*   [x] `libraries/stats` created and compiled.
*   [x] T-tests implemented and verified.
*   [x] Correction methods implemented and verified.
