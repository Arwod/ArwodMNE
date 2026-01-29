# S2-DESIGN (架构阶段)

## 1. 映射关系设计

| 类别 (Library) | 包含的单元测试 (Tests) |
| :--- | :--- |
| **channels** | test_channel_management, test_sensorSet, test_montage_distance_preservation |
| **connectivity** | test_connectivity, test_spectral_connectivity, test_granger_causality |
| **decoding** | test_decoding, test_csp_properties |
| **fiff** | test_fiff_coord_trans, test_fiff_rwr, test_fiff_mne_types_io, test_fiff_cov, test_fiff_digitizer, test_edf2fiff_rwr |
| **fwd** | test_mne_forward_solution, test_forward_solution_linearity, test_forward_solution_processing |
| **inverse** | test_inverse_linearity, test_inverse_operator, test_dipole_fit, test_mixed_norm, test_rap_music, test_resolution_analysis, test_sparse_reconstruction, test_sparse_reconstruction_gamma_map, test_lcmv_constraint, test_lcmv_cov, test_beamformer_consistency, test_dics |
| **mne** | test_mne_anonymize, test_mne_msh_display_surface_set, test_mne_project_to_surface, test_coregistration, test_morphing |
| **preprocessing** | test_artifact_detection, test_ecg_eog_processing, test_filtering, test_fir_linear_phase, test_ica_invertibility, test_maxwell_orthogonality |
| **rtprocessing** | test_rt_pipeline_integration, test_hpiFit, test_hpiDataUpdater, test_hpiFit_integration, test_hpiModelParameter |
| **simulation** | test_simulation_statistical_consistency, test_noise_addition, test_signalModel |
| **stats** | test_stats, test_parametric_tests, test_permutation_properties, test_correction_properties |
| **tfr** | test_tfr_energy_conservation, test_tfr_stockwell, test_tfr_multitaper, test_csd_hermitian, test_psd_nonnegativity, test_source_psd |
| **utils** | test_utils_circularbuffer, test_interpolation, test_geometryinfo, test_data_roundtrip, test_data_io_roundtrip, test_data_compatibility, test_verification, test_gap_filling |

## 2. CMake 组织架构

### 2.1 根级 CMake (src/testframes/CMakeLists.txt)
将现有的平铺 `add_subdirectory` 替换为按类别加载：
```cmake
add_subdirectory(channels)
add_subdirectory(connectivity)
...
```

### 2.2 类别级 CMake (src/testframes/category/CMakeLists.txt)
每个目录下创建一个 `CMakeLists.txt`：
```cmake
add_subdirectory(test_a)
add_subdirectory(test_b)
```

## 3. 迁移影响分析
由于 MNE-CPP 的测试数据通常是通过相对路径（如 `../resources/data`）访问的，而测试二进制文件仍然输出到统一的 `out/Release/tests` 目录（由根级 `set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ...)` 决定），因此文件移动**不会**破坏测试运行时的数据访问。
