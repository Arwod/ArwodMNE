# MNE Python到C++核心算法移植需求文档

## 介绍

本文档定义了将MNE Python版本的核心算法功能移植到MNE C++版本的需求。MNE Python是社区活跃的版本，包含了最新的算法实现和功能，而MNE C++版本需要这些算法来保持功能完整性和性能优势。

## 术语表

- **MNE_Python**: MNE项目的Python实现版本，位于mne-python目录
- **MNE_CPP**: MNE项目的C++实现版本，位于ArwodMNE目录  
- **Core_Algorithm**: 核心数学算法，包括信号处理、源定位、时频分析等
- **Beamformer**: 波束成形器算法，用于源定位
- **TFR**: 时频表示(Time-Frequency Representation)算法
- **ICA**: 独立成分分析(Independent Component Analysis)算法
- **Preprocessing**: 预处理算法模块
- **Source_Localization**: 源定位算法模块
- **Connectivity**: 连接性分析算法模块
- **Statistics**: 统计分析算法模块

## 需求

### 需求1: 时频分析算法完整移植

**用户故事:** 作为一个神经科学研究者，我希望在C++版本中使用完整的时频分析功能，以便获得与Python版本相同的分析结果和更好的性能。

#### 验收标准

1. WHEN 用户调用tfr_morlet时，THE TFR_System SHALL 提供Morlet小波时频分解
2. WHEN 用户使用tfr_multitaper时，THE TFR_System SHALL 实现多锥度时频分析
3. WHEN 用户调用tfr_stockwell时，THE TFR_System SHALL 提供Stockwell变换
4. WHEN 用户计算交叉谱密度时，THE TFR_System SHALL 支持csd_fourier、csd_morlet、csd_multitaper方法
5. WHEN 用户需要功率谱密度时，THE TFR_System SHALL 提供psd_array_welch和psd_array_multitaper
6. WHEN 用户使用短时傅里叶变换时，THE TFR_System SHALL 实现stft和istft函数
7. THE TFR_System SHALL 支持AverageTFR、EpochsTFR、RawTFR等数据结构
8. WHEN 用户需要DPSS窗函数时，THE TFR_System SHALL 提供dpss_windows计算
9. WHEN 用户计算AR模型时，THE TFR_System SHALL 实现fit_iir_model_raw功能

### 需求2: 波束成形器算法完整实现

**用户故事:** 作为一个MEG/EEG分析专家，我希望C++版本包含完整的波束成形器算法套件，以便进行精确的源定位分析。

#### 验收标准

1. WHEN 用户使用make_lcmv时，THE Beamformer_System SHALL 创建LCMV空间滤波器
2. WHEN 用户调用apply_lcmv系列函数时，THE Beamformer_System SHALL 支持apply_lcmv、apply_lcmv_cov、apply_lcmv_epochs、apply_lcmv_raw
3. WHEN 用户使用make_dics时，THE Beamformer_System SHALL 创建DICS波束成形器
4. WHEN 用户调用apply_dics系列函数时，THE Beamformer_System SHALL 支持apply_dics、apply_dics_csd、apply_dics_epochs、apply_dics_tfr_epochs
5. WHEN 用户需要RAP-MUSIC算法时，THE Beamformer_System SHALL 实现rap_music和trap_music
6. WHEN 用户计算分辨率矩阵时，THE Beamformer_System SHALL 提供make_lcmv_resolution_matrix
7. THE Beamformer_System SHALL 支持Beamformer类和read_beamformer功能
8. WHEN 用户指定权重归一化时，THE Beamformer_System SHALL 支持unit-noise-gain和neural-activity-index方法

### 需求3: 预处理算法完整套件

**用户故事:** 作为一个信号处理工程师，我希望C++版本提供完整的预处理工具链，以便进行高质量的数据清理和准备。

#### 验收标准

1. WHEN 用户执行ICA分解时，THE Preprocessing_System SHALL 支持FastICA、Infomax算法和ICA类
2. WHEN 用户需要自动伪影检测时，THE Preprocessing_System SHALL 提供annotate_amplitude、annotate_muscle_zscore、annotate_movement
3. WHEN 用户处理ECG伪影时，THE Preprocessing_System SHALL 实现find_ecg_events、create_ecg_epochs、compute_proj_ecg
4. WHEN 用户处理EOG伪影时，THE Preprocessing_System SHALL 实现find_eog_events、create_eog_epochs、compute_proj_eog
5. WHEN 用户使用Maxwell滤波时，THE Preprocessing_System SHALL 实现maxwell_filter、compute_maxwell_basis、find_bad_channels_maxwell
6. WHEN 用户需要Xdawn算法时，THE Preprocessing_System SHALL 提供Xdawn类和XdawnTransformer
7. WHEN 用户进行通道插值时，THE Preprocessing_System SHALL 支持interpolate_bridged_electrodes、equalize_bads
8. WHEN 用户计算CSD时，THE Preprocessing_System SHALL 实现compute_current_source_density
9. WHEN 用户需要实时处理时，THE Preprocessing_System SHALL 提供oversampled_temporal_projection
10. THE Preprocessing_System SHALL 支持EOGRegression、corrmap、peak_finder等高级功能

### 需求4: 最小范数估计算法完整实现

**用户故事:** 作为一个源成像研究者，我希望C++版本包含所有最小范数估计变体，以便进行全面的源定位分析。

#### 验收标准

1. WHEN 用户创建逆算子时，THE MinimumNorm_System SHALL 提供make_inverse_operator和InverseOperator类
2. WHEN 用户应用逆解时，THE MinimumNorm_System SHALL 支持apply_inverse、apply_inverse_cov、apply_inverse_epochs、apply_inverse_raw、apply_inverse_tfr_epochs
3. WHEN 用户需要分辨率分析时，THE MinimumNorm_System SHALL 实现make_inverse_resolution_matrix、get_cross_talk、get_point_spread
4. WHEN 用户计算源功率谱时，THE MinimumNorm_System SHALL 提供compute_source_psd、compute_source_psd_epochs
5. WHEN 用户分析诱发功率时，THE MinimumNorm_System SHALL 实现source_induced_power、source_band_induced_power
6. WHEN 用户需要SNR估计时，THE MinimumNorm_System SHALL 提供estimate_snr功能
7. THE MinimumNorm_System SHALL 支持read_inverse_operator、write_inverse_operator、prepare_inverse_operator
8. WHEN 用户计算秩时，THE MinimumNorm_System SHALL 实现compute_rank_inverse
9. THE MinimumNorm_System SHALL 支持所有INVERSE_METHODS常量定义

### 需求5: 统计分析算法完整集成

**用户故事:** 作为一个统计分析专家，我希望C++版本包含强大的统计工具，以便进行严格的统计推断。

#### 验收标准

1. WHEN 用户执行聚类置换检验时，THE Statistics_System SHALL 实现permutation_cluster_test、permutation_cluster_1samp_test
2. WHEN 用户进行时空聚类分析时，THE Statistics_System SHALL 提供spatio_temporal_cluster_test、spatio_temporal_cluster_1samp_test
3. WHEN 用户需要多重比较校正时，THE Statistics_System SHALL 实现fdr_correction、bonferroni_correction
4. WHEN 用户进行参数检验时，THE Statistics_System SHALL 提供f_oneway、f_mway_rm、ttest_1samp_no_p、ttest_ind_no_p
5. WHEN 用户需要置换检验时，THE Statistics_System SHALL 实现permutation_t_test、bootstrap_confidence_interval
6. WHEN 用户进行回归分析时，THE Statistics_System SHALL 提供linear_regression、linear_regression_raw
7. WHEN 用户处理ERP数据时，THE Statistics_System SHALL 支持erp模块功能
8. WHEN 用户需要邻接矩阵时，THE Statistics_System SHALL 实现combine_adjacency
9. THE Statistics_System SHALL 提供summarize_clusters_stc用于结果总结

### 需求6: 解码和机器学习算法实现

**用户故事:** 作为一个机器学习研究者，我希望C++版本提供完整的解码算法，以便进行脑机接口和模式识别研究。

#### 验收标准

1. WHEN 用户使用CSP算法时，THE Decoding_System SHALL 实现CSP和SPoC类
2. WHEN 用户需要时间解码时，THE Decoding_System SHALL 提供SlidingEstimator、GeneralizingEstimator
3. WHEN 用户进行空间滤波时，THE Decoding_System SHALL 实现SpatialFilter、UnsupervisedSpatialFilter
4. WHEN 用户使用SSD算法时，THE Decoding_System SHALL 提供SSD类
5. WHEN 用户需要感受野分析时，THE Decoding_System SHALL 实现ReceptiveField
6. WHEN 用户进行时频解码时，THE Decoding_System SHALL 提供TimeFrequency类
7. WHEN 用户使用EMS算法时，THE Decoding_System SHALL 实现EMS和compute_ems
8. WHEN 用户需要数据变换时，THE Decoding_System SHALL 提供Scaler、Vectorizer、FilterEstimator、PSDEstimator
9. THE Decoding_System SHALL 支持BaseEstimator、LinearModel、TransformerMixin基类
10. WHEN 用户进行交叉验证时，THE Decoding_System SHALL 实现cross_val_multiscore

### 需求7: 稀疏逆解算法实现

**用户故事:** 作为一个高级源成像研究者，我希望C++版本支持稀疏逆解算法，以便进行更精确的源定位。

#### 验收标准

1. WHEN 用户使用Gamma-MAP时，THE InverseSparse_System SHALL 实现gamma_map算法
2. WHEN 用户需要混合范数解时，THE InverseSparse_System SHALL 提供mixed_norm算法
3. WHEN 用户进行时频稀疏重建时，THE InverseSparse_System SHALL 实现tf_mixed_norm
4. WHEN 用户从偶极子创建源估计时，THE InverseSparse_System SHALL 提供make_stc_from_dipoles
5. THE InverseSparse_System SHALL 支持L1和L2正则化的组合优化

### 需求8: 仿真算法完整实现

**用户故事:** 作为一个方法学研究者，我希望C++版本提供完整的数据仿真功能，以便进行算法验证和测试。

#### 验收标准

1. WHEN 用户仿真诱发响应时，THE Simulation_System SHALL 实现simulate_evoked、add_noise
2. WHEN 用户仿真原始数据时，THE Simulation_System SHALL 提供simulate_raw、add_ecg、add_eog、add_chpi
3. WHEN 用户仿真源活动时，THE Simulation_System SHALL 实现simulate_stc、simulate_sparse_stc
4. WHEN 用户需要源仿真器时，THE Simulation_System SHALL 提供SourceSimulator类
5. WHEN 用户选择标签内源时，THE Simulation_System SHALL 实现select_source_in_label
6. THE Simulation_System SHALL 支持metrics模块用于仿真质量评估

### 需求9: 正向建模算法完整实现

**用户故事:** 作为一个生物物理建模专家，我希望C++版本提供精确的正向建模工具，以便计算准确的导联场矩阵。

#### 验收标准

1. WHEN 用户创建正向解时，THE Forward_System SHALL 实现make_forward_solution、make_forward_dipole
2. WHEN 用户应用正向解时，THE Forward_System SHALL 提供apply_forward、apply_forward_raw
3. WHEN 用户处理正向解时，THE Forward_System SHALL 支持read_forward_solution、write_forward_solution
4. WHEN 用户转换正向解时，THE Forward_System SHALL 实现convert_forward_solution
5. WHEN 用户限制正向解时，THE Forward_System SHALL 提供restrict_forward_to_label、restrict_forward_to_stc
6. WHEN 用户计算先验时，THE Forward_System SHALL 实现compute_depth_prior、compute_orient_prior
7. WHEN 用户需要场映射时，THE Forward_System SHALL 提供make_field_map
8. THE Forward_System SHALL 支持Forward类和相关的内部函数
9. WHEN 用户平均正向解时，THE Forward_System SHALL 实现average_forward_solutions

### 需求10: 数据输入输出完整支持

**用户故事:** 作为一个数据分析工程师，我希望C++版本支持所有主要的数据格式，以便处理各种来源的神经电生理数据。

#### 验收标准

1. WHEN 用户读取原始数据时，THE IO_System SHALL 支持read_raw_fif、read_raw_edf、read_raw_brainvision等多种格式
2. WHEN 用户处理数组数据时，THE IO_System SHALL 提供RawArray类
3. WHEN 用户需要数据匿名化时，THE IO_System SHALL 实现anonymize_info
4. WHEN 用户连接数据时，THE IO_System SHALL 提供concatenate_raws
5. WHEN 用户读取事件数据时，THE IO_System SHALL 支持read_epochs_eeglab、read_epochs_fieldtrip等
6. WHEN 用户处理诱发数据时，THE IO_System SHALL 支持read_evoked_besa、read_evoked_fieldtrip等
7. THE IO_System SHALL 提供BaseRaw、Raw等基础数据类
8. WHEN 用户处理信息结构时，THE IO_System SHALL 实现read_info、write_info
9. WHEN 用户处理基准点时，THE IO_System SHALL 支持read_fiducials、write_fiducials

### 需求11: 通道和蒙太奇管理系统

**用户故事:** 作为一个电极配置专家，我希望C++版本提供完整的通道管理和蒙太奇功能，以便处理各种电极配置。

#### 验收标准

1. WHEN 用户创建数字化蒙太奇时，THE Channels_System SHALL 实现make_dig_montage、DigMontage类
2. WHEN 用户使用标准蒙太奇时，THE Channels_System SHALL 提供make_standard_montage、get_builtin_montages
3. WHEN 用户读取自定义蒙太奇时，THE Channels_System SHALL 支持read_custom_montage、read_dig_*系列函数
4. WHEN 用户处理通道邻接性时，THE Channels_System SHALL 实现find_ch_adjacency、get_builtin_ch_adjacencies
5. WHEN 用户管理通道时，THE Channels_System SHALL 提供combine_channels、equalize_channels、rename_channels
6. WHEN 用户创建布局时，THE Channels_System SHALL 实现make_eeg_layout、make_grid_layout、Layout类
7. THE Channels_System SHALL 支持通道选择功能make_1020_channel_selections
8. WHEN 用户处理坏通道时，THE Channels_System SHALL 实现unify_bad_channels

### 需求12: 可视化系统完整实现

**用户故事:** 作为一个数据可视化专家，我希望C++版本提供强大的可视化功能，以便创建高质量的科学图表。

#### 验收标准

1. WHEN 用户绘制地形图时，THE Visualization_System SHALL 实现plot_topomap、plot_evoked_topomap
2. WHEN 用户可视化原始数据时，THE Visualization_System SHALL 提供plot_raw、plot_raw_psd
3. WHEN 用户绘制诱发响应时，THE Visualization_System SHALL 实现plot_evoked、plot_compare_evokeds
4. WHEN 用户可视化时期数据时，THE Visualization_System SHALL 提供plot_epochs、plot_epochs_image
5. WHEN 用户进行3D可视化时，THE Visualization_System SHALL 实现plot_source_estimates、Brain类
6. WHEN 用户绘制传感器时，THE Visualization_System SHALL 提供plot_sensors、plot_montage
7. WHEN 用户可视化连接性时，THE Visualization_System SHALL 实现circular_layout
8. THE Visualization_System SHALL 支持多种后端backends模块
9. WHEN 用户绘制ICA组件时，THE Visualization_System SHALL 实现plot_ica_components、plot_ica_sources

### 需求13: 连接性分析算法实现

**用户故事:** 作为一个网络神经科学研究者，我希望C++版本提供全面的连接性分析工具，以便研究大脑网络。

#### 验收标准

1. WHEN 用户计算相干性时，THE Connectivity_System SHALL 提供谱相干性和虚相干性算法
2. WHEN 用户需要相位锁定值时，THE Connectivity_System SHALL 计算PLV和wPLI指标
3. WHEN 用户分析因果关系时，THE Connectivity_System SHALL 实现Granger因果性
4. WHEN 用户计算相位滞后指数时，THE Connectivity_System SHALL 提供PLI和dwPLI
5. THE Connectivity_System SHALL 支持源空间和传感器空间连接性
6. WHEN 用户需要统计显著性时，THE Connectivity_System SHALL 提供置换检验
7. THE Connectivity_System SHALL 实现现有的connectivity模块功能

### 需求14: 滤波算法优化增强

**用户故事:** 作为一个实时数据处理开发者，我希望C++版本提供高性能的滤波算法，以便进行实时信号处理。

#### 验收标准

1. WHEN 用户应用FIR滤波器时，THE Filter_System SHALL 提供线性相位响应
2. WHEN 用户使用IIR滤波器时，THE Filter_System SHALL 支持Butterworth和Chebyshev设计
3. WHEN 用户需要陷波滤波时，THE Filter_System SHALL 去除特定频率干扰
4. WHEN 用户进行希尔伯特变换时，THE Filter_System SHALL 计算解析信号
5. THE Filter_System SHALL 支持零相位滤波和因果滤波
6. WHEN 用户处理实时数据时，THE Filter_System SHALL 提供低延迟滤波
7. THE Filter_System SHALL 增强现有的rtprocessing模块滤波功能

### 需求15: 数据结构兼容性和互操作性

**用户故事:** 作为一个软件集成工程师，我希望C++版本的数据结构与Python版本完全兼容，以便实现无缝的数据交换。

#### 验收标准

1. WHEN 用户保存数据时，THE DataStructure_System SHALL 使用与Python兼容的格式
2. WHEN 用户加载Python生成的文件时，THE DataStructure_System SHALL 正确解析所有字段
3. WHEN 用户访问元数据时，THE DataStructure_System SHALL 提供相同的信息结构
4. WHEN 用户进行数据转换时，THE DataStructure_System SHALL 保持数值精度
5. THE DataStructure_System SHALL 支持所有MNE数据类型
6. WHEN 用户需要互操作性时，THE DataStructure_System SHALL 提供Python绑定接口
7. THE DataStructure_System SHALL 确保Epochs、Evoked、Raw等核心类的完全兼容性