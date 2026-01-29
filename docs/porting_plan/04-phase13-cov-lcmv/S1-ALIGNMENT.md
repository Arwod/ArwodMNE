# Phase 13: Covariance & LCMV Refactoring - S1 Alignment

## 1. 项目背景
Phase 13 旨在增强 `libraries/inverse` 库中的协方差矩阵计算 (`Covariance`) 和 LCMV 波束形成器 (`LCMV`)。目前这些类的实现较为基础，缺乏 MNE-Python 中的高级功能，如自动正则化、多 Epochs 处理、以及灵活的滤波器应用接口。

## 2. 任务目标
### 核心需求
1.  **Covariance 类重构**:
    *   **对象化**: `Covariance` 应成为一个包含数据、通道名、Bad 通道、估计方法（empirical, shrunk 等）的类，而不仅仅是静态函数。
    *   **Epochs 支持**: `compute_covariance` 应支持从 Epochs 列表计算。
    *   **I/O**: 支持简单的读写（至少是内部结构，未来对接 .fif）。
2.  **LCMV 类增强**:
    *   **Input**: 接受 `Covariance` 对象而非裸矩阵。
    *   **Max-Power Orientation**: 实现 `pick_ori="max-power"`，即选择功率最大的方向。
    *   **Weight Normalization**: 支持 `weight_norm` (unit-noise-gain)。
    *   **API 对齐**: 拆分为 `make_lcmv` (计算 filters) 和 `apply_lcmv` (应用 filters)，并引入 `Beamformer` 类（可选，或作为 LCMV 的一部分）。

### 对应 Python 功能
*   `mne.compute_covariance`
*   `mne.beamformer.make_lcmv`
*   `mne.beamformer.apply_lcmv`

## 3. 需求理解与边界确认

### 3.1 现有代码分析
*   `covariance.h` 只有静态方法 `compute_empirical`。
*   `lcmv.h` 只有静态方法 `compute_weights`，且只支持简单的 Tikhonov 正则化。
*   `dics.cpp` (Phase 12) 已经展示了更复杂的波束形成逻辑（如 CSD 处理），LCMV 应与之看齐。

### 3.2 边界限制
*   **输入**: Epochs, Evoked, ForwardSolution (或 Leadfield + SourceSpace).
*   **输出**: SourceEstimate, Covariance Matrix.
*   **不包含**:
    *   复杂的自动秩估计 (Auto Rank Estimation) - 初始版本可手动指定秩或使用简单的阈值。
    *   非常规的协方差估计器 (如 OAS, Ledoit-Wolf) - 仅实现 Empirical 和简单的 Shrunk (Diagonal loading)。

## 4. 关键决策点

### Q1: Covariance 数据结构
*   MNE-Python 的 `Covariance` 是一个 Dict-like object。
*   **决策**: C++ 中定义 `class Covariance`，包含 `Eigen::MatrixXd data` (full matrix), `std::vector<string> names`, `int nfree` (degrees of freedom), `double loglik`, `std::string method`.

### Q2: LCMV 架构
*   是否创建一个通用的 `Beamformer` 基类？
*   DICS 和 LCMV 有很多共同点（如 Leadfield 处理、正则化）。
*   **决策**: 暂时保持独立，但接口风格统一。LCMV 将返回一个 `Beamformer` 结构体/类（包含 weights, whitening matrix 等），用于后续 `apply`。

## 5. 下一步计划
1.  定义 `Covariance` 类结构。
2.  定义 `LCMV` 新接口。
3.  编写 Consensus。
