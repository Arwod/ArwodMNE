# Phase 10: Decoding Foundation (CSP) - S1 Alignment

## 1. 项目背景
本项目是 MNE-CPP 移植计划的一部分，旨在将 Python 版 MNE 的核心功能移植到 C++ 环境中。当前处于 **Phase 10**，目标是建立基础的解码（Decoding）能力，填补 C++ 版本在 BCI（脑机接口）领域的空白。

## 2. 任务目标
本阶段的核心任务是创建一个新的 C++ 库模块 `libraries/decoding`，并在其中实现 **CSP (Common Spatial Patterns)** 算法。

### 核心需求
1.  **新建模块**: 在 `src/libraries` 下创建 `decoding` 目录及相应的构建配置（CMakeLists.txt）。
2.  **实现 CSP 类**:
    *   提供 `fit(data, labels)` 接口：计算空间滤波器。
    *   提供 `transform(data)` 接口：应用滤波器并提取特征（对数方差）。
    *   支持二分类任务。
3.  **算法对齐**: 算法逻辑需与 `mne-python` 的 `mne.decoding.CSP` 保持一致。
4.  **验证**: 使用模拟数据或标准数据集验证 C++ 实现与 Python 实现的一致性。

## 3. 需求理解与边界确认

### 3.1 输入/输出定义
*   **输入**:
    *   `fit`: 3D 矩阵 (Epochs x Channels x Times) 或协方差矩阵列表，以及对应的标签 (Labels)。
    *   `transform`: 3D 矩阵 (Epochs x Channels x Times)。
*   **输出**:
    *   `fit`: 训练好的 CSP 对象（包含滤波器 `filters_`, 模式 `patterns_`）。
    *   `transform`: 2D 特征矩阵 (Epochs x Components)。

### 3.2 现有架构分析
*   **依赖库**: 必须依赖 `Eigen` 进行矩阵运算（特征值分解）。
*   **数据结构**: 应复用 `FIFFLIB` 中的数据结构（如 `FiffEvoked`, `FiffCov`）或直接使用 `Eigen::MatrixXd`。考虑到 CSP 通常处理 Epochs 数据，可能需要定义或复用适合 3D 数据的结构，或者简化为处理 `std::vector<MatrixXd>`。
*   **构建系统**: 需集成到现有的 `qmake` 或 `cmake` 系统中（项目似乎混用了两者，但新模块应优先支持 CMake）。

### 3.3 边界限制 (Scope)
*   **包含**:
    *   基础 CSP 算法 (Common Spatial Patterns)。
    *   二分类支持。
    *   基于 `Eigen` 的广义特征值分解。
*   **不包含**:
    *   多分类 CSP (One-vs-Rest) - 暂不实现，但需预留接口。
    *   正则化 CSP (Regularized CSP)。
    *   CSP 之外的其他解码算法 (如 SPoC, FilterBank CSP)。
    *   与 GUI 的集成（仅实现算法库）。

## 4. 关键决策点 (Questions & Decisions)

### Q1: 数据输入格式
MNE-CPP 中缺乏统一的 "Epochs" C++ 类（类似 Python 的 `mne.Epochs`）。
*   **方案 A**: 定义一个新的 `Epochs` 类。
*   **方案 B**: 使用 `std::vector<Eigen::MatrixXd>` 作为输入，每个矩阵代表一个 Epoch (Channels x Times)。
*   **决策**: 采用 **方案 B**。这是最轻量且灵活的方式，便于与其他模块集成。

### Q2: 协方差计算
CSP 依赖于协方差矩阵计算。
*   **方案 A**: 复用 `libraries/connectivity` 或 `libraries/utils` 中的协方差计算逻辑。
*   **方案 B**: 在 CSP 类内部实现简单的协方差计算。
*   **决策**: 优先 **复用** 现有的协方差计算逻辑。如果现有逻辑不适配，则在 `decoding` 模块内部实现辅助函数。

### Q3: 命名空间
*   **决策**: 新代码应位于 `DECODINGLIB` 命名空间下。

## 5. 下一步计划
1.  确认上述理解无误。
2.  进入 **S2: Architect** 阶段，设计类结构和接口。
