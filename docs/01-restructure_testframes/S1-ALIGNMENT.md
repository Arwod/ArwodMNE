# S1-ALIGNMENT (对齐阶段)

## 1. 项目上下文分析
* **当前结构**: `src/testframes` 下所有单元测试目录均为平铺状态，缺乏层次。
* **目标结构**: `src/libraries` 具有清晰的模块化目录（如 `fiff`, `mne`, `inverse` 等）。
* **技术栈**: CMake 构建系统。

## 2. 需求理解确认
* **原始需求**: 将 `src/testframes` 下的单元测试按照 `src/libraries` 的目录结构进行重组。
* **边界确认**:
    * 仅涉及 `src/testframes` 目录下的子目录移动。
    * 需要同步更新相关的 `CMakeLists.txt` 文件以确保构建不受影响。
* **需求理解**:
    * 需要将现有的 60+ 个测试目录映射到约 13 个库类别中。
    * 重组后，`src/testframes` 将只包含类别子目录（如 `fiff/`, `mne/`），而具体的测试将位于这些子目录内。

## 3. 智能决策与疑问澄清
* **决策点 1**: 如何处理跨模块的测试？
    * **决策**: 根据测试名称的主要前缀和其链接的主要库进行归类。例如 `test_fiff_xxx` 归入 `fiff`。
* **决策点 2**: 是否需要为每个类别创建新的 `CMakeLists.txt`？
    * **决策**: 是。在每个类别目录下创建一个简单的 `CMakeLists.txt`，用于 `add_subdirectory` 该类别下的所有测试。

## 4. 验收标准
* `src/testframes` 目录结构与 `src/libraries` 逻辑一致。
* 项目 CMake 配置成功，所有测试目标仍然可用。
* `ctest` 能够正常发现并运行所有移动后的测试。
