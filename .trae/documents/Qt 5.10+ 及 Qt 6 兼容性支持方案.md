# Qt 5.10+ 及 Qt 6 兼容性修改方案 (针对 Qt 6 环境优化)

虽然您本地仅使用 Qt 6，但为了满足项目对 Qt 5.10+ 的兼容性要求，我们将对 CMake 配置进行标准化。这样既能保证在您的 Qt 6 环境下编译通过，也能确保代码在 Qt 5 环境下的理论兼容性。

## 1. 全局 CMake 配置 (src/CMakeLists.txt)
- 将 `cmake_minimum_required` 设置为 **3.16**。
- 优化 Qt 查找逻辑，确保优先使用 Qt 6，并定义版本变量供子项目使用。

## 2. 修复硬编码的 Qt 6 引用
- 遍历所有单元测试和库的 `CMakeLists.txt`。
- 将硬编码的 `find_package(Qt6 ...)` 替换为版本无关的 `find_package(QT NAMES Qt6 Qt5 ...)`。
- 将 `Qt6::Core`、`Qt6::Test` 等链接库替换为 `Qt${QT_VERSION_MAJOR}::Core` 等变量形式。

## 3. 模块化兼容性处理
- **mne_stats**: 保持对 `Core5Compat` 的支持（仅在 Qt 6 时加载）。
- **mne_disp**: 处理 `OpenGLWidgets`。在 Qt 6 中显式链接 `Qt6::OpenGLWidgets`，在 Qt 5 中则回退到 `Qt5::Widgets` 或 `Qt5::OpenGL`。

## 4. 源码层面
- 保持现有的 `#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)` 宏检查，确保 `endl` 等符号在不同版本下正常工作。

## 5. 验证 (您的本地环境)
- 执行全量编译：`cmake --build build`。
- 运行所有单元测试：`python3 run_tests.py`。
- 确保 43 个测试全部通过。

完成后，项目将具备“一次编写，两代支持”的能力。是否开始执行？