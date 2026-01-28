# S6-TODO (待办事项与配置指引)

## 1. 待办事项
* **[建议] 持续整理**: 未来新增单元测试时，请务必放入对应的类别目录中。
* **[建议] 跨模块测试**: 如果存在明显的跨模块集成测试，建议在 `src/testframes` 下新建 `integration` 目录存放。

## 2. 操作指引
* **新增测试步骤**:
    1. 确定测试归属的库类别（如 `fiff`）。
    2. 在 `src/testframes/fiff/` 下创建新文件夹。
    3. 在 `src/testframes/fiff/CMakeLists.txt` 中添加 `add_subdirectory(your_new_test)`。
* **重新构建**:
    ```bash
    cd build
    cmake ../src -DBUILD_TESTS=ON
    make -j8
    ```
