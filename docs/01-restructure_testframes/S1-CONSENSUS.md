# S1-CONSENSUS (共识文档)

## 1. 明确的需求描述
重构 `src/testframes` 目录，使其映射 `src/libraries` 的结构。

## 2. 技术实现方案
1. **目录创建**: 在 `src/testframes` 下创建类别文件夹。
2. **文件移动**: 将现有的测试文件夹移动到对应的类别文件夹中。
3. **CMake 更新**:
    * 修改 `src/testframes/CMakeLists.txt`。
    * 在每个新文件夹中创建 `CMakeLists.txt`。
4. **验证**: 运行 `cmake` 并检查生成的测试目标。

## 3. 技术约束与集成方案
* 必须保持测试的可发现性。
* 避免破坏测试内部的相对路径（大部分测试使用 `applicationDirPath()`，因此不受影响）。

## 4. 任务边界与验收标准
* **边界**: 不修改测试代码逻辑，仅修改目录结构和 CMake 组织方式。
* **验收标准**: `cmake` 无报错，`ctest` 运行正常。
