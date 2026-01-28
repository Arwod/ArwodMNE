# S1-CONSENSUS (共识文档)

## 1. 明确的需求描述
* 目标：将单元测试失败数量从 19 降至 0。
* 重点修复模块：`test_fiff_coord_trans`, `test_coregistration`, `test_fiff_stream` 等。

## 2. 技术实现方案
* **环境准备**: 确保 `tools/download_data.bat` (或 shell 等价脚本) 运行完成，测试数据放置在 `resources/data`。
* **逻辑修正**: 针对 Eigen 5.0.0-dev 的兼容性修复。
* **调试与修复**: 
    * 使用调试信息定位 SIGSEGV 位置。
    * 检查 `FiffStream` 和 `FiffCoordTrans` 中的资源管理。

## 3. 技术约束与集成方案
* 必须保持对 Qt 5.10+ 和 Qt 6 的双重兼容。
* 遵循 MNE-CPP 的代码规范（使用 `fiff` 库提供的 I/O 接口）。

## 4. 任务边界与验收标准
* **边界**: 仅修复已有的单元测试用例，不增加新的测试逻辑，除非是为了复现 bug。
* **验收标准**: 执行 `ctest` 结果为 100% 通过。
