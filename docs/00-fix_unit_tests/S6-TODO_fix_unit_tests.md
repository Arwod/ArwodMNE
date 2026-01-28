# S6-TODO (待办事项与配置指引)

## 1. 待办事项
* **[建议] CI 配置更新**: 确保 CI 环境在运行测试前自动执行数据下载和解压。
* **[建议] 库代码审查**: 尽管目前测试通过，建议进一步审查 `libraries/` 中是否还有其他地方使用了 Eigen 的 1D 索引（`.array()(i)`）。
* **[建议] Qt 版本测试**: 当前在 Qt 6.9.0 下验证通过，建议在 Qt 5.15 下进行回归测试。

## 2. 操作指引
* **重新运行所有测试**:
  ```bash
  for t in out/Release/tests/test_*; do if [ -x "$t" ]; then "$t"; fi; done
  ```
* **更新数据**:
  如果未来增加了新的测试用例，请运行 `bash tools/download_data.bat all`。
