# S6-TODO (待办事项与配置指引)

## 1. 待办事项
* **[建议] 持续监控**: 特别是 `test_sparse_reconstruction_gamma_map` 等属性测试，建议在 CI 中增加多次迭代运行以捕获罕见的边缘失败。

## 2. 操作指引
* **单次全量运行命令**:
  ```bash
  total=0; passed=0; failed=0; for t in out/Release/tests/test_*; do if [ -x "$t" ]; then total=$((total+1)); if "$t" > /dev/null 2>&1; then passed=$((passed+1)); else echo "FAILED: $t"; failed=$((failed+1)); fi; fi; done; echo "Passed: $passed / $total"
  ```
