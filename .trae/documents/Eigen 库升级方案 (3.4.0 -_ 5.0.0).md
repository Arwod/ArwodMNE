## Eigen 库升级方案 (从 3.4.0 升级至 5.0.0)

目前项目使用的 Eigen 版本为 **3.4.0**，存放于 `src/external/eigen-3.4.0`。根据调研，Eigen 已于 2025 年 9 月发布了最新的稳定版本 **5.0.0**，该版本引入了语义化版本控制（Semantic Versioning）并提升了性能和 C++14/17 的支持。

### **1. 版本信息对比**
- **当前版本**: 3.4.0 (位于 [eigen-3.4.0](file:///Users/eric/Public/work/code/mne-project/ArwodMNE/src/external/eigen-3.4.0))
- **目标版本**: 5.0.0 (最新稳定版，发布日期 2025-09-30)
- **环境要求**: Eigen 5.0 要求 C++14 标准。当前项目已配置为 **C++17** ([CMakeLists.txt:L8](file:///Users/eric/Public/work/code/mne-project/ArwodMNE/src/CMakeLists.txt#L8))，完全满足要求。

### **2. 升级步骤**

#### **第一阶段：获取与解压**
1. 从 [Eigen GitLab 官方仓库](https://gitlab.com/libeigen/eigen) 下载 5.0.0 版本的源码包。
2. 将源码解压至新目录 `src/external/eigen-5.0.0`。
   - **注意**: 建议保留原目录 `eigen-3.4.0` 直到升级验证完成。

#### **第二阶段：构建配置更新**
1. 修改 [src/external/CMakeLists.txt](file:///Users/eric/Public/work/code/mne-project/ArwodMNE/src/external/CMakeLists.txt)：
   - 将 `add_subdirectory(eigen-3.4.0/)` 替换为 `add_subdirectory(eigen-5.0.0/)`。

#### **第三阶段：代码兼容性检查**
虽然初步搜索未发现核心库中使用受影响的 API，但在编译过程中需重点关注以下 Eigen 5.0 的重大变化：
- **命名空间变更**: `Eigen::all` 和 `Eigen::last` 已移动至 `Eigen::placeholders::all` 和 `Eigen::placeholders::last`。
- **随机数生成**: 随机数生成行为有所变化，若有测试依赖于特定的随机序列，可能需要调整。

#### **第四阶段：验证与交付**
1. **清理并重新编译**:
   ```bash
   rm -rf build
   cmake -S src -B build -DBUILD_TESTS=ON
   cmake --build build --parallel 8
   ```
2. **运行全量测试**:
   使用已有的 [run_tests.py](file:///Users/eric/Public/work/code/mne-project/ArwodMNE/run_tests.py) 脚本运行所有 42 个单元测试，确保数值计算结果的一致性。
3. **清理旧版本**:
   测试全部通过后，删除 `src/external/eigen-3.4.0` 目录。

### **3. 风险评估**
- **风险等级**: **低**。Eigen 是纯头文件库，替换过程不涉及复杂的链接问题。
- **潜在影响**: 主要在于数值计算的极小差异或特定 API 的废弃警告。通过现有的端到端验证测试（Task 18.4）可以有效覆盖这些风险。

是否按照此方案开始执行升级操作？